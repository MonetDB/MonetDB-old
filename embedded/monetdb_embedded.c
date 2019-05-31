/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

/*
 * H. Muehleisen, M. Raasveldt
 * Inverse RAPI
 */

#include "monetdb_config.h"

#include "monetdb_embedded.h"
#include "gdk.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_embedded.h"
#include "mtime.h"
#include "blob.h"
#include "sql_mvc.h"
#include "sql_catalog.h"
#include "sql_gencode.h"
#include "sql_scenario.h"
#include "sql_optimizer.h"
#include "rel_exp.h"
#include "rel_rel.h"
#include "rel_updates.h"
#include "monet_options.h"

static int monetdb_embedded_initialized = 0;

int
monetdb_is_initialized(void)
{
	return monetdb_embedded_initialized > 0;
}

static char*
validate_connection(monetdb_connection conn, const char* call)
{
	if (!monetdb_is_initialized())
		return createException(MAL, call, SQLSTATE(HY001) "Embedded MonetDB is not started");
	if (!MCvalid((Client) conn))
		return createException(MAL, call, SQLSTATE(HY001) "Invalid connection");
	return MAL_SUCCEED;
}

MT_Lock embedded_lock = MT_LOCK_INITIALIZER("embedded_lock");

static void monetdb_destroy_column(monetdb_column* column);

typedef struct {
	monetdb_result res;
	res_table *monetdb_resultset;
	monetdb_column **converted_columns;
} monetdb_result_internal;

char*
monetdb_connect(monetdb_connection *conn)
{
	mvc *m;
	char* msg = MAL_SUCCEED;
	Client mc = NULL;

	if (!monetdb_is_initialized()) {
		msg = createException(MAL, "embedded.monetdb_connect", SQLSTATE(42000) "Embedded MonetDB is not started");
		goto cleanup;
	}
	if (!conn) {
		msg = createException(MAL, "embedded.monetdb_connect", SQLSTATE(42000) "monetdb_connection parameter is NULL");
		goto cleanup;
	}
	mc = MCinitClient((oid) 0, 0, 0);
	if (!MCvalid(mc)) {
		msg = createException(MAL, "embedded.monetdb_connect", SQLSTATE(HY001) "Failed to initialize client");
		goto cleanup;
	}
	mc->curmodule = mc->usermodule = userModule();
	if (mc->usermodule == NULL) {
		msg = createException(MAL, "embedded.monetdb_connect", SQLSTATE(HY001) "Failed to initialize client MAL module");
		goto cleanup;
	}
	if ((msg = SQLinitClient(mc)) != MAL_SUCCEED)
		goto cleanup;
	if ((msg = getSQLContext(mc, NULL, &m, NULL)) != MAL_SUCCEED)
		goto cleanup;
	m->session->auto_commit = 1;

cleanup:
	if (msg && mc) {
		char* other = monetdb_disconnect(mc);
		if (other)
			GDKfree(other);
		*conn = NULL;
	} else if(conn)
		*conn = mc;
	return msg;
}

char*
monetdb_disconnect(monetdb_connection conn)
{
	char* msg = MAL_SUCCEED;

	if ((msg = validate_connection(conn, "embedded.monetdb_disconnect")) != MAL_SUCCEED)
		return msg;
	SQLexitClient((Client) conn);
	MCcloseClient((Client) conn);
	return MAL_SUCCEED;
}

static char*
monetdb_shutdown_internal(void)
{
	if (monetdb_embedded_initialized) {
		malEmbeddedReset();
		monetdb_embedded_initialized = 0;
	}
	return MAL_SUCCEED;
}

char*
monetdb_startup(char* dbdir, char silent, char sequential)
{
	char* msg = MAL_SUCCEED;
	monetdb_result* res = NULL;
	void* c;
	opt *set = NULL;
	int setlen;
	gdk_return gdk_res;

	MT_lock_set(&embedded_lock);
	GDKfataljumpenable = 1;
	if(setjmp(GDKfataljump) != 0) {
		msg = GDKfatalmsg;
		// we will get here if GDKfatal was called.
		if (msg == NULL)
			msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY002) "GDKfatal() with unspecified error");
		goto cleanup;
	}

	if (monetdb_embedded_initialized)
		goto cleanup;

	if (silent)
		MT_fprintf_silent(true);

	if ((setlen = mo_builtin_settings(&set)) == 0) {
		msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY001) MAL_MALLOC_FAIL);
		goto cleanup;
	}
	if ((setlen = mo_add_option(&set, setlen, opt_cmdline, "gdk_dbpath", dbdir)) == 0) {
		mo_free_options(set, setlen);
		msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY001) MAL_MALLOC_FAIL);
		goto cleanup;
	}
	if (sequential)
		setlen = mo_add_option(&set, setlen, opt_cmdline, "sql_optimizer", "sequential_pipe");
	else
		setlen = mo_add_option(&set, setlen, opt_cmdline, "sql_optimizer", "default_pipe");
	if (setlen == 0) {
		mo_free_options(set, setlen);
		msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY001) MAL_MALLOC_FAIL);
		goto cleanup;
	}
	gdk_res = GDKinit(set, setlen);
	mo_free_options(set, setlen);
	if (gdk_res == GDK_FAIL) {
		msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY002) "GDKinit() failed");
		goto cleanup;
	}

	if ((msg = malEmbeddedBoot()) != MAL_SUCCEED)
		goto cleanup;
	if (!SQLisInitialized()) {
		msg = createException(MAL, "embedded.monetdb_startup", SQLSTATE(HY002) "SQL initialization failed");
		goto cleanup;
	}

	monetdb_embedded_initialized = true;

	if ((msg = monetdb_connect(&c)) != MAL_SUCCEED)
		goto cleanup;
	GDKfataljumpenable = 0;

	// we do not want to jump after this point, since we cannot do so between threads
	// sanity check, run a SQL query
	if ((msg = monetdb_query(c, "SELECT id FROM _tables LIMIT 1;", &res, NULL, NULL)) != MAL_SUCCEED) {
		monetdb_embedded_initialized = false;
		goto cleanup;
	}
	if ((msg = monetdb_cleanup_result(c, res)) != MAL_SUCCEED) {
		monetdb_embedded_initialized = false;
		goto cleanup;
	}
	if ((msg = monetdb_disconnect(c)) != MAL_SUCCEED) {
		monetdb_embedded_initialized = false;
		goto cleanup;
	}

cleanup:
	if (msg)
		monetdb_shutdown_internal();
	MT_lock_unset(&embedded_lock);
	return msg;
}

static char*
monetdb_query_internal(monetdb_connection conn, char* query, monetdb_result** result, int64_t* affected_rows,
					   int64_t* prepare_id, char language)
{
	char* msg = MAL_SUCCEED, *commit_msg;
	Client c = (Client) conn;
	mvc* m;
	backend *b;
	char *nq, *qname = "somequery";
	size_t query_len;
	buffer query_buf;
	stream *query_stream;
	monetdb_result_internal *res_internal = NULL;

	if ((msg = validate_connection(conn, "embedded.monetdb_query_internal")) != MAL_SUCCEED)
		return msg;

	b = (backend *) c->sqlcontext;
	m = b->mvc;

	if (!query)
		return createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(42000) "Query missing");
	if (!(query_stream = buffer_rastream(&query_buf, qname)))
		return createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) "WARNING: could not setup query stream.");
	query_len = strlen(query) + 3;
	nq = GDKmalloc(query_len);
	if (!nq)
		return createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) "WARNING: could not setup query stream.");
	sprintf(nq, "%s\n;", query);

	query_buf.pos = 0;
	query_buf.len = query_len;
	query_buf.buf = nq;

	if (!(c->fdin = bstream_create(query_stream, query_len))) {
		close_stream(query_stream);
		return createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) "WARNING: could not setup query stream.");
	}
	if (bstream_next(c->fdin) < 0) {
		close_stream(query_stream);
		throw(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) "Internal error with ");
	}

	b->language = language;
	m->scanner.mode = LINE_N;
	m->scanner.rs = c->fdin;
	b->output_format = OFMT_NONE;
	m->user_id = m->role_id = USER_MONETDB;
	m->errstr[0] = '\0';

	if ((msg = MSinitClientPrg(c, "user", qname)) != MAL_SUCCEED)
		goto cleanup;
	if ((msg = SQLparser(c)) != MAL_SUCCEED)
		goto cleanup;
	if ((msg = SQLparser(c)) != MAL_SUCCEED)
		goto cleanup;

	if (prepare_id && m->emode == m_prepare)
		*prepare_id = b->q->id;

	if ((msg = SQLengine(c)) != MAL_SUCCEED)
		goto cleanup;

	if (!m->results && m->rowcnt >= 0 && affected_rows)
		*affected_rows = m->rowcnt;

	if (result) {
		res_internal = GDKzalloc(sizeof(monetdb_result_internal));
		if (!res_internal) {
			msg = createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto cleanup;
		}
		if (m->emode == m_execute)
			res_internal->res.type = (m->results) ? (char) Q_TABLE : (char) Q_UPDATE;
		else if (m->emode & m_prepare)
			res_internal->res.type = (char) Q_PREPARE;
		else
			res_internal->res.type = (char) m->type;
		res_internal->res.id = (size_t) m->last_id;
		*result  = (monetdb_result*) res_internal;
		m->reply_size = -2; /* do not clean up result tables */

		if (m->results) {
			res_internal->res.ncols = m->results->nr_cols;
			if (m->results->nr_cols > 0 && m->results->order) {
				BAT* bb = BATdescriptor(m->results->order);
				if (!bb) {
					msg = createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY002) RUNTIME_OBJECT_MISSING);
					goto cleanup;
				}
				res_internal->res.nrows = BATcount(bb);
				BBPunfix(bb->batCacheid);
			}
			res_internal->monetdb_resultset = m->results;
			res_internal->converted_columns = GDKzalloc(sizeof(monetdb_column*) * res_internal->res.ncols);
			if (!res_internal->converted_columns) {
				msg = createException(MAL, "embedded.monetdb_query_internal", SQLSTATE(HY001) MAL_MALLOC_FAIL);
				goto cleanup;
			}
			m->results = NULL;
		}
	}

cleanup:
	GDKfree(nq);
	MSresetInstructions(c->curprg->def, 1);
	bstream_destroy(c->fdin);
	c->fdin = NULL;

	commit_msg = SQLautocommit(m); //need always to commit even if msg is set
	if ((msg != MAL_SUCCEED || commit_msg != MAL_SUCCEED)) {
		if (res_internal) {
			char* other = monetdb_cleanup_result(conn, (monetdb_result*) res_internal);
			if (other)
				GDKfree(other);
		}
		if (result)
			*result = NULL;
		if (msg == MAL_SUCCEED) //if the error happened in the autocommit, set it as the returning error message
			msg = commit_msg;
		else if(commit_msg) //otherwise if msg is set, discard commit_msg
			GDKfree(commit_msg);
	}
	return msg;
}

char*
monetdb_clear_prepare(monetdb_connection conn, int64_t id)
{
	char query[100];

	sprintf(query, "release "LLFMT, id);
	return(monetdb_query_internal(conn, query, NULL, NULL, NULL, 'X'));
}

char*
monetdb_send_close(monetdb_connection conn, int64_t id)
{
	char query[100];

	if (id < 1)
		return createException(MAL, "embedded.monetdb_send_close", SQLSTATE(42000) "Invalid value, must be positive.");
	sprintf(query, "close "LLFMT, id);
	return(monetdb_query_internal(conn, query, NULL, NULL, NULL, 'X'));
}

char*
monetdb_set_autocommit(monetdb_connection conn, char value)
{
	char query[100];

	if (value != 1 && value != 0)
		return createException(MAL, "embedded.monetdb_set_autocommit", SQLSTATE(42000) "Invalid value, need 0 or 1.");
	sprintf(query, "auto_commit %i", value);
	return(monetdb_query_internal(conn, query, NULL, NULL, NULL, 'X'));
}

char*
monetdb_query(monetdb_connection conn, char* query, monetdb_result** result, int64_t* affected_rows, int64_t* prepare_id)
{
	return(monetdb_query_internal(conn, query, result, affected_rows, prepare_id, 'S'));
}

char*
monetdb_append(monetdb_connection conn, const char* schema, const char* table, append_data *data, size_t column_count)
{
	Client c = (Client) conn;
	mvc *m;
	char* msg = MAL_SUCCEED;

	if ((msg = validate_connection(conn, "embedded.monetdb_append")) != MAL_SUCCEED)
		return msg;
	if (table == NULL)
		return createException(MAL, "embedded.monetdb_append", SQLSTATE(42000) "table parameter is NULL");
	if (data == NULL)
		return createException(MAL, "embedded.monetdb_append", SQLSTATE(42000) "data parameter is NULL");
	if (column_count < 1)
		return createException(MAL, "embedded.monetdb_append", SQLSTATE(42000) "column_count must be higher than 0");

	if ((msg = getSQLContext(c, NULL, &m, NULL)) != MAL_SUCCEED)
		return msg;
	if (m->session->status < 0 && m->session->auto_commit == 0)
		return createException(SQL, "embedded.monetdb_append", SQLSTATE(25005) "Current transaction is aborted (please ROLLBACK)");
	if (!m->sa) // unclear why this is required
		m->sa = sa_create();
	if (!m->sa)
		return createException(SQL, "embedded.monetdb_append", SQLSTATE(HY001) MAL_MALLOC_FAIL);

	if ((msg = SQLtrans(m)) != MAL_SUCCEED)
		return msg;
	{
		node *n;
		size_t i;
		sql_rel *rel;
		sql_query *query = query_create(m);
		list *exps = sa_list(m->sa), *args = sa_list(m->sa), *col_types = sa_list(m->sa);
		sql_schema *s = mvc_bind_schema(m, schema);
		sql_table *t;
		sql_subfunc *f = sql_find_func(m->sa, mvc_bind_schema(m, "sys"), "append", 1, F_UNION, NULL);

		if (!s)
			return createException(SQL, "embedded.monetdb_append", SQLSTATE(3F000) "Schema missing %s", schema);
		t = mvc_bind_table(m, s, table);
		if (!t)
			return createException(SQL, "embedded.monetdb_append", SQLSTATE(42S02) "Table missing %s.%s", schema, table);
		if (column_count != (size_t)list_length(t->columns.set))
			return createException(SQL, "embedded.monetdb_append", SQLSTATE(21S01) "Incorrect number of columns");
		for (i = 0, n = t->columns.set->h; i < column_count && n; i++, n = n->next) {
			sql_column *col = n->data;
			list_append(args, exp_atom_lng(m->sa, data[i].batid));
			list_append(exps, exp_column(m->sa, t->base.name, col->base.name, &col->type, CARD_MULTI, col->null, 0));
			list_append(col_types, &col->type);
		}

		f->res = col_types;
		rel = rel_insert(query, rel_basetable(m, t, t->base.name), rel_table_func(m->sa, NULL, exp_op(m->sa,  args, f), exps, 1));
		m->scanner.rs = NULL;
		m->errstr[0] = '\0';

		if (backend_dumpstmt((backend *) c->sqlcontext, c->curprg->def, rel, 1, 1, "append") < 0)
			return createException(SQL, "embedded.monetdb_append", SQLSTATE(HY001) "Append plan generation failure");
		if ((msg = SQLoptimizeQuery(c, c->curprg->def)) != MAL_SUCCEED)
			return msg;
		if ((msg = SQLengine(c)) != MAL_SUCCEED)
			return msg;
	}
	return SQLautocommit(m);
}

char*
monetdb_cleanup_result(monetdb_connection conn, monetdb_result* result)
{
	char* msg = MAL_SUCCEED;
	monetdb_result_internal* res = (monetdb_result_internal *) result;

	if ((msg = validate_connection(conn, "embedded.monetdb_cleanup_result")) != MAL_SUCCEED)
		return msg;
	if (!result)
		return createException(MAL, "embedded.monetdb_cleanup_result", SQLSTATE(42000) "Parameter result is NULL");

	if (res->monetdb_resultset)
		res_tables_destroy(res->monetdb_resultset);

	if (res->converted_columns) {
		size_t i;
		for (i = 0; i < res->res.ncols; i++)
			monetdb_destroy_column(res->converted_columns[i]);
		GDKfree(res->converted_columns);
	}
	GDKfree(res);

	return msg;
}

char*
monetdb_get_columns(monetdb_connection conn, const char* schema_name, const char *table_name, size_t *column_count,
					char ***column_names, int **column_types)
{
	mvc *m;
	sql_schema *s;
	sql_table *t;
	char* msg = MAL_SUCCEED;
	int columns;
	node *n;
	Client c = (Client) conn;

	if ((msg = validate_connection(conn, "embedded.monetdb_get_columns")) != MAL_SUCCEED)
		return msg;

	if (!column_count)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(42000) "Parameter column_count is NULL");
	if (!column_names)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(42000) "Parameter column_names is NULL");
	if (!column_types)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(42000) "Parameter column_types is NULL");
	if (!schema_name)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(42000) "Parameter schema_name is NULL");
	if (!table_name)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(42000) "Parameter table_name is NULL");

	if ((msg = getSQLContext(c, NULL, &m, NULL)) != MAL_SUCCEED)
		return msg;

	s = mvc_bind_schema(m, schema_name);
	if (s == NULL)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(3F000) "Missing schema!");
	t = mvc_bind_table(m, s, table_name);
	if (t == NULL)
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(3F000) "Could not find table %s", table_name);

	columns = t->columns.set->cnt;
	*column_count = columns;
	*column_names = GDKzalloc(sizeof(char*) * columns);
	*column_types = GDKzalloc(sizeof(int) * columns);
	if (*column_names == NULL || *column_types == NULL) {
		if (*column_names) {
			GDKfree(*column_names);
			*column_names = NULL;
		}
		if (*column_types) {
			GDKfree(*column_types);
			*column_types = NULL;
		}
		return createException(MAL, "embedded.monetdb_get_columns", SQLSTATE(HY001) MAL_MALLOC_FAIL);
	}

	for (n = t->columns.set->h; n; n = n->next) {
		sql_column *col = n->data;
		(*column_names)[col->colnr] = col->base.name;
		(*column_types)[col->colnr] = col->type.type->localtype;
	}

	return msg;
}

char*
monetdb_shutdown(void)
{
	char* msg = MAL_SUCCEED;
	MT_lock_set(&embedded_lock);
	msg = monetdb_shutdown_internal();
	MT_lock_unset(&embedded_lock);
	return msg;
}

#define GENERATE_BASE_HEADERS(type, tpename) \
	static int tpename##_is_null(type value)

#define GENERATE_BASE_FUNCTIONS(tpe, tpename, mname) \
	GENERATE_BASE_HEADERS(tpe, tpename); \
	static int tpename##_is_null(tpe value) { return value == mname##_nil; }

GENERATE_BASE_FUNCTIONS(int8_t, int8_t, bte)
GENERATE_BASE_FUNCTIONS(int16_t, int16_t, sht)
GENERATE_BASE_FUNCTIONS(int32_t, int32_t, int)
GENERATE_BASE_FUNCTIONS(int64_t, int64_t, lng)
GENERATE_BASE_FUNCTIONS(size_t, size_t, oid)

GENERATE_BASE_FUNCTIONS(float, float, flt)
GENERATE_BASE_FUNCTIONS(double, double, dbl)

GENERATE_BASE_HEADERS(char*, str);
GENERATE_BASE_HEADERS(monetdb_data_blob, blob);

GENERATE_BASE_HEADERS(monetdb_data_date, date);
GENERATE_BASE_HEADERS(monetdb_data_time, time);
GENERATE_BASE_HEADERS(monetdb_data_timestamp, timestamp);

#define GENERATE_BAT_INPUT_BASE(tpe)                                           \
	monetdb_column_##tpe *bat_data = GDKzalloc(sizeof(monetdb_column_##tpe));  \
	if (!bat_data) {                                                           \
		msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL); \
		goto wrapup;                                                           \
	}                                                                          \
	bat_data->type = monetdb_##tpe;                                            \
	bat_data->is_null = tpe##_is_null;                                         \
	bat_data->scale = pow(10, sqltpe->scale);                                  \
	column_result = (monetdb_column*) bat_data;

#define GENERATE_BAT_INPUT(b, tpe, mtype)                                      \
	{                                                                          \
		GENERATE_BAT_INPUT_BASE(tpe);                                          \
		bat_data->count = BATcount(b);                                         \
		bat_data->null_value = mtype##_nil;                                    \
		bat_data->data = GDKzalloc(bat_data->count * sizeof(bat_data->null_value)); \
		if (!bat_data->data) {                                                 \
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL); \
			goto wrapup;                                                       \
		}                                                                      \
		size_t it = 0;                                                         \
		mtype* val = (mtype*)Tloc(b, 0);                                       \
		/* bat is dense, materialize it */                                     \
		for (it = 0; it < bat_data->count; it++, val++)                        \
			bat_data->data[it] = (tpe) *val;                                   \
	}

static void data_from_date(date d, monetdb_data_date *ptr);
static void data_from_time(daytime d, monetdb_data_time *ptr);
static void data_from_timestamp(timestamp d, monetdb_data_timestamp *ptr);

char*
monetdb_result_fetch(monetdb_column** res, monetdb_result* mres, size_t column_index)
{
	BAT* b = NULL;
	int bat_type;
	char* msg = NULL;
	monetdb_result_internal* result = (monetdb_result_internal*) mres;
	sql_subtype* sqltpe = NULL;
	monetdb_column* column_result = NULL;
	size_t j = 0;

	if (!res) {
		msg = createException(MAL, "embedded.monetdb_result_fetch", "Parameter res is NULL");
		goto wrapup;
	}
	if (column_index >= mres->ncols) {
		msg = createException(MAL, "embedded.monetdb_result_fetch", "Index out of range");
		goto wrapup;
	}
	// check if we have the column converted already
	if (result->converted_columns[column_index]) {
		*res = result->converted_columns[column_index];
		return MAL_SUCCEED;
	}

	// otherwise we have to convert the column
	b = BATdescriptor(result->monetdb_resultset->cols[column_index].b);
	if (!b) {
		msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY002) RUNTIME_OBJECT_MISSING);
		goto wrapup;
	}
	bat_type = b->ttype;
	sqltpe = &result->monetdb_resultset->cols[column_index].type;

	if (bat_type == TYPE_bit || bat_type == TYPE_bte) {
		GENERATE_BAT_INPUT(b, int8_t, bte);
	} else if (bat_type == TYPE_sht) {
		GENERATE_BAT_INPUT(b, int16_t, sht);
	} else if (bat_type == TYPE_int) {
		GENERATE_BAT_INPUT(b, int32_t, int);
	} else if (bat_type == TYPE_oid) {
		GENERATE_BAT_INPUT(b, size_t, oid);
	} else if (bat_type == TYPE_lng) {
		GENERATE_BAT_INPUT(b, int64_t, lng);
	} else if (bat_type == TYPE_flt) {
		GENERATE_BAT_INPUT(b, float, flt);
	} else if (bat_type == TYPE_dbl) {
		GENERATE_BAT_INPUT(b, double, dbl);
	} else if (bat_type == TYPE_str) {
		BATiter li;
		BUN p = 0, q = 0;
		GENERATE_BAT_INPUT_BASE(str);
		bat_data->count = BATcount(b);
		bat_data->data = GDKzalloc(sizeof(char *) * bat_data->count);
		bat_data->null_value = NULL;
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}

		j = 0;
		li = bat_iterator(b);
		BATloop(b, p, q)
		{
			char *t = (char *)BUNtail(li, p);
			if (strcmp(t, str_nil) == 0) {
				bat_data->data[j] = NULL;
			} else {
				bat_data->data[j] = GDKstrdup(t);
				if (!bat_data->data[j]) {
					goto wrapup;
				}
			}
			j++;
		}
	} else if (bat_type == TYPE_date) {
		date *baseptr;
		GENERATE_BAT_INPUT_BASE(date);
		bat_data->count = BATcount(b);
		bat_data->data = GDKmalloc(sizeof(bat_data->null_value) * bat_data->count);
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}

		baseptr = (date *)Tloc(b, 0);
		for (j = 0; j < bat_data->count; j++)
			data_from_date(baseptr[j], bat_data->data + j);
		data_from_date(date_nil, &bat_data->null_value);
	} else if (bat_type == TYPE_daytime) {
		daytime *baseptr;
		GENERATE_BAT_INPUT_BASE(time);
		bat_data->count = BATcount(b);
		bat_data->data = GDKmalloc(sizeof(bat_data->null_value) * bat_data->count);
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}

		baseptr = (daytime *)Tloc(b, 0);
		for (j = 0; j < bat_data->count; j++)
			data_from_time(baseptr[j], bat_data->data + j);
		data_from_time(daytime_nil, &bat_data->null_value);
	} else if (bat_type == TYPE_timestamp) {
		timestamp *baseptr;
		GENERATE_BAT_INPUT_BASE(timestamp);
		bat_data->count = BATcount(b);
		bat_data->data = GDKmalloc(sizeof(bat_data->null_value) * bat_data->count);
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}

		baseptr = (timestamp *)Tloc(b, 0);
		for (j = 0; j < bat_data->count; j++)
			data_from_timestamp(baseptr[j], bat_data->data + j);
		data_from_timestamp(*timestamp_nil, &bat_data->null_value);
	} else if (bat_type == TYPE_blob) {
		BATiter li;
		BUN p = 0, q = 0;
		GENERATE_BAT_INPUT_BASE(blob);
		bat_data->count = BATcount(b);
		bat_data->data = GDKmalloc(sizeof(monetdb_data_blob) * bat_data->count);
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}
		j = 0;

		li = bat_iterator(b);
		BATloop(b, p, q)
		{
			blob *t = (blob *)BUNtail(li, p);
			if (t->nitems == ~(size_t)0) {
				bat_data->data[j].size = 0;
				bat_data->data[j].data = NULL;
			} else {
				bat_data->data[j].size = t->nitems;
				bat_data->data[j].data = GDKmalloc(t->nitems);
				if (!bat_data->data[j].data) {
					msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
					goto wrapup;
				}
				memcpy(bat_data->data[j].data, t->data, t->nitems);
			}
			j++;
		}
		bat_data->null_value.size = 0;
		bat_data->null_value.data = NULL;
	} else {
		// unsupported type: convert to string
		BATiter li;
		BUN p = 0, q = 0;
		GENERATE_BAT_INPUT_BASE(str);
		bat_data->count = BATcount(b);
		bat_data->null_value = NULL;
		bat_data->data = GDKzalloc(sizeof(char *) * bat_data->count);
		if (!bat_data->data) {
			msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(HY001) MAL_MALLOC_FAIL);
			goto wrapup;
		}
		j = 0;

		li = bat_iterator(b);
		BATloop(b, p, q)
		{
			void *t = BUNtail(li, p);
			if (BATatoms[bat_type].atomCmp(t, BATatoms[bat_type].atomNull) == 0) {
				bat_data->data[j] = NULL;
			} else {
				char *sresult = NULL;
				size_t length = 0;
				if (BATatoms[bat_type].atomToStr(&sresult, &length, t, true) == 0) {
					msg = createException(MAL, "embedded.monetdb_result_fetch", SQLSTATE(42000) "Failed to convert element to string");
					goto wrapup;
				}
				bat_data->data[j] = sresult;
			}
			j++;
		}
	}
	BBPunfix(b->batCacheid);
	result->converted_columns[column_index] = column_result;
	*res = result->converted_columns[column_index];
	return MAL_SUCCEED;
wrapup:
	if (b)
		BBPunfix(b->batCacheid);
	monetdb_destroy_column(column_result);
	return msg;
}

char*
monetdb_result_fetch_rawcol(void** res, monetdb_result* mres, size_t column_index)
{
	monetdb_result_internal* result = (monetdb_result_internal*) mres;
	if (column_index >= mres->ncols) // index out of range
		return createException(MAL, "embedded.monetdb_result_fetch_rawcol", SQLSTATE(42000) "Index out of range");
	*res = &(result->monetdb_resultset->cols[column_index]);
	return MAL_SUCCEED;
}

void
data_from_date(date d, monetdb_data_date *ptr)
{
	int day, month, year;

	MTIMEfromdate(d, &day, &month, &year);
	ptr->day = day;
	ptr->month = month;
	ptr->year = year;
}

void
data_from_time(daytime d, monetdb_data_time *ptr)
{
	int hour, min, sec, msec;

	MTIMEfromtime(d, &hour, &min, &sec, &msec);
	ptr->hours = hour;
	ptr->minutes = min;
	ptr->seconds = sec;
	ptr->ms = msec;
}

void
data_from_timestamp(timestamp d, monetdb_data_timestamp *ptr)
{
	data_from_date(d.payload.p_days, &ptr->date);
	data_from_time(d.payload.p_msecs, &ptr->time);
}

static date
date_from_data(monetdb_data_date *ptr)
{
	return MTIMEtodate(ptr->day, ptr->month, ptr->year);
}

static daytime
time_from_data(monetdb_data_time *ptr)
{
	return MTIMEtotime(ptr->hours, ptr->minutes, ptr->seconds, ptr->ms);
}

static timestamp
timestamp_from_data(monetdb_data_timestamp *ptr)
{
	timestamp d;
	d.payload.p_days = date_from_data(&ptr->date);
	d.payload.p_msecs = time_from_data(&ptr->time);
	return d;
}

int
date_is_null(monetdb_data_date value)
{
	monetdb_data_date null_value;
	data_from_date(date_nil, &null_value);
	return value.year == null_value.year && value.month == null_value.month &&
		   value.day == null_value.day;
}

int
time_is_null(monetdb_data_time value)
{
	monetdb_data_time null_value;
	data_from_time(daytime_nil, &null_value);
	return value.hours == null_value.hours &&
		   value.minutes == null_value.minutes &&
		   value.seconds == null_value.seconds && value.ms == null_value.ms;
}

int
timestamp_is_null(monetdb_data_timestamp value)
{
	return is_timestamp_nil(timestamp_from_data(&value));
}

int
str_is_null(char *value)
{
	return value == NULL;
}

int
blob_is_null(monetdb_data_blob value)
{
	return value.data == NULL;
}

void
monetdb_destroy_column(monetdb_column* column)
{
	size_t j;

	if (!column)
		return;

	if (column->type == monetdb_str) {
		// FIXME: clean up individual strings
		char** data = (char**)column->data;
		for(j = 0; j < column->count; j++) {
			if (data[j])
				GDKfree(data[j]);
		}
	} else if (column->type == monetdb_blob) {
		monetdb_data_blob* data = (monetdb_data_blob*)column->data;
		for(j = 0; j < column->count; j++) {
			if (data[j].data)
				GDKfree(data[j].data);
		}
	}
	GDKfree(column->data);
	GDKfree(column);
}
