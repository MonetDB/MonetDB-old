/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#include "monetdb_config.h"

#include "sql_scripts.h"
#include "gdk.h"
#include "mal_client.h"
#include "sql_catalog.h"
#include "sql_execute.h"

#include "createdb_inline1.h"
#include "createdb_inline2.h"

static str
install_sql_scripts_array(Client c, char* scripts_array, const char* array_name)
{
	str msg = MAL_SUCCEED;
	size_t createdb_len;
	buffer* createdb_buf;
	stream* createdb_stream;
	bstream* createdb_bstream;

	assert(scripts_array && array_name);
	createdb_len = strlen(scripts_array);
	if ((createdb_buf = GDKmalloc(sizeof(buffer))) == NULL)
		throw(MAL, "sql.install_sql_scripts_array", SQLSTATE(HY001) MAL_MALLOC_FAIL);
	buffer_init(createdb_buf, scripts_array, createdb_len);
	if ((createdb_stream = buffer_rastream(createdb_buf, "sql.install_sql_scripts_array")) == NULL) {
		GDKfree(createdb_buf);
		throw(MAL, "sql.install_sql_scripts_array", SQLSTATE(HY001) MAL_MALLOC_FAIL);
	}
	if ((createdb_bstream = bstream_create(createdb_stream, createdb_len)) == NULL) {
		mnstr_destroy(createdb_stream);
		GDKfree(createdb_buf);
		throw(MAL, "sql.install_sql_scripts_array", SQLSTATE(HY001) MAL_MALLOC_FAIL);
	}
	if (bstream_next(createdb_bstream) >= 0)
		msg = SQLstatementIntern(c, &createdb_bstream->buf, "sql.install_sql_scripts_array", TRUE, FALSE, NULL);
	else
		msg = createException(MAL, "sql.install_sql_scripts_array", SQLSTATE(HY0002) "Could not load %s script", array_name);

	bstream_destroy(createdb_bstream);
	GDKfree(createdb_buf);
	return msg;
}

str
install_sql_scripts1(Client c)
{
	str err;
	if ((err = install_sql_scripts_array(c, createdb_inline1, "createdb_inline1")) != MAL_SUCCEED)
		return err;
	return MAL_SUCCEED;
}

str
install_sql_scripts2(Client c)
{
	str err;
	if ((err = install_sql_scripts_array(c, createdb_inline2, "createdb_inline2")) != MAL_SUCCEED)
		return err;
	return MAL_SUCCEED;
}
