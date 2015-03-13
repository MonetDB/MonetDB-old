#include "monetdb_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "registrar.h"
#include "mtime.h"

#include "sql_mvc.h"
#include "sql.h"

#include "stream.h"
#include "sql_scenario.h"
#include "mal_linker.h"

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

/*
 * keeps BAT and other properties of columns of a table.
 */
typedef struct {
	bat *column_bats; /* keeps bats of the columns: lower array */
	str *column_names; /* names of columns that are kept in the higher array */
	str *column_types_strs; /* type strings of columns */
} temp_subcontainer;

/*
 * keeps (some) tables of a schema.
 */
typedef struct {
	str schema_name; /* schema or vault name */
	temp_subcontainer *tables_columns; /* keeps tables: higher array */
	str *table_names; /* names of tables that are kept in the higher array */
	int *num_columns; /* number of columns in each table in the higher array */
	sht num_tables;
} temp_container;

typedef struct {
	int tid; /* thread id */
	str *file_paths; /* array of file paths to loop on in the thread */
	lng loop_start;
	lng loop_end;
	int mode; /* carries to the thread */
	mvc *mvc; 
	int base_file_id;
	temp_container* tc;
} thread_argv;

lng get_line_num(str filename);
lng get_file_paths(str repo_path, str** ret_file_paths);

str get_column_info_of_table(sql_table* t, str** ret_column_names, str** ret_column_type_strs, sht** ret_column_types);
bit is_actual_data_table(str tname);
str create_temp_container_segments_mode(mvc* m, temp_container* ret_tc, int is_with_mount);

str mseed_create_temp_container(temp_container* ret_tc);
str mseed_create_temp_container_with_data_tables(temp_container* ret_tc);
str mseed_create_temp_container_segments_mode(temp_container* ret_tc);
str mseed_create_temp_container_with_data_tables_segments_mode(temp_container* ret_tc);
str mseed_register(int assigned_file_id, str file_path, temp_container* ret_tc);
str mseed_register_and_mount(int assigned_file_id, str file_path, temp_container* ret_tc);
str mseed_register_segments_mode(int assigned_file_id, str file_path, temp_container* ret_tc);
str mseed_register_and_mount_segments_mode(int assigned_file_id, str file_path, temp_container* ret_tc);
int concatenate_strs(str* words_to_concat, int num_words_to_concat, str* ret_concatenated);
str prepare_insertion(Client cntxt, temp_container* tc);
str insert_into_vault(Client cntxt, temp_container* tc);
str clean_up_after_insertion(Client cntxt, temp_container* tc);
str SQLstatementIntern(Client c, str *expr, str nme, int execute, bit output, res_table **result);
str register_clean_up(temp_container* tc);
void *register_files(void *args);

// static str runSQLQuery(Client c, char **result, char * query);

double  timetol      = -1.0; /* Time tolerance for continuous traces */
double  sampratetol  = -1.0; /* Sample rate tolerance for continuous traces */
int verbose = 1;
static lng file_counter = 0;
static lng num_registered_files = 0;

/*
 * returns number of lines in a file.
 *
 * WARNING: always counts EOF as a line. So proper return is taken if the file does
 * not have a newline at the end.
 */
lng get_line_num(str filename)
{
	FILE *f;
	char c;
	lng lines = 0;

	f = fopen(filename, "r");

	if(f == NULL)
		return 0;

	while((c = fgetc(f)) != EOF)
		if(c == '\n')
			lines++;

	fclose(f);

/* 	if(c != '\n') */
/* 		lines++; */

	return lines;
}

/*
 * returns number of file_paths in repo_path.
 *
 * stores each file_path in ret_file_paths.
 *
 * repo_path may be either a file that is containing file_paths in the repo one per line,
 * or a (recursive) directory containing the repository.
 *
 * TODO: if a directory path is given, traverse the directory recursively and collect all the files.
 */
lng get_file_paths(str repo_path, str** ret_file_paths)
{
	lng num_file_paths = 0;
	struct stat s;
	str* file_paths = NULL;
	if( stat(repo_path,&s) == 0 )
	{
		if( s.st_mode & S_IFDIR )
		{
			/* it's a directory */
			/* traverse and collect all the files */
		}
		else if( s.st_mode & S_IFREG )
		{
			/* it's a file */
			/* each line is a file_path */

			FILE *file;
			num_file_paths = get_line_num(repo_path);
			printf("num_file_paths: "LLFMT"\n", num_file_paths);

			*ret_file_paths = file_paths = (str*) GDKmalloc(num_file_paths * sizeof(str));
			assert(file_paths != NULL);

			file = fopen (repo_path, "r");

			if ( file != NULL )
			{
				char line [255]; /* or other suitable maximum line size */
				lng i = 0;
				while ( fgets ( line, sizeof(line), file ) != NULL ) /* read a line */
				{
					int len_line = strlen(line);
					/* 			if(len_line == 1) */
					/* 				continue; */
					if(line[len_line-1] == '\n')
						line[len_line-1] = '\0';
					file_paths[i] = GDKstrdup(line);
					i++;
				}
				fclose ( file );
			}
			else
			{
				perror ( repo_path ); /* why didn't the file open? */
			}
		}
		else
		{
			/* something else */
			return -1;
		}
	}
	else
	{
		/* error */
		return -1;
	}

	return num_file_paths;

}

/*
 * fills the temp_container structure with the "mseed" metadata tables' info.
 *
 * returns error or MAL_SUCCEED
 *
 * TODO: This function is now hardcoding every info. It can be made generic,
 * because required info is in sql_catalog.
 */
str mseed_create_temp_container(temp_container* ret_tc)
{
	/* cat: (metadata) catalog, fil: (metadata) files. */
	int num_tables = 2;
	int num_c_fil = 8;
	int num_c_cat = 7;
	int c, t;

	str sch_name = "mseed";

	str cn_fil[] = {"file_location", "dataquality", "network", "station", "location", "channel", "encoding", "byte_order"};
	str cn_cat[] = {"file_location", "seq_no", "record_length", "start_time", "frequency", "sample_count", "sample_type"};

	str cts_fil[] = {"string", "char", "string", "string", "string", "string", "tinyint", "boolean"};
	str cts_cat[] = {"string", "int", "int", "timestamp", "double", "bigint", "char"};

	sht ct_fil[] = {TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_bte, TYPE_bit};
	sht ct_cat[] = {TYPE_str, TYPE_int, TYPE_int, TYPE_timestamp, TYPE_dbl, TYPE_lng, TYPE_str};

	str tn[] = {"files", "catalog"};
	int num_c[] = {8, 7};

	bat *cb_fil = (bat*)GDKmalloc(num_c_fil*sizeof(bat));

	bat *cb_cat = (bat*)GDKmalloc(num_c_cat*sizeof(bat));

	temp_subcontainer *tscs = (temp_subcontainer*)GDKmalloc(num_tables*sizeof(temp_subcontainer));

	BAT *aBAT;

	assert(cb_fil!=NULL);
	assert(cb_cat!=NULL);
	assert(tscs!=NULL);

	/* cb_fil */
	for(c = 0; c < num_c_fil; c++)
	{
		aBAT = BATnew(TYPE_void, ct_fil[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_fil[c] = aBAT->batCacheid);
	}

	/* cb_cat */
	for(c = 0; c < num_c_cat; c++)
	{
		aBAT = BATnew(TYPE_void, ct_cat[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_cat[c] = aBAT->batCacheid);
	}

	(tscs+0)->column_bats = cb_fil;

	(tscs+1)->column_bats = cb_cat;

	(tscs+0)->column_names = (str*) GDKmalloc(num_c[0]*sizeof(str));
	(tscs+0)->column_types_strs = (str*) GDKmalloc(num_c[0]*sizeof(str));
	for(c = 0; c < num_c[0]; c++)
	{
		(tscs+0)->column_names[c] = GDKstrdup(cn_fil[c]);
		(tscs+0)->column_types_strs[c] = GDKstrdup(cts_fil[c]);
	}

	(tscs+1)->column_names = (str*) GDKmalloc(num_c[1]*sizeof(str));
	(tscs+1)->column_types_strs = (str*) GDKmalloc(num_c[1]*sizeof(str));
	for(c = 0; c < num_c[1]; c++)
	{
		(tscs+1)->column_names[c] = GDKstrdup(cn_cat[c]);
		(tscs+1)->column_types_strs[c] = GDKstrdup(cts_cat[c]);
	}

	ret_tc->schema_name = sch_name;
	ret_tc->tables_columns = tscs;

	ret_tc->table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	ret_tc->num_columns = (int*) GDKmalloc(num_tables*sizeof(int));
	for(t = 0; t < num_tables; t++)
	{
		ret_tc->table_names[t] = GDKstrdup(tn[t]);
		ret_tc->num_columns[t] = num_c[t];
	}

	ret_tc->num_tables = num_tables;

	return MAL_SUCCEED;
}

/*
 * fills the temp_container structure with the "mseed" meta-data and data tables' info.
 *
 * returns error or MAL_SUCCEED
 *
 * TODO: This function is now hardcoding every info. It can be made generic,
 * because required info is in sql_catalog.
 */
str mseed_create_temp_container_with_data_tables(temp_container* ret_tc)
{
	/* cat: (metadata) catalog, fil: (metadata) files. */
	int num_tables = 3;
	int num_c_fil = 8;
	int num_c_cat = 7;
	int num_c_dat = 4;
	int c, t;
	
	str sch_name = "mseed";
	
	str cn_fil[] = {"file_location", "dataquality", "network", "station", "location", "channel", "encoding", "byte_order"};
	str cn_cat[] = {"file_location", "seq_no", "record_length", "start_time", "frequency", "sample_count", "sample_type"};
	str cn_dat[] = {"file_location", "seq_no", "sample_time", "sample_value"};
	
	str cts_fil[] = {"string", "char", "string", "string", "string", "string", "tinyint", "boolean"};
	str cts_cat[] = {"string", "int", "int", "timestamp", "double", "bigint", "char"};
	str cts_dat[] = {"string", "int", "timestamp", "int"};
	
	sht ct_fil[] = {TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_bte, TYPE_bit};
	sht ct_cat[] = {TYPE_str, TYPE_int, TYPE_int, TYPE_timestamp, TYPE_dbl, TYPE_lng, TYPE_str};
	sht ct_dat[] = {TYPE_str, TYPE_int, TYPE_timestamp, TYPE_int};
	
	str tn[] = {"files", "catalog", "data"};
	int num_c[] = {8, 7, 4};
	
	bat *cb_fil = (bat*)GDKmalloc(num_c_fil*sizeof(bat));
	bat *cb_cat = (bat*)GDKmalloc(num_c_cat*sizeof(bat));
	bat *cb_dat = (bat*)GDKmalloc(num_c_dat*sizeof(bat));
	
	temp_subcontainer *tscs = (temp_subcontainer*)GDKmalloc(num_tables*sizeof(temp_subcontainer));
	
	BAT *aBAT;
	
	assert(cb_fil!=NULL);
	assert(cb_cat!=NULL);
	assert(cb_dat!=NULL);
	assert(tscs!=NULL);
	
	/* cb_fil */
	for(c = 0; c < num_c_fil; c++)
	{
		aBAT = BATnew(TYPE_void, ct_fil[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_fil[c] = aBAT->batCacheid);
	}
	
	/* cb_cat */
	for(c = 0; c < num_c_cat; c++)
	{
		aBAT = BATnew(TYPE_void, ct_cat[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_cat[c] = aBAT->batCacheid);
	}
	
	/* cb_dat */
	for(c = 0; c < num_c_dat; c++)
	{
		aBAT = BATnew(TYPE_void, ct_dat[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_dat[c] = aBAT->batCacheid);
	}
	
	(tscs+0)->column_bats = cb_fil;
	(tscs+1)->column_bats = cb_cat;
	(tscs+2)->column_bats = cb_dat;
	
	(tscs+0)->column_names = (str*) GDKmalloc(num_c[0]*sizeof(str));
	(tscs+0)->column_types_strs = (str*) GDKmalloc(num_c[0]*sizeof(str));
	for(c = 0; c < num_c[0]; c++)
	{
		(tscs+0)->column_names[c] = GDKstrdup(cn_fil[c]);
		(tscs+0)->column_types_strs[c] = GDKstrdup(cts_fil[c]);
	}
	
	(tscs+1)->column_names = (str*) GDKmalloc(num_c[1]*sizeof(str));
	(tscs+1)->column_types_strs = (str*) GDKmalloc(num_c[1]*sizeof(str));
	for(c = 0; c < num_c[1]; c++)
	{
		(tscs+1)->column_names[c] = GDKstrdup(cn_cat[c]);
		(tscs+1)->column_types_strs[c] = GDKstrdup(cts_cat[c]);
	}
	
	(tscs+2)->column_names = (str*) GDKmalloc(num_c[2]*sizeof(str));
	(tscs+2)->column_types_strs = (str*) GDKmalloc(num_c[2]*sizeof(str));
	for(c = 0; c < num_c[2]; c++)
	{
		(tscs+2)->column_names[c] = GDKstrdup(cn_dat[c]);
		(tscs+2)->column_types_strs[c] = GDKstrdup(cts_dat[c]);
	}
	
	ret_tc->schema_name = sch_name;
	ret_tc->tables_columns = tscs;
	
	ret_tc->table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	ret_tc->num_columns = (int*) GDKmalloc(num_tables*sizeof(int));
	for(t = 0; t < num_tables; t++)
	{
		ret_tc->table_names[t] = GDKstrdup(tn[t]);
		ret_tc->num_columns[t] = num_c[t];
	}
	
	ret_tc->num_tables = num_tables;
	
	return MAL_SUCCEED;
}

str get_column_info_of_table(sql_table* t, str** ret_column_names, str** ret_column_type_strs, sht** ret_column_types)
{
	int num_columns;
	int c = 0;
	node *n;
	
	num_columns = list_length(t->columns.set);
	
	*ret_column_names = (str*) GDKmalloc(num_columns*sizeof(str));
	*ret_column_type_strs = (str*) GDKmalloc(num_columns*sizeof(str));
	*ret_column_types = (sht*) GDKmalloc(num_columns*sizeof(sht));
	
	for (n = t->columns.set->h; n; n = n->next) {
		sql_column *col = n->data;
		
		(*ret_column_names)[c] = GDKstrdup(col->base.name);
		if(strstr(col->type.type->sqlname, "varchar") != NULL) 
		{
			char varchar_str[15];
			char digits_str[5];
			strcpy(varchar_str, col->type.type->sqlname);
			strcat(varchar_str, "(");
			sprintf(digits_str, "%d", col->type.digits);
			strcat(varchar_str, digits_str);
			strcat(varchar_str, ")");
			(*ret_column_type_strs)[c] = GDKstrdup(varchar_str);
		}
		else
			(*ret_column_type_strs)[c] = GDKstrdup(col->type.type->sqlname);
		(*ret_column_types)[c] = col->type.type->localtype;
		
		c++;
	}
	
	return MAL_SUCCEED;
}

bit is_actual_data_table(str tname)
{
	str actual_data_str = "data";
	if(strstr(tname, actual_data_str) != NULL) 
		return TRUE;
	else 
		return FALSE;
}

/*
 * fills the temp_container structure with the "mseed" metadata tables' info.
 *
 * returns error or MAL_SUCCEED
 *
 * TODO: This function is now hardcoding every info. It can be made generic,
 * because required info is in sql_catalog.
 */
str create_temp_container_segments_mode(mvc* m, temp_container* ret_tc, int is_with_mount)
{
	sql_schema *s = mvc_bind_schema(m, m->session->schema_name);
// 	sql_table *t;
	node *n;
	int num_tables;
	int i, c;
	BAT *aBAT;
	str msg;
	
	str* table_names;
	str** column_names; 
	str** column_type_strs; 
	sht** column_types;
	int* num_c;
	bat** column_bats;
	temp_subcontainer *tscs;
	
	num_tables = 0;
	
	if (s == NULL)
		throw(SQL,"create_temp_container_segments_mode","3F000!Schema missing");
		
	for (n = s->tables.set->h; n; n = n->next) {
		sql_table *t = n->data;
		
		/* skip the views and so. */
		if(!isTable(t))
			continue;
		
		/* skip the actual data tables if not is_with_mount */
		if(!is_with_mount && is_actual_data_table(t->base.name))
			continue;
		
		num_tables++;
	}
	
	table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	column_names = (str**) GDKmalloc(num_tables*sizeof(str*)); 
	column_type_strs = (str**) GDKmalloc(num_tables*sizeof(str*)); 
	column_types = (sht**) GDKmalloc(num_tables*sizeof(sht*));
	num_c = (int*) GDKmalloc(num_tables*sizeof(int));
	column_bats = (bat**) GDKmalloc(num_tables*sizeof(bat*));
	tscs = (temp_subcontainer*)GDKmalloc(num_tables*sizeof(temp_subcontainer));
	
	assert(tscs!=NULL);
	
	i = 0;
	
	for (n = s->tables.set->h; n; n = n->next) {
		sql_table *t = n->data;
		
		/* skip the views and so. */
		if(!isTable(t))
			continue;
		
		/* skip the actual data tables if not is_with_mount */
		if(!is_with_mount && is_actual_data_table(t->base.name))
			continue;
		
		table_names[i] = GDKstrdup(t->base.name);
		
		num_c[i] = list_length(t->columns.set);
		
		printf("%d. num_c: %d\n", i, num_c[i]);
		
		msg = get_column_info_of_table(t, &(column_names[i]), &(column_type_strs[i]), &(column_types[i]));
		if (msg)
			return msg;
		
		column_bats[i] = (bat*)GDKmalloc(num_c[i]*sizeof(bat));
		
		assert(column_bats[i]!=NULL);
		
		for(c = 0; c < num_c[i]; c++)
		{
			aBAT = BATnew(TYPE_void, column_types[i][c], 0, PERSISTENT); /* create empty BAT for each column. */
			if ( aBAT == NULL)
				throw(MAL,"create_temp_container_segments_mode",MAL_MALLOC_FAIL);
			BATseqbase(aBAT, 0);
			if ( aBAT == NULL)
				throw(MAL,"create_temp_container_segments_mode",MAL_MALLOC_FAIL);
			BBPkeepref(column_bats[i][c] = aBAT->batCacheid);
		}
		
		(tscs+i)->column_bats = column_bats[i];
		
		(tscs+i)->column_names = (str*) GDKmalloc(num_c[i]*sizeof(str));
		(tscs+i)->column_types_strs = (str*) GDKmalloc(num_c[i]*sizeof(str));
		for(c = 0; c < num_c[i]; c++)
		{
			(tscs+i)->column_names[c] = GDKstrdup(column_names[i][c]);
			(tscs+i)->column_types_strs[c] = GDKstrdup(column_type_strs[i][c]);
		}
		
		i++;
	}
	
	ret_tc->schema_name = GDKstrdup(m->session->schema_name);
	ret_tc->tables_columns = tscs;
	
	ret_tc->table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	ret_tc->num_columns = (int*) GDKmalloc(num_tables*sizeof(int));
	for(i = 0; i < num_tables; i++)
	{
		ret_tc->table_names[i] = GDKstrdup(table_names[i]);
		ret_tc->num_columns[i] = num_c[i];
	}
	
	ret_tc->num_tables = num_tables;
	
	return MAL_SUCCEED;
}


/*
 * fills the temp_container structure with the "mseed" metadata tables' info.
 *
 * returns error or MAL_SUCCEED
 *
 * TODO: This function is now hardcoding every info. It can be made generic,
 * because required info is in sql_catalog.
 */
str mseed_create_temp_container_segments_mode(temp_container* ret_tc)
{	
	/* seg: (metadata) segments, fil: (metadata) files. */
	int num_tables = 2;
	int num_c_fil = 8;
	int num_c_seg = 8;
	int c, t;
	
	str sch_name = "mseed";
	
	str cn_fil[] = {"file_location", "dataquality", "network", "station", "location", "channel", "encoding", "byte_order"};
	str cn_seg[] = {"file_location", "segment_id", "start_time", "end_time", "prev_gap", "frequency", "sample_count", "sample_type"};
	
	str cts_fil[] = {"string", "char", "string", "string", "string", "string", "tinyint", "boolean"};
	str cts_seg[] = {"string", "int", "timestamp", "timestamp", "double", "double", "bigint", "char"};
	
	sht ct_fil[] = {TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_bte, TYPE_bit};
	sht ct_seg[] = {TYPE_str, TYPE_int, TYPE_timestamp, TYPE_timestamp, TYPE_dbl, TYPE_dbl, TYPE_lng, TYPE_str};
	
	str tn[] = {"files", "segments"};
	int num_c[] = {8, 8};
	
	bat *cb_fil = (bat*)GDKmalloc(num_c_fil*sizeof(bat));
	
	bat *cb_seg = (bat*)GDKmalloc(num_c_seg*sizeof(bat));
	
	temp_subcontainer *tscs = (temp_subcontainer*)GDKmalloc(num_tables*sizeof(temp_subcontainer));
	
	BAT *aBAT;
	
	assert(cb_fil!=NULL);
	assert(cb_seg!=NULL);
	assert(tscs!=NULL);
	
	/* cb_fil */
	for(c = 0; c < num_c_fil; c++)
	{
		aBAT = BATnew(TYPE_void, ct_fil[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_fil[c] = aBAT->batCacheid);
	}
	
	/* cb_seg */
	for(c = 0; c < num_c_seg; c++)
	{
		aBAT = BATnew(TYPE_void, ct_seg[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_seg[c] = aBAT->batCacheid);
	}
	
	(tscs+0)->column_bats = cb_fil;
	
	(tscs+1)->column_bats = cb_seg;
	
	(tscs+0)->column_names = (str*) GDKmalloc(num_c[0]*sizeof(str));
	(tscs+0)->column_types_strs = (str*) GDKmalloc(num_c[0]*sizeof(str));
	for(c = 0; c < num_c[0]; c++)
	{
		(tscs+0)->column_names[c] = GDKstrdup(cn_fil[c]);
		(tscs+0)->column_types_strs[c] = GDKstrdup(cts_fil[c]);
	}
	
	(tscs+1)->column_names = (str*) GDKmalloc(num_c[1]*sizeof(str));
	(tscs+1)->column_types_strs = (str*) GDKmalloc(num_c[1]*sizeof(str));
	for(c = 0; c < num_c[1]; c++)
	{
		(tscs+1)->column_names[c] = GDKstrdup(cn_seg[c]);
		(tscs+1)->column_types_strs[c] = GDKstrdup(cts_seg[c]);
	}
	
	ret_tc->schema_name = sch_name;
	ret_tc->tables_columns = tscs;
	
	ret_tc->table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	ret_tc->num_columns = (int*) GDKmalloc(num_tables*sizeof(int));
	for(t = 0; t < num_tables; t++)
	{
		ret_tc->table_names[t] = GDKstrdup(tn[t]);
		ret_tc->num_columns[t] = num_c[t];
	}
	
	ret_tc->num_tables = num_tables;
	
	return MAL_SUCCEED;
}


/*
 * fills the temp_container structure with the "mseed" meta-data and data tables' info.
 *
 * returns error or MAL_SUCCEED
 *
 * TODO: This function is now hardcoding every info. It can be made generic,
 * because required info is in sql_catalog.
 */
str mseed_create_temp_container_with_data_tables_segments_mode(temp_container* ret_tc)
{
	/* seg: (metadata) segments, fil: (metadata) files. */
	int num_tables = 3;
	int num_c_fil = 8;
	int num_c_seg = 8;
	int num_c_dat = 4;
	int c, t;
	
	str sch_name = "mseed";
	
	str cn_fil[] = {"file_location", "dataquality", "network", "station", "location", "channel", "encoding", "byte_order"};
	str cn_seg[] = {"file_location", "segment_id", "start_time", "end_time", "prev_gap", "frequency", "sample_count", "sample_type"};
	str cn_dat[] = {"file_location", "segment_id", "sample_time", "sample_value"};
	
	str cts_fil[] = {"string", "char", "string", "string", "string", "string", "tinyint", "boolean"};
	str cts_seg[] = {"string", "int", "timestamp", "timestamp", "double", "double", "bigint", "char"};
	str cts_dat[] = {"string", "int", "timestamp", "int"};
	
	sht ct_fil[] = {TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_str, TYPE_bte, TYPE_bit};
	sht ct_seg[] = {TYPE_str, TYPE_int, TYPE_timestamp, TYPE_timestamp, TYPE_dbl, TYPE_dbl, TYPE_lng, TYPE_str};
	sht ct_dat[] = {TYPE_str, TYPE_int, TYPE_timestamp, TYPE_int};
	
	str tn[] = {"files", "segments", "data"};
	int num_c[] = {8, 8, 4};
	
	bat *cb_fil = (bat*)GDKmalloc(num_c_fil*sizeof(bat));
	bat *cb_seg = (bat*)GDKmalloc(num_c_seg*sizeof(bat));
	bat *cb_dat = (bat*)GDKmalloc(num_c_dat*sizeof(bat));
	
	temp_subcontainer *tscs = (temp_subcontainer*)GDKmalloc(num_tables*sizeof(temp_subcontainer));
	
	BAT *aBAT;
	
	assert(cb_fil!=NULL);
	assert(cb_seg!=NULL);
	assert(cb_dat!=NULL);
	assert(tscs!=NULL);
	
	/* cb_fil */
	for(c = 0; c < num_c_fil; c++)
	{
		aBAT = BATnew(TYPE_void, ct_fil[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_fil[c] = aBAT->batCacheid);
	}
	
	/* cb_seg */
	for(c = 0; c < num_c_seg; c++)
	{
		aBAT = BATnew(TYPE_void, ct_seg[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_seg[c] = aBAT->batCacheid);
	}
	
	/* cb_dat */
	for(c = 0; c < num_c_dat; c++)
	{
		aBAT = BATnew(TYPE_void, ct_dat[c], 0, PERSISTENT); /* create empty BAT for each column. */
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BATseqbase(aBAT, 0);
		if ( aBAT == NULL)
			throw(MAL,"mseed_create_temp_container",MAL_MALLOC_FAIL);
		BBPkeepref(cb_dat[c] = aBAT->batCacheid);
	}
	
	(tscs+0)->column_bats = cb_fil;
	(tscs+1)->column_bats = cb_seg;
	(tscs+2)->column_bats = cb_dat;
	
	(tscs+0)->column_names = (str*) GDKmalloc(num_c[0]*sizeof(str));
	(tscs+0)->column_types_strs = (str*) GDKmalloc(num_c[0]*sizeof(str));
	for(c = 0; c < num_c[0]; c++)
	{
		(tscs+0)->column_names[c] = GDKstrdup(cn_fil[c]);
		(tscs+0)->column_types_strs[c] = GDKstrdup(cts_fil[c]);
	}
	
	(tscs+1)->column_names = (str*) GDKmalloc(num_c[1]*sizeof(str));
	(tscs+1)->column_types_strs = (str*) GDKmalloc(num_c[1]*sizeof(str));
	for(c = 0; c < num_c[1]; c++)
	{
		(tscs+1)->column_names[c] = GDKstrdup(cn_seg[c]);
		(tscs+1)->column_types_strs[c] = GDKstrdup(cts_seg[c]);
	}
	
	(tscs+2)->column_names = (str*) GDKmalloc(num_c[2]*sizeof(str));
	(tscs+2)->column_types_strs = (str*) GDKmalloc(num_c[2]*sizeof(str));
	for(c = 0; c < num_c[2]; c++)
	{
		(tscs+2)->column_names[c] = GDKstrdup(cn_dat[c]);
		(tscs+2)->column_types_strs[c] = GDKstrdup(cts_dat[c]);
	}
	
	ret_tc->schema_name = sch_name;
	ret_tc->tables_columns = tscs;
	
	ret_tc->table_names = (str*) GDKmalloc(num_tables*sizeof(str));
	ret_tc->num_columns = (int*) GDKmalloc(num_tables*sizeof(int));
	for(t = 0; t < num_tables; t++)
	{
		ret_tc->table_names[t] = GDKstrdup(tn[t]);
		ret_tc->num_columns[t] = num_c[t];
	}
	
	ret_tc->num_tables = num_tables;
	
	return MAL_SUCCEED;
}

/*
 * concatenates num_words_to_concat strings that are in words_to_concat into
 * one string and stores it in ret_concatenated.
 *
 * returns the total_len of the resulting string without the null terminator.
 */
int concatenate_strs(str* words_to_concat, int num_words_to_concat, str* ret_concatenated)
{
	int w;
	int total_len = 1; /* null terminator */
	str tmp;

	for(w = 0; w < num_words_to_concat; w++)
		total_len += strlen(words_to_concat[w]);

	*ret_concatenated = tmp = (str)GDKmalloc(total_len*sizeof(char));

	for(w = 0; w < num_words_to_concat; w++)
	{
		tmp = stpcpy(tmp, words_to_concat[w]);
	}
	*tmp = '\0';

	return total_len-1; /* without null terminator; */
}

/*
 * forms and executes sql 'CREATE FUNCTION' queries according to the attributes of
 * the tables_to_be_filled which are in temp_container tc.
 *
 * returns error or MAL_SUCCEED.
 */
str prepare_insertion(Client cntxt, temp_container* tc)
{
/* form a sql query str like this: */
/* 	CREATE FUNCTION mseed_register_fil(ticket bigint, table_idx int) */
/* 	RETURNS table(file_location string, dataquality char, network string, station string, location string, channel string, encoding tinyint, byte_order boolean) external name registrar.register_table; */

	int t, c;
	str space = " ";
	str comma_space = ", ";

	for(t = 0; t < tc->num_tables; t++)
	{
		int concat_len, num_words_to_concat;
		str q, msg, concatenated=NULL;
		str* words_to_concat;

		if(tc->num_columns[t] <= 0)
			break; /* not a metadata table */

		num_words_to_concat = 4*(tc->num_columns[t]) - 1;
		words_to_concat = (str*)GDKmalloc(num_words_to_concat*sizeof(str));

		for(c = 0; c < tc->num_columns[t]; c++)
		{
			words_to_concat[4*c] = tc->tables_columns[t].column_names[c];
			words_to_concat[4*c+1] = space;
			words_to_concat[4*c+2] = tc->tables_columns[t].column_types_strs[c];
			if(c != tc->num_columns[t]-1)
				words_to_concat[4*c+3] = comma_space;
		}

		concat_len = concatenate_strs(words_to_concat, num_words_to_concat, &concatenated);
		if(concat_len < 1 || concatenated == NULL)
		{
			throw(MAL,"registrar.prepare_insertion",MAL_MALLOC_FAIL);
		}

		q = (str)GDKmalloc(512*sizeof(char));
		sprintf(q, "CREATE FUNCTION %s.%s_%s_reg(ticket bigint, table_idx int) RETURNS table(%s) external name registrar.register_table;\n", tc->schema_name, tc->schema_name, tc->table_names[t], concatenated);

		if((msg =SQLstatementIntern(cntxt,&q,"registrar.create.function",TRUE,FALSE,NULL))!= MAL_SUCCEED)
		{/* create function query not succeeded, what to do */
			return msg;
		}

	}

	return MAL_SUCCEED;

}

/*
 * forms and executes sql 'INSERT INTO ... SELECT * FROM' queries for each of
 * the tables_to_be_filled which are in temp_container tc. Uses the new sql
 * functions created by a prepare_insertion call.
 *
 * returns error or MAL_SUCCEED.
 */
str insert_into_vault(Client cntxt, temp_container* tc)
{
/* form a sql query str like this: */
/* INSERT INTO mseed.files SELECT * FROM mseed_files_reg(ticket, table_idx); */

	int t;
	lng ticket = (lng) (size_t) tc;
	str msg;

	for(t = 0; t < tc->num_tables; t++)
	{
		str q = (str)GDKmalloc(512*sizeof(char));
		
		sprintf(q, "INSERT INTO %s.%s SELECT * FROM %s.%s_%s_reg("LLFMT", %d);\n", tc->schema_name, tc->table_names[t], tc->schema_name, tc->schema_name, tc->table_names[t], ticket, t);

		if((msg =SQLstatementIntern(cntxt,&q,"registrar.insert",TRUE,FALSE,NULL))!= MAL_SUCCEED)
		{/* insert into query not succeeded, what to do */
			return msg;
		}
		GDKfree(q);
	}

	return MAL_SUCCEED;
}

/*
 * forms and executes sql 'DROP FUNCTION' queries for each of
 * the tables_to_be_filled which are in temp_container tc. Drops the new sql
 * functions created by a prepare_insertion call.
 *
 * returns error or MAL_SUCCEED.
 */
str clean_up_after_insertion(Client cntxt, temp_container* tc)
{
	/* form a sql query str like this: */
	/* INSERT INTO mseed.files SELECT * FROM mseed_files_reg(ticket, table_idx); */
	
	int t;
	str msg;
	
	for(t = 0; t < tc->num_tables; t++)
	{
		str s = (str)GDKmalloc(512*sizeof(char));
		
		sprintf(s, "DROP FUNCTION %s.%s_%s_reg(BIGINT, INTEGER);\n", tc->schema_name, tc->schema_name, tc->table_names[t]);
		
		if((msg =SQLstatementIntern(cntxt,&s,"registrar.clean_up_after_insertion",TRUE,FALSE,NULL))!= MAL_SUCCEED)
		{/* drop function not succeeded, what to do */
			return msg;
		}
		GDKfree(s);
	}
	
	return MAL_SUCCEED;
}


/*
 * frees the memory that tc occupies and releases the references to the BATs
 *
 * returns MAL_SUCCEED.
 */
str register_clean_up(temp_container* tc)
{
	int t, c;

	for(t = 0; t < tc->num_tables; t++)
	{
		for(c = 0; c < tc->num_columns[t]; c++)
		{
			BBPdecref(tc->tables_columns[t].column_bats[c], TRUE);
			GDKfree(tc->tables_columns[t].column_names[c]);
			GDKfree(tc->tables_columns[t].column_types_strs[c]);
		}

		GDKfree(tc->tables_columns[t].column_bats);
		GDKfree(tc->table_names[t]);

	}

	GDKfree(tc->tables_columns);
/* 	GDKfree(tc->schema_name); */
	GDKfree(tc->table_names);
	GDKfree(tc->num_columns);

	GDKfree(tc);

	return MAL_SUCCEED;
}


/*
 * appends the metadata of the input "mseed" file provided in the file_path,
 * to the end of BATs of temp_container ret_tc.
 *
 * returns error or MAL_SUCCEED.
 *
 * WARNING: this is the DEVELOPER-PROVIDED function.
 *
 * TODO: A better interface can be provided to submit values for the attributes
 * of tables_to_be_filled.
 */
str mseed_register(int assigned_file_id, str file_path, temp_container* ret_tc)
{

	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int retcode;
	short int verbose = 1;
	BAT *aBAT = NULL;
	int files_done = FALSE;
	timestamp start_timestamp;
	int seq_no_fake = 1;
	lng st;
	str ch = (str) GDKmalloc(2*sizeof(char));
	ch[1] = '\0';

	/* while ((retcode = ms_readmsr (&msr, file_path, 0, NULL, NULL, 1, 0, verbose)) == MS_NOERROR) */
	while ((retcode = ms_readmsr_r (&msfp, &msr, file_path, 0, NULL, NULL, 1, 0, verbose)) == MS_NOERROR)
	{
		if(!files_done)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) file_path, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[1], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = msr->dataquality;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[2], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->network, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[3], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->station, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[4], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->location, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[5], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->channel, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[6], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->encoding), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[7], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[8])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->byteorder), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[8], TRUE);

			files_done = TRUE;
		}

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[0])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[0], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[1])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
/* 		BUNappend(aBAT, (ptr) &(msr->sequence_number), FALSE); */
		BUNappend(aBAT, (ptr) &(seq_no_fake), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[1], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[2])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->reclen), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[2], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[3])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		st = (lng) msr->starttime / 1000;
		MTIMEtimestamp_lng(&start_timestamp, &st);
		BUNappend(aBAT, (ptr) &start_timestamp, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[3], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[4])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->samprate), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[4], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[5])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->samplecnt), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[5], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[6])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		ch[0] = msr->sampletype;
		BUNappend(aBAT, (ptr) ch, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[6], TRUE);

		seq_no_fake++;
	}

	GDKfree(ch);
	
	/* Cleanup memory and close file */
	/* ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0); */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);

	if ( retcode != MS_ENDOFFILE )
		throw(MAL, "mseed_register", "Cannot read %s: %s\n", file_path, ms_errorstr(retcode));

	return MAL_SUCCEED;
}

/*
 * appends the meta-data and actual data of the input "mseed" file provided in the file_path,
 * to the end of BATs of temp_container ret_tc.
 *
 * returns error or MAL_SUCCEED.
 *
 * WARNING: this may be an optional DEVELOPER-PROVIDED function.
 *
 * TODO: A better interface can be provided to submit values for the attributes
 * of tables_to_be_filled.
 */
str mseed_register_and_mount(int assigned_file_id, str file_path, temp_container* ret_tc)
{

	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int retcode;
	short int verbose = 1;
	short int data_flag = 1;
	BAT *aBAT = NULL;
	BAT *btime = NULL, *bdata = NULL, *bfile = NULL, *bseqno = NULL;
	int files_done = FALSE;
	timestamp start_timestamp;
	int seq_no_fake = 1;
	lng st;
	lng i;
	str ch = (str) GDKmalloc(2*sizeof(char));
	ch[1] = '\0';

	/* while ((retcode = ms_readmsr (&msr, file_path, 0, NULL, NULL, 1, data_flag, verbose)) == MS_NOERROR) */
	while ((retcode = ms_readmsr_r (&msfp, &msr, file_path, 0, NULL, NULL, 1, data_flag, verbose)) == MS_NOERROR)
	{
		if(!files_done)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) file_path, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[1], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = msr->dataquality;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[2], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->network, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[3], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->station, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[4], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->location, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[5], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->channel, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[6], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->encoding), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[7], TRUE);

			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[8])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->byteorder), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[8], TRUE);

			files_done = TRUE;
		}

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[0])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[0], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[1])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
/* 		BUNappend(aBAT, (ptr) &(msr->sequence_number), FALSE); */
		BUNappend(aBAT, (ptr) &(seq_no_fake), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[1], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[2])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->reclen), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[2], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[3])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		st = (lng) msr->starttime / 1000;
		MTIMEtimestamp_lng(&start_timestamp, &st);
		BUNappend(aBAT, (ptr) &start_timestamp, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[3], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[4])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->samprate), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[4], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[5])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		BUNappend(aBAT, (ptr) &(msr->samplecnt), FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[5], TRUE);

		if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[6])) == NULL)
			throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
		ch[0] = msr->sampletype;
		BUNappend(aBAT, (ptr) ch, FALSE);
		BBPdecref(ret_tc->tables_columns[1].column_bats[6], TRUE);

		/* mount */
		{
			int seq_no = seq_no_fake;
			double sample_interval = HPTMODULUS / msr->samprate; /* calculate sampling interval from frequency */
			lng sampling_time = msr->starttime;

			lng num_samples = msr->samplecnt;
			int *data_samples = msr->datasamples;

			if ((bfile = BATdescriptor(ret_tc->tables_columns[2].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			if ((bseqno = BATdescriptor(ret_tc->tables_columns[2].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			if ((btime = BATdescriptor(ret_tc->tables_columns[2].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			if ((bdata = BATdescriptor(ret_tc->tables_columns[2].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);

			for(i = 0; i<num_samples; i++)
			{

				timestamp sampling_timestamp;
				lng st = sampling_time / 1000;
				MTIMEtimestamp_lng(&sampling_timestamp, &st);

				/* For each sample add one row to the table */
				BUNappend(bfile, (ptr) &assigned_file_id, FALSE);
				BUNappend(bseqno, (ptr) &seq_no, FALSE);
				BUNappend(btime, (ptr) &sampling_timestamp, FALSE);
				BUNappend(bdata, (ptr) (data_samples+i), FALSE);
				sampling_time += sample_interval;
			}

			BBPdecref(ret_tc->tables_columns[2].column_bats[0], TRUE);
			BBPdecref(ret_tc->tables_columns[2].column_bats[1], TRUE);
			BBPdecref(ret_tc->tables_columns[2].column_bats[2], TRUE);
			BBPdecref(ret_tc->tables_columns[2].column_bats[3], TRUE);
		}

		seq_no_fake++;
	}

	/* Cleanup memory and close file */
	/* ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0); */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);

	if ( retcode != MS_ENDOFFILE )
		throw(MAL, "mseed_register", "Cannot read %s: %s\n", file_path, ms_errorstr(retcode));

	return MAL_SUCCEED;
}

/*
 * appends the metadata of the input "mseed" file provided in the file_path,
 * to the end of BATs of temp_container ret_tc.
 *
 * returns error or MAL_SUCCEED.
 *
 * WARNING: this is the DEVELOPER-PROVIDED function.
 *
 * TODO: A better interface can be provided to submit values for the attributes
 * of tables_to_be_filled.
 */
str mseed_register_segments_mode(int assigned_file_id, str file_path, temp_container* ret_tc)
{
	MSTraceList *mstl = 0;
	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int dataquality = 0;
	int verbose = 1;
	int retcode;
	MSTraceID *id = 0;
	MSTraceSeg *seg = 0;
	double gap;
	double delta;
	int tracecnt = 0;
	int nogap;
	
	timestamp start_timestamp;
	timestamp end_timestamp;
	int files_done = FALSE;
	BAT *aBAT = NULL;
	int segment_id_fake = 0;
	lng st;
	str ch = (str) GDKmalloc(2*sizeof(char));
	ch[1] = '\0';
	
// 	printf(""LLFMT".file\n", file_counter);
	file_counter++;
	mstl = mstl_init (NULL);
	
	/* while ((retcode = ms_readmsr (&msr, file_path, 0, NULL, NULL, 1, 0, verbose)) == MS_NOERROR) */
	while ((retcode = ms_readmsr_r (&msfp, &msr, file_path, 0, NULL, NULL, 1, 0, verbose)) == MS_NOERROR)
	{
		if(!files_done)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) file_path, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[1], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = msr->dataquality;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[2], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->network, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[3], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->station, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[4], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->location, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[5], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->channel, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[6], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->encoding), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[7], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[8])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->byteorder), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[8], TRUE);
			
			files_done = TRUE;
		}
		
		/* collect all records in a tracelist */
		mstl_addmsr (mstl, msr, dataquality, 1, timetol, sampratetol);
		
	}
	
	if ( retcode != MS_ENDOFFILE )
		throw(MAL, "mseed_register_segments_mode", "Cannot read %s: %s\n", file_path, ms_errorstr(retcode));
	
	if (!mstl)
	{
		throw(MAL, "mseed_register_segments_mode", RUNTIME_OBJECT_MISSING);
	}
	
	/* Loop through trace list */
	id = mstl->traces;
	while (id)
	{
		/* Loop through segment list */
		seg = id->first;
		while (seg)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(segment_id_fake), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[1], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			st = (lng) seg->starttime / 1000;
			MTIMEtimestamp_lng(&start_timestamp, &st);
			BUNappend(aBAT, (ptr) &start_timestamp, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[2], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			st = (lng) seg->endtime / 1000;
			MTIMEtimestamp_lng(&end_timestamp, &st);
			BUNappend(aBAT, (ptr) &end_timestamp, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[3], TRUE);
			
			/* Compute gap with previous segment */
			gap = 0.0;
			nogap = 0;
			
			if ( seg->prev )
				gap = (double) (seg->starttime - seg->prev->endtime) / HPTMODULUS;
			else
				nogap = 1;
			
			/* Check that any overlap is not larger than the trace coverage */
			if ( gap < 0.0 )
			{
				delta = ( seg->samprate ) ? (1.0 / seg->samprate) : 0.0;
				
				if ( (gap * -1.0) > (((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta) )
					gap = -(((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta);
			}
						
// 			printf("%d. segment, frq: %lf, sampcnt: %lld\n", segment_id_fake, seg->samprate, (long long int)seg->samplecnt);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			if(nogap)
				BUNappend(aBAT, (ptr) &(dbl_nil), FALSE);
			else
				BUNappend(aBAT, (ptr) &(gap), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[4], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(seg->samprate), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[5], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(seg->samplecnt), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[6], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = seg->sampletype;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[7], TRUE);
			
			segment_id_fake++;
			seg = seg->next;
		}
		
		tracecnt++;
		id = id->next;
	}
	
	if(tracecnt != 1)
		printf("Trace count is greater than 1. Total %d traces in file %s", tracecnt, file_path);
	
	GDKfree(ch);
	
	/* Cleanup memory and close file */
	/* ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0); */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);
	if ( mstl )
		mstl_free (&mstl, 0);
	
	if ( retcode != MS_ENDOFFILE )
		throw(MAL, "mseed_register", "Cannot read %s: %s\n", file_path, ms_errorstr(retcode));
	
	num_registered_files++;
	
	return MAL_SUCCEED;
}


/*
 * appends the meta-data and actual data of the input "mseed" file provided in the file_path,
 * to the end of BATs of temp_container ret_tc.
 *
 * returns error or MAL_SUCCEED.
 *
 * WARNING: this may be an optional DEVELOPER-PROVIDED function.
 *
 * TODO: A better interface can be provided to submit values for the attributes
 * of tables_to_be_filled.
 */
str mseed_register_and_mount_segments_mode(int assigned_file_id, str file_path, temp_container* ret_tc)
{
	
	MSTraceList *mstl = 0;
	MSRecord *msr = NULL;
	MSFileParam *msfp = NULL;
	int dataquality = 0;
	int retcode;
	MSTraceID *id = 0;
	MSTraceSeg *seg = 0;
	double gap;
	double delta;
	int tracecnt = 0;
	int nogap;
	
	BAT *aBAT = NULL;
	int segment_id_fake = 0;
	lng st;
	str ch = (str) GDKmalloc(2*sizeof(char));
	timestamp start_timestamp;
	timestamp end_timestamp;
	int files_done = FALSE;
	BAT *btime = NULL, *bvalue = NULL, *bfile = NULL, *bsegid = NULL;
	
	short int data_flag = 1;
	ch[1] = '\0';
	
	printf(""LLFMT".file\n", file_counter);
	file_counter++;
	
	mstl = mstl_init (NULL);
	
	/* while ((retcode = ms_readmsr (&msr, file_path, 0, NULL, NULL, 1, 0, verbose)) == MS_NOERROR) */
	while ((retcode = ms_readmsr_r (&msfp, &msr, file_path, 0, NULL, NULL, 1, data_flag, verbose)) == MS_NOERROR)
	{
		if(!files_done)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) file_path, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[1], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = msr->dataquality;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[2], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->network, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[3], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->station, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[4], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->location, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[5], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) msr->channel, FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[6], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->encoding), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[7], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[0].column_bats[8])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(msr->byteorder), FALSE);
			BBPdecref(ret_tc->tables_columns[0].column_bats[8], TRUE);
			
			files_done = TRUE;
		}
		
		/* collect all records in a tracelist */
		mstl_addmsr (mstl, msr, dataquality, 1, timetol, sampratetol);
		
	}
	
	if ( retcode != MS_ENDOFFILE )
		throw(MAL, "mseed_register_and_mount_segments_mode", "Cannot read %s: %s\n", file_path, ms_errorstr(retcode));
	
	if (!mstl)
	{
		throw(MAL, "mseed_register_and_mount_segments_mode", RUNTIME_OBJECT_MISSING);
	}
	
	/* Loop through trace list */
	id = mstl->traces;
	while (id)
	{
		/* Loop through segment list */
		seg = id->first;
		while (seg)
		{
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[0])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &assigned_file_id, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[0], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[1])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(segment_id_fake), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[1], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[2])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			st = (lng) seg->starttime / 1000;
			MTIMEtimestamp_lng(&start_timestamp, &st);
			BUNappend(aBAT, (ptr) &start_timestamp, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[2], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[3])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			st = (lng) seg->endtime / 1000;
			MTIMEtimestamp_lng(&end_timestamp, &st);
			BUNappend(aBAT, (ptr) &end_timestamp, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[3], TRUE);
			
			/* Compute gap with previous segment */
			gap = 0.0;
			nogap = 0;
			
			if ( seg->prev )
				gap = (double) (seg->starttime - seg->prev->endtime) / HPTMODULUS;
			else
				nogap = 1;
			
			/* Check that any overlap is not larger than the trace coverage */
			if ( gap < 0.0 )
			{
				delta = ( seg->samprate ) ? (1.0 / seg->samprate) : 0.0;
				
				if ( (gap * -1.0) > (((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta) )
					gap = -(((double)(seg->endtime - seg->starttime)/HPTMODULUS) + delta);
			}
			
			printf("%d. segment, frq: %lf, sampcnt: %lld\n", segment_id_fake, seg->samprate, (long long int)seg->samplecnt);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[4])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			if(nogap)
				BUNappend(aBAT, (ptr) &(dbl_nil), FALSE);
			else
				BUNappend(aBAT, (ptr) &(gap), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[4], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[5])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(seg->samprate), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[5], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[6])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			BUNappend(aBAT, (ptr) &(seg->samplecnt), FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[6], TRUE);
			
			if ((aBAT = BATdescriptor(ret_tc->tables_columns[1].column_bats[7])) == NULL)
				throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
			ch[0] = seg->sampletype;
			BUNappend(aBAT, (ptr) ch, FALSE);
			BBPdecref(ret_tc->tables_columns[1].column_bats[7], TRUE);
		
			/* mount */
			{
				lng i;
				int segment_id = segment_id_fake;
				double sample_interval = HPTMODULUS / seg->samprate; /* calculate sampling interval from frequency */
				lng sampling_time = seg->starttime;
				
				lng num_samples = seg->numsamples;
				int *data_samples = seg->datasamples;
				
				if ((bfile = BATdescriptor(ret_tc->tables_columns[2].column_bats[0])) == NULL)
					throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
				if ((bsegid = BATdescriptor(ret_tc->tables_columns[2].column_bats[1])) == NULL)
					throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
				if ((btime = BATdescriptor(ret_tc->tables_columns[2].column_bats[2])) == NULL)
					throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
				if ((bvalue = BATdescriptor(ret_tc->tables_columns[2].column_bats[3])) == NULL)
					throw(MAL, "mseed_register", RUNTIME_OBJECT_MISSING);
				
				for(i = 0; i<num_samples; i++)
				{
					
					timestamp sampling_timestamp;
					lng st = sampling_time / 1000;
					MTIMEtimestamp_lng(&sampling_timestamp, &st);
					
					/* For each sample add one row to the table */
					BUNappend(bfile, (ptr) &assigned_file_id, FALSE);
					BUNappend(bsegid, (ptr) &segment_id, FALSE);
					BUNappend(btime, (ptr) &sampling_timestamp, FALSE);
					BUNappend(bvalue, (ptr) (data_samples+i), FALSE);
					sampling_time += sample_interval;
				}
				
				BBPdecref(ret_tc->tables_columns[2].column_bats[0], TRUE);
				BBPdecref(ret_tc->tables_columns[2].column_bats[1], TRUE);
				BBPdecref(ret_tc->tables_columns[2].column_bats[2], TRUE);
				BBPdecref(ret_tc->tables_columns[2].column_bats[3], TRUE);
			}
		
			segment_id_fake++;
			seg = seg->next;
		}
		tracecnt++;
		id = id->next;
	}
	
	if(tracecnt != 1)
		printf("Warning! Unexpectedly, trace count is greater than 1 in a file! Total %d traces in file %s", tracecnt, file_path);
	
	GDKfree(ch);
	
	/* Cleanup memory and close file */
	/* ms_readmsr (&msr, NULL, 0, NULL, NULL, 0, 0, 0); */
	ms_readmsr_r (&msfp, &msr, NULL, 0, NULL, NULL, 0, 0, 0);
	if (mstl)
		mstl_free (&mstl, 0);
	
	num_registered_files++;
	
	return MAL_SUCCEED;
}

// static str
// runSQLQuery(Client c, char **result, char * query)
// {
// 	str msg = MAL_SUCCEED;
// 	str qmsg = MAL_SUCCEED;
// 	char * resultstring = NULL;
// 	struct buffer * resultbuffer;
// 	stream * resultstream;
// // 	Client c;
// 	int len = 0;
// // 	backend *be;
// 	
// 	resultbuffer = buffer_create(BLOCK);
// 	resultstream = buffer_wastream(resultbuffer, "resultstring");
// 	
// // 	c = MCinitClient(CONSOLE, fin, resultstream);
// // 	c->nspace = newModule(NULL, putName("user", 4));
// // 	
// // 	// TODO: lookup user_id in bat
// // 	c->user = 1;
// // 	initLibraries();
// // 	msg = setScenario(c, "sql");
// // 	msg = SQLinitClient(c);
// // 	MSinitClientPrg(c, "user", "main");
// // 	(void) MCinitClientThread(c);
// // 	// TODO: check that be <> NULL
// // // 	be = (backend*)c->sqlcontext;
// 	
// 	c->fdout = resultstream;
// 	
// 	qmsg = SQLstatementIntern(c, &query, "rest", TRUE, TRUE);
// 	if (qmsg == MAL_SUCCEED) {
// 		resultstring = buffer_get_buf(resultbuffer);
// 		*result = GDKstrdup(resultstring);
// 		free(resultstring);
// 	} else {
// 			len = strlen(qmsg) + 19;
// 			resultstring = malloc(len);
// 			snprintf(resultstring, len, "{ \"error\": \"%s\" }\n", qmsg);
// 			*result = GDKstrdup(resultstring);
// 			free(resultstring);
// 		}
// 	buffer_destroy(resultbuffer);
// 	c->fdout = GDKstdout;
// 	return msg;
// }

void *register_files(void *args)
{
// 	temp_container *tc;
	lng i;
	str err = NULL;
	int start, finish;
	int current_file_id;
	
	thread_argv targv = *((thread_argv*)args);
	
	/* create temp_container */
// 	tc = (temp_container*)GDKmalloc(sizeof(temp_container));
// 	assert(tc != NULL);

	err = create_temp_container_segments_mode(targv.mvc, targv.tc, targv.mode); /* depending on design can get different argument(s) */
// 	if(targv.mode == 0)
// 		err = mseed_create_temp_container_segments_mode(targv.tc); /* depending on design can get different argument(s) */
// 	else
// 		err = mseed_create_temp_container_with_data_tables_segments_mode(targv.tc); /* depending on design can get different argument(s) */
	if(err != MAL_SUCCEED)
	{/* temp_container creation failed, what to do */
		throw(MAL,"registrar.register_repo", "temp_container creation failed in thread %d: %s\n", targv.tid, err);
	}
	
	start = GDKms();
	
	current_file_id = targv.base_file_id;
	
	/* loop through the file_paths in repo */
	if(targv.mode == 0)
	{
		for(i = targv.loop_start; i < targv.loop_end; i++)
		{
			err = mseed_register_segments_mode(current_file_id, targv.file_paths[i], targv.tc);
			if(err != MAL_SUCCEED)
			{/* current file cannot be registered, what to do */
				/*throw(MAL,"registrar.register_repo", "Current file cannot be registered: %s\n", err); */
				printf("registrar.register_repo: current file cannot be registered in thread %d: %s\n", targv.tid, err);
			}
			
			current_file_id++;
		}
	}
	else
	{
		for(i = targv.loop_start; i < targv.loop_end; i++)
		{
			err = mseed_register_and_mount_segments_mode(current_file_id, targv.file_paths[i], targv.tc);
			if(err != MAL_SUCCEED)
			{/* current file cannot be registered, what to do */
				/* throw(MAL,"registrar.register_repo", "Current file cannot be registered: %s\n", err); */
				printf("registrar.register_repo: current file cannot be registered and/or mounted in thread %d: %s\n", targv.tid, err);
			}
			
			current_file_id++;
		}
	}
	finish = GDKms();
	printf("# In thread %d, time for extraction and transformation of (meta-)data: %d milliseconds\n", targv.tid, finish - start);
	
	return NULL;
}

/*
 * takes a repository path repo_path, finds out the files in it, creates a
 * temp_container of the metadata to be inserted, for each file calls the
 * developer-provided register function which fills in the temp_container,
 * then using prepare_insertion and insert_into_vault calls appends the
 * metadata to the tables_to_be_filled.
 *
 * returns error or MAL_SUCCEED.
 *
 * can be called from MAL or SQL levels.
 */
str register_repo(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	str *repo_path = (str*) getArgReference(stk,pci,pci->retc); /* arg 1: repo_path */
	int mode = *(int*) getArgReference(stk,pci,pci->retc+1); /* arg 2: mode 0:register only, mode 1: register+mount */
	int num_threads = *(int*) getArgReference(stk,pci,pci->retc+2); /* arg 3: 1: no threads, >1: multi-threaded */
	str *file_paths = NULL;
	lng num_file_paths;
	temp_container *tc;
	temp_container **thread_tcs;
	lng i;
	str err = NULL;
	int start, finish, total_start, total_finish;
	mvc *m = NULL;
	int max_file_id = -1;
	int current_file_id;
	ValRecord *v;
	str q, msg, s;

	/* fetch file_paths from repo_path */
	num_file_paths = get_file_paths(*repo_path, &file_paths);
	if(num_file_paths < 1 && file_paths == NULL)
	{/* problematic repo, what to do */
		throw(MAL,"registrar.register_repo", "Problematic repository: %s\n", err);
	}

	total_start = GDKms();
	
	if((err = getSQLContext(cntxt, mb, &m, NULL))!= MAL_SUCCEED)
	{/* getting mvc failed, what to do */
		return err;
	}
	
	/* reset the file counters */
	file_counter = 0;
	num_registered_files = 0;
	
	v = stack_get_var(m, "fid");
	
	if(v == NULL)
	{
		q = "DECLARE fid INTEGER;\n";
		
		if((msg =SQLstatementIntern(cntxt,&q,"registrar.register_repo",TRUE,FALSE,NULL))!= MAL_SUCCEED)
		{/* declare fid not succeeded, what to do */
			return msg;
		}
	}
	
	s = (str)GDKmalloc(256*sizeof(char));
	
	sprintf(s, "SELECT MAX(file_id) INTO fid FROM %s.files;\n", m->session->schema_name);
	
	if((msg =SQLstatementIntern(cntxt,&s,"registrar.register_repo",TRUE,FALSE,NULL))!= MAL_SUCCEED)
	{/* insert into query not succeeded, what to do */
		return msg;
	}
	
	v = stack_get_var(m, "fid");
	
	if(v->val.ival < 0)
		max_file_id = -1;
	else
		max_file_id = v->val.ival;
	current_file_id = max_file_id + 1;
	
	GDKfree(s);
	
	if(num_threads > 1)
	{
		
		/* multi-threaded */
		int j; /* thread index */
		lng loop_start = 0;
		lng num_file_paths_per_thread = num_file_paths / num_threads;
		pthread_t *threads = (pthread_t*)GDKmalloc(num_threads*sizeof(pthread_t));
		thread_argv *targvs = (thread_argv*)GDKmalloc(num_threads*sizeof(thread_argv));
		thread_tcs = (temp_container**)GDKmalloc(num_threads*sizeof(temp_container*));
		
		for(j = 0; j < num_threads; j++)
		{
			thread_tcs[j] = (temp_container*)GDKmalloc(num_threads*sizeof(temp_container));
		}
		
		for(j = 0; j < num_threads; j++)
		{
			targvs[j].tid = j;
			targvs[j].file_paths = file_paths;
			targvs[j].loop_start = loop_start;
			if(j != (num_threads - 1))
				targvs[j].loop_end = loop_start + num_file_paths_per_thread;
			else
				targvs[j].loop_end = num_file_paths;
			targvs[j].base_file_id = max_file_id + 1 + targvs[j].loop_start;
			loop_start = targvs[j].loop_end;
			targvs[j].mode = mode;
			targvs[j].mvc = m;
			
			targvs[j].tc = thread_tcs[j];
			
			pthread_create(&threads[j], NULL, register_files, (void *) &targvs[j]);
		}
		
		for(j = 0; j < num_threads; j++)
		{
			pthread_join(threads[j], NULL); 
		}
		
		/* prepare sql functions for inserting temp_container into tables_to_be_filled */
		err = prepare_insertion(cntxt, thread_tcs[0]);
		if(err != MAL_SUCCEED)
		{/* preparing the insertion failed, what to do */
			throw(MAL,"registrar.register_repo", "Insertion prepare failed: %s\n", err);
		}
		
		start = GDKms();
		for(j = 0; j < num_threads; j++)
		{
			/* insert temp_container into tables_to_be_filled */
			err = insert_into_vault(cntxt, thread_tcs[j]);
			if(err != MAL_SUCCEED)
			{/* inserting the temp_container into one of the tables failed, what to do */
				throw(MAL,"registrar.register_repo", "Inserting the temp_container into one of the tables failed: %s\n", err);
			}
		}
		
		err = clean_up_after_insertion(cntxt, thread_tcs[0]);
		if(err != MAL_SUCCEED)
		{/* cleaning up the insertion failed, what to do */
			throw(MAL,"registrar.register_repo", "Insertion clean up failed: %s\n", err);
		}
		
		for(j = 0; j < num_threads; j++)
		{	
			err = register_clean_up(thread_tcs[j]);
			if(err != MAL_SUCCEED)
			{/* inserting the temp_container into one of the tables failed, what to do */
				throw(MAL,"registrar.register_repo", "Cleaning up the temp_container failed: %s\n", err);
			}
		}
		
		finish = GDKms();
		printf("# Time for loading of (meta-)data: %d milliseconds\n", finish - start);
		
		
		GDKfree(targvs);
		GDKfree(threads);
	}
	else
	{
		/* single threaded */
		/* create temp_container */
		tc = (temp_container*)GDKmalloc(sizeof(temp_container));
		assert(tc != NULL);
		
		err = create_temp_container_segments_mode(m, tc, mode); /* depending on design can get different argument(s) */
		
		if(err != MAL_SUCCEED)
		{/* temp_container creation failed, what to do */
			throw(MAL,"registrar.register_repo", "temp_container creation failed: %s\n", err);
		}

		start = GDKms();
		/* loop through the file_paths in repo */
		if(mode == 0)
		{
			for(i = 0; i < num_file_paths; i++)
			{
				err = mseed_register_segments_mode(current_file_id, file_paths[i], tc);
				if(err != MAL_SUCCEED)
				{/* current file cannot be registered, what to do */
					/* throw(MAL,"registrar.register_repo", "Current file cannot be registered: %s\n", err); */
					printf("registrar.register_repo: current file cannot be registered: %s\n", err);
				}
				
				current_file_id++;
			}
		}
		else
		{
			for(i = 0; i < num_file_paths; i++)
			{
				err = mseed_register_and_mount_segments_mode(current_file_id, file_paths[i], tc);
				if(err != MAL_SUCCEED)
				{/* current file cannot be registered, what to do */
					/* throw(MAL,"registrar.register_repo", "Current file cannot be registered: %s\n", err); */
					printf("registrar.register_repo: current file cannot be registered and/or mounted: %s\n", err);
				}
				
				current_file_id++;
			}
		}
		finish = GDKms();
		printf("# Time for extraction and transformation of (meta-)data: %d milliseconds\n", finish - start);

		start = GDKms();
		/* prepare sql functions for inserting temp_container into tables_to_be_filled */
		err = prepare_insertion(cntxt, tc);
		if(err != MAL_SUCCEED)
		{/* preparing the insertion failed, what to do */
			throw(MAL,"registrar.register_repo", "Insertion prepare failed: %s\n", err);
		}

		/* insert temp_container into tables_to_be_filled */
		err = insert_into_vault(cntxt, tc);
		if(err != MAL_SUCCEED)
		{/* inserting the temp_container into one of the tables failed, what to do */
			throw(MAL,"registrar.register_repo", "Inserting the temp_container into one of the tables failed: %s\n", err);
		}
		
		err = clean_up_after_insertion(cntxt, tc);
		if(err != MAL_SUCCEED)
		{/* cleaning up the insertion failed, what to do */
			throw(MAL,"registrar.register_repo", "Insertion clean up failed: %s\n", err);
		}

		err = register_clean_up(tc);
		if(err != MAL_SUCCEED)
		{/* inserting the temp_container into one of the tables failed, what to do */
			throw(MAL,"registrar.register_repo", "Cleaning up the temp_container failed: %s\n", err);
		}
		
		finish = GDKms();
		printf("# Time for loading of (meta-)data: %d milliseconds\n", finish - start);

	}
	
	if(mvc_commit(m, 0, NULL) < 0)
	{/* committing failed */
		throw(MAL,"registrar.register_repo", "committing failed\n");
	}
	
	GDKfree(file_paths);
	
	total_finish = GDKms();
	
	printf("#. Average Time for loading (meta-)data of one file: %lf milliseconds\n", (((double)total_finish - total_start)/num_file_paths));
	printf("#. Number of files registered | failed registration: "LLFMT" | "LLFMT".\n", num_registered_files, num_file_paths - num_registered_files);
	
	return MAL_SUCCEED;
}

/*
 * maps the BATs of temp_container to the columns of one of the tables_to_be_filled,
 * because SQL does not allow functions to return arbitrary tables without predefining
 * them. prepare_insertion functions predefines them and this function does the mapping.
 *
 * takes temp_container pointer as a lng (ticket), since it is only called from
 * SQL level. table_idx specifies for which of the tables it is called for.
 *
 * returns error or MAL_SUCCEED.
 */
str register_table(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int c;
	lng ticket = *(lng*) getArgReference(stk,pci,pci->retc); /* arg 1: ticket to the temp_container */
	int table_idx = *(int*) getArgReference(stk,pci,pci->retc+1); /* arg 2: index of the table to be registered in the temp_container */

	temp_container *tc = (temp_container*) (size_t) ticket; /* filled temp_container taken */

	cntxt = cntxt; /* to escape 'unused' parameter error. */
	mb = mb; /* to escape 'unused' parameter error. */

	if(pci->retc != tc->num_columns[table_idx])
	{/* inconsistency in the number of return BATs, what to do */
		throw(MAL,"registrar.register_table", OPERATION_FAILED);
	}

	for(c = 0; c < pci->retc; c++)
	{
		*(int*) getArgReference(stk,pci,c) = tc->tables_columns[table_idx].column_bats[c];
		BBPincref(tc->tables_columns[table_idx].column_bats[c], TRUE);
	}

	return MAL_SUCCEED;
}










