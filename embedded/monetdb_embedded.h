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
#ifndef _EMBEDDED_LIB_
#define _EMBEDDED_LIB_

#include "monetdb_config.h"
#include "gdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__)
#if !defined(LIBEMBEDDED)
#define embedded_export extern __declspec(dllimport)
#else
#define embedded_export extern __declspec(dllexport)
#endif
#else
#define embedded_export extern
#endif

typedef struct append_data {
	char* colname;
	size_t batid; /* Disclaimer: this header is GDK-free */
} append_data;

typedef struct {
	unsigned char day;
	unsigned char month;
	int year;
} monetdb_data_date;

typedef struct {
	unsigned int ms;
	unsigned char seconds;
	unsigned char minutes;
	unsigned char hours;
} monetdb_data_time;

typedef struct {
	monetdb_data_date date;
	monetdb_data_time time;
} monetdb_data_timestamp;

typedef struct {
	size_t size;
	void* data;
} monetdb_data_blob;

typedef enum  {
	monetdb_int8_t, monetdb_int16_t, monetdb_int32_t, monetdb_int64_t, monetdb_size_t,
	monetdb_float, monetdb_double,
	monetdb_str, monetdb_blob,
	monetdb_date, monetdb_time, monetdb_timestamp
} monetdb_types;

typedef struct {
	monetdb_types type;
	void *data;
	size_t count;
	char* name;
} monetdb_column;

typedef struct {
	size_t nrows;
	size_t ncols;
	char type;
	size_t id;
} monetdb_result;

typedef void* monetdb_connection;

#define DEFAULT_STRUCT_DEFINITION(ctype, typename) \
	typedef struct                                 \
	{                                              \
		monetdb_types type;                        \
		ctype *data;                               \
		size_t count;                              \
		ctype null_value;                          \
		double scale;                              \
		int (*is_null)(ctype value);               \
	} monetdb_column_##typename

DEFAULT_STRUCT_DEFINITION(int8_t, int8_t);
DEFAULT_STRUCT_DEFINITION(int16_t, int16_t);
DEFAULT_STRUCT_DEFINITION(int32_t, int32_t);
DEFAULT_STRUCT_DEFINITION(int64_t, int64_t);
DEFAULT_STRUCT_DEFINITION(size_t, size_t);

DEFAULT_STRUCT_DEFINITION(float, float);
DEFAULT_STRUCT_DEFINITION(double, double);

DEFAULT_STRUCT_DEFINITION(char *, str);
DEFAULT_STRUCT_DEFINITION(monetdb_data_blob, blob);

DEFAULT_STRUCT_DEFINITION(monetdb_data_date, date);
DEFAULT_STRUCT_DEFINITION(monetdb_data_time, time);
DEFAULT_STRUCT_DEFINITION(monetdb_data_timestamp, timestamp);

embedded_export str monetdb_connect(monetdb_connection *conn);
embedded_export str monetdb_disconnect(monetdb_connection conn);
embedded_export str monetdb_startup(char* dbdir, char silent, char sequential);
embedded_export int monetdb_is_initialized(void);

embedded_export str monetdb_set_autocommit(monetdb_connection conn, char value);
embedded_export str monetdb_query(monetdb_connection conn, char* query, monetdb_result** result, lng* affected_rows, lng* prepare_id);

embedded_export str monetdb_result_fetch(monetdb_column** res, monetdb_result* mres, size_t column_index);
embedded_export str monetdb_result_fetch_rawcol(void** res, monetdb_result* mres, size_t column_index); // actually a res_col

embedded_export str monetdb_clear_prepare(monetdb_connection conn, lng id);
embedded_export str monetdb_send_close(monetdb_connection conn, lng id);

embedded_export str monetdb_append(monetdb_connection conn, const char* schema, const char* table, append_data *data, size_t column_count);
embedded_export str monetdb_cleanup_result(monetdb_connection conn, monetdb_result* result);
embedded_export str monetdb_get_columns(monetdb_connection conn, const char* schema_name, const char *table_name, size_t *column_count, char ***column_names, int **column_types);

embedded_export str monetdb_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
