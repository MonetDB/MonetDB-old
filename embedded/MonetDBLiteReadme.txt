MonetDBLite is not set to compile by default, enable it by setting the cache variable ENABLE_EMBEDDED to SHARED or
OBJECT. The former is used to build MonetDBLite as a C shared library. The latter compiles and generates the relocatable
files without any linking. This will be the better approach to compile the language bindings I think.

To simplify the compilation process set the following variables:

cmake -DCMAKE_INSTALL_PREFIX=<build dir> -DENABLE_EMBEDDED=SHARED -DENABLE_ODBC=NO -DENABLE_MAPI=NO -DENABLE_GDK=NO \
      -DENABLE_MONETDB5=NO -DENABLE_SQL=NO -DENABLE_TESTING=NO <source dir>

The ABI is listed on the monetdb_embedded.h header file.
- The major changes are more C99 compliance and most of the calls return a char* which is set to non-NULL whenever an
error is thrown. The error message must be freed with "void freeException(str)" function call.
- The append_data struct is gone, because the "colname" field was never used, instead only the batids are passed as
argument.
- The functions monetdb_result_fetch and monetdb_result_fetch_rawcol take monetdb_connection as the first parameter.
- If libmonetdblite is on a different directory, related to the executable and libmonetdblite's directory is not set on
LD_LIBRARY_PATH or equivalent, use the "str initLinker(const char* path)" function BEFORE starting the database to tell
where it is located.

The updated readme.c example:
-----------------------------------------------------------------------------------------------------------------------
#include "monetdb_embedded.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#define error(msg) {fprintf(stderr, "Failure: %s\n", msg); return -1;}

int
main(void)
{
	char* err = NULL;
	monetdb_connection conn = NULL;
	monetdb_result* result = NULL;

	// first argument is a string for the db directory or NULL for in-memory mode
	if ((err = monetdb_startup(NULL, 1, 0)) != NULL)
		error(err)
	if ((err = monetdb_connect(&conn)) != NULL)
		error(err)
	if ((err = monetdb_query(conn, "CREATE TABLE test (x integer, y string)", NULL, NULL, NULL)) != NULL)
		error(err)
	if ((err = monetdb_query(conn, "INSERT INTO test VALUES (42, 'Hello'), (NULL, 'World')", NULL, NULL, NULL)) != NULL)
		error(err)
	if ((err = monetdb_query(conn, "SELECT x, y FROM test; ", &result, NULL, NULL)) != NULL)
		error(err)

	fprintf(stdout, "Query result with %zu cols and %"PRId64" rows\n", result->ncols, result->nrows);
	for (int64_t r = 0; r < result->nrows; r++) {
		for (size_t c = 0; c < result->ncols; c++) {
			monetdb_column* rcol;
			if ((err = monetdb_result_fetch(conn, &rcol, result, c)) != NULL)
				error(err)
			switch (rcol->type) {
				case monetdb_int32_t: {
					monetdb_column_int32_t * col = (monetdb_column_int32_t *) rcol;
					if (col->data[r] == col->null_value) {
						printf("NULL");
					} else {
						printf("%d", col->data[r]);
					}
					break;
				}
				case monetdb_str: {
					monetdb_column_str * col = (monetdb_column_str *) rcol;
					if (col->is_null(col->data[r])) {
						printf("NULL");
					} else {
						printf("%s", (char*) col->data[r]);
					}
					break;
				}
				default: {
					printf("UNKNOWN");
				}
			}

			if (c + 1 < result->ncols) {
				printf(", ");
			}
		}
		printf("\n");
	}

	if ((err = monetdb_cleanup_result(conn, result)) != NULL)
		error(err)
	if ((err = monetdb_disconnect(conn)) != NULL)
		error(err)
	if ((err = monetdb_shutdown()) != NULL)
		error(err)
	return 0;
}
-----------------------------------------------------------------------------------------------------------------------
To compile with GCC on Linux do (note that cmake installs libraries on lib64 directory on 64-bit architectures on
certain operating systems):

gcc -o readme readme.c -I<build dir>/include/monetdb -L<build dir>/lib -lmonetdblite

It should produce the "readme" executable output the following:
Query result with 2 cols and 2 rows
42, Hello
NULL, World
