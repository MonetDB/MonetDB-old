
#include "monetdb_config.h"
#include "stream.h"
#include "mstring.h"
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include "mapi.h"
#include <monetdbe.h>

extern int dump_database(Mapi mid, stream *toConsole, bool describe, bool useInserts);

#define error(msg) {fprintf(stderr, "Failure: %s\n", msg); return -1;}

int
main(void) 
{
	char* err = NULL;
	Mapi mid = (Mapi)malloc(sizeof(struct MapiStruct));

	if ((mid->msg = monetdbe_open(&mid->mdbe, NULL)) != NULL)
		error(mid->msg);

	if ((err = monetdbe_query(mid->mdbe, "CREATE TABLE test (b bool, t tinyint, s smallint, x integer, l bigint, "
#ifdef HAVE_HGE
		"h hugeint, "
#else
		"h bigint, "
#endif
		"f float, d double, y string)", NULL, NULL)) != NULL)
		error(err)
	if ((err = monetdbe_query(mid->mdbe, "INSERT INTO test VALUES (TRUE, 42, 42, 42, 42, 42, 42.42, 42.42, 'Hello'), (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'World')", NULL, NULL)) != NULL)
		error(err)

	/* open file stream */
	stream *fd = open_wastream("/tmp/backup");

	if (dump_database(mid, fd, 0, 0)) {
		if (mid->msg)
			error(mid->msg)
		fprintf(stderr, "database backup failed\n");
	}
	close_stream(fd);

	if ((mid->msg = monetdbe_close(mid->mdbe)) != NULL)
		error(mid->msg);
}
