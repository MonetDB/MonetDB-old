/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/* Author(s) M.L. Kersten
 * Module import
 * The import statement simple switches the parser to a new input file, which
 * takes precedence. The context for which the file should be interpreted
 * is determined by the module name supplied.
 * Typically this involves a module, whose definitions are stored at
 * a known location.
 * The import context is located. If the module already exists,
 * we should silently skip parsing the file. This is handled at the parser level.
 * The files are extracted from a default location,
 * namely the DBHOME/modules directory.
 *
 * If the string starts with '/' or '~' the context is not changed.
 *
 * Every IMPORT statement denotes a possible dynamic load library.
 * Make sure it is loaded as well.
*/

#include "monetdb_config.h"
#include "mal_import.h"
#include "mal_builder.h"
#include "mal_interpreter.h"
#include "mal_linker.h"		/* for loadModuleLibrary() */
#include "mal_parser.h"
#include "mal_private.h"
#include "mal_exception.h"

/* #define _DEBUG_IMPORT_*/
void
slash_2_dir_sep(str fname)
{
	char *s;

	for (s = fname; *s; s++)
		if (*s == '/')
			*s = DIR_SEP;
}

static str
malResolveFile(str fname)
{
	char path[PATHLENGTH];
	str script;

	snprintf(path, PATHLENGTH, "%s", fname);
	slash_2_dir_sep(path);
	if ((script = MSP_locate_script(path)) == NULL) {
		/* this function is also called for scripts that are not located
		 * in the modpath, so if we can't find it, just default to
		 * whatever was given, as it can be in current dir, or an
		 * absolute location to somewhere */
		script = GDKstrdup(fname);
	}
	return script;
}

static stream *
malOpenSource(str file)
{
	stream *fd = NULL;

	if (file)
		fd = open_rastream(file);
	return fd;
}

/*
 * Beware that we have to isolate the execution of the source file
 * in its own environment. 
*/

#ifdef HAVE_EMBEDDED
extern char* mal_init_inline;
#endif

/*File and input processing
 * A recurring situation is to execute a stream of simple MAL instructions
 * stored on a file or comes from standard input. 
 * An Include operation simply pushes the input stream on the LIFO queue.
 */
str
evalFile(Client cntxt, str fname, int listing, int included)
{
	Client c;
	stream *fd;
	str p, filename, base;
	str files[MAXMULTISCRIPT];
	int cnt, i;
	str msg = MAL_SUCCEED;

	base = fname = malResolveFile(fname);
	if (fname == NULL) 
		throw(MAL,"mal.import", "#WARNING: could not open file: %s\n", fname);

	// load each individual file in the list
	// we should follow the sort order
	files[0] = filename = fname;
	cnt = 1;
	while ((p=strchr(filename, PATH_SEP)) != NULL) {
		*p = 0;
		//fprintf(stderr,"#loading %s\n", filename);
		filename = p+1;
		files[cnt++]= filename;
	}
	for(i=cnt-1; i>=0; i--){
		fd = malOpenSource(files[i]);
		if (fd == 0 || mnstr_errnr(fd) == MNSTR_OPEN_ERROR) {
			if(fd) mnstr_destroy(fd);
			msg = createException(MAL,"mal.import", "#WARNING: could not open file: %s\n", files[i]);
			GDKfree(base);
			return msg;
		} 

		if( included){
#ifdef _DEBUG_IMPORT_
			fprintf(stderr,"include file %s\n",files[i]);
#endif
			if( MCpushClientInput(cntxt, bstream_create(fd, 32 * BLOCK), listing, "")){
				GDKfree(base);
				throw(MAL,"mal.evalFile","Could not push the input stream");
			}
		} else {
#ifdef _DEBUG_IMPORT_
			fprintf(stderr,"load file %s using new client\n",files[i]);
#endif
			c = MCinitClient((oid)0,bstream_create(fd, 32 * BLOCK), cntxt->fdout);
			c->curmodule = c->usermodule = userModule();
			GDKfree(c->prompt);
			c->prompt= NULL;
			c->promptlength = 0;
			c->blkmode = 1; // collect all statements
			msg = defaultScenario(c);
			if( msg == MAL_SUCCEED){ 
				(void) MSinitClientPrg(c, "user", "main");  /* create new context */
				c->listing = listing;
				msg = runScenario(c);
			}
			c->fdout = NULL;	// to avoid accidental closing a default output channel
			MCcloseClient(c);
		}
	}
	GDKfree(base);
	return msg;
}
/*
 * The include operation simply pushes the file onto the preferred input queue.
 */
str
malInclude(Client c, str name, int listing)
{
	return evalFile(c, name,listing, 1);
}

/* patch a newline character if needed */
static str mal_cmdline(char *s, int *len)
{
	if (s[*len - 1] != '\n') {
		char *n = GDKmalloc(*len + 1 + 1);
		if (n == NULL)
			return s;
		strncpy(n, s, *len);
		n[*len] = '\n';
		n[*len + 1] = 0;
		(*len)++;
		return n;
	}
	return s;
}

/*
 * Compile the string, but don't execute it
 */
str
compileString(Client cntxt, str s)
{	int len = (int) strlen(s);
	buffer *b;
	str msg = MAL_SUCCEED;
	str qry, old=s ;
	Client c;

	if(0) s= mal_cmdline(s,&len);
	mal_unquote(qry = GDKstrdup(s));
	if( old != s)
		GDKfree(s);

    b = (buffer *) GDKzalloc(sizeof(buffer));
    if (b == NULL) {
        GDKfree(qry);
        return MAL_MALLOC_FAIL;
    }
    buffer_init(b, qry, len+2);

	c = MCinitClient((oid)0,0,0);
	c->fdin = bstream_create(buffer_rastream(b, "compileString"), b->len);
	strncpy(c->fdin->buf,s,len);
	c->curmodule = c->usermodule = userModule();
	GDKfree(c->prompt);
	c->prompt= NULL;
	c->promptlength = 0;

	msg = defaultScenario(c);
	if( msg == MAL_SUCCEED){ 
		(void) MSinitClientPrg(c, "user", "main");  /* create new context */
		c->blkmode = 1; // collect all statements
		while(msg == MAL_SUCCEED && c->fdin->eof == 0){
			msg = MALreader(c);
			if( msg == MAL_SUCCEED)
				msg = MALparser(c);
		}
		pushEndInstruction(c->curprg->def);
		chkProgram(c->usermodule, c->curprg->def);
	}
	c->fdout = 0;
	cntxt->curprg = c->curprg;
	c->curprg = 0;
	MCcloseClient(c);
	return MAL_SUCCEED;
}

str
callString(Client cntxt, str s)
{
	Client c;
	str msg = MAL_SUCCEED;

	c = MCinitClient((oid)0,cntxt->fdin,cntxt->fdout);
	msg = compileString(c,s);
	if( msg == MAL_SUCCEED)
		runMAL(c, c->curprg->def,0,0);
	c->fdin= 0;
	c->fdout = 0;
	MCcloseClient(c);
	return msg;
}
