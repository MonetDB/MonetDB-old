/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

/* (author) M.L. Kersten
 */
#include "monetdb_config.h"
#include "mal_session.h"
#include "mal_instruction.h" /* for pushEndInstruction() */
#include "mal_interpreter.h" /* for runMAL(), garbageElement() */
#include "mal_parser.h"	     /* for parseMAL() */
#include "mal_namespace.h"
#include "mal_builder.h"
#include "mal_authorize.h"
#include "mal_sabaoth.h"
#include "mal_private.h"
#include <gdk.h>	/* for opendir and friends */

/*#define _DEBUG_SESSION_*/
#ifdef HAVE_EMBEDDED
// FIXME:
//#include "mal_init_inline.h"
#endif

#define NL(X) ((X)=='\n' || (X)=='\r')
/*
 * The MonetDB server uses a startup script to boot the system.
 * This script is an ordinary MAL program, but will mostly
 * consist of include statements to load modules of general interest.
 * The startup script is run as user Admin.
 */
str
malBootstrap(void)
{
	Client c;
	str msg = MAL_SUCCEED;
	str bootfile = "mal_init", s = NULL;

	c = MCinitClient((oid) 0, 0, 0);
	assert(c != NULL);
	c->curmodule = c->usermodule = userModule();
	if ( (msg = defaultScenario(c)) ) {
		GDKfree(msg);
		fprintf(stderr,"#malBootstrap:Failed to initialise default scenario");
		mal_exit();
	}
	MSinitClientPrg(c, "user", "main");
	if( MCinitClientThread(c) < 0){
		fprintf(stderr,"#malBootstrap:Failed to create client thread");
		mal_exit();
	}
	//s = malInclude(c, bootfile, 0);
	// Completely execute the bootfile before continuing.
	s = evalFile(c, bootfile, 0,0);
	if (s != NULL) {
		fprintf(stderr, "!%s\n", s);
		GDKfree(s);
		mal_exit();
	}
	pushEndInstruction(c->curprg->def);
	chkProgram(c->usermodule, c->curprg->def);
	if ( (msg= c->curprg->def->errors) != MAL_SUCCEED ) {
		mnstr_printf(c->fdout,"!%s%s",msg, (msg[strlen(msg)-1] == '\n'? "":"\n"));
		mnstr_flush(c->fdout);
		if( GDKerrbuf && GDKerrbuf[0]){
			mnstr_printf(c->fdout,"!GDKerror: %s\n",GDKerrbuf);
			mnstr_flush(c->fdout);
		}
#ifdef HAVE_EMBEDDED
		return ;
#endif
	}
	s = MALengine(c);
	if (s != MAL_SUCCEED) {
		GDKfree(s);
#ifdef HAVE_EMBEDDED
		return;
#endif
	}
	return msg;
}

/*
 * Every client has a 'main' function to collect the statements.  Once
 * the END instruction has been found, it is added to the symbol table
 * and a fresh container is being constructed.  Note, this scheme makes
 * testing for recursive function calls a little more difficult.
 * Therefore, type checking should be performed afterwards.
 *
 * In interactive mode,  the closing statement is never reached.  The
 * 'main' procedure is typically cleaned between successive external
 * messages except for its variables, which are considerd global.  This
 * storage container is re-used when during the previous call nothing
 * was added.  At the end of the session we have to garbage collect the
 * BATs introduced.
 */
static str
MSresetClientPrg(Client cntxt, str mod, str nme)
{
	MalBlkPtr mb;
	InstrPtr p;

	cntxt->itrace = 0;  /* turn off any debugging */
	mb = cntxt->curprg->def;
	mb->stop = 1;
	mb->errors = MAL_SUCCEED;
	p = mb->stmt[0];
	assert(p);

	p->gc = 0;
	p->retc = 1;
	p->argc = 1;
	p->argv[0] = 0;

#ifdef _DEBUG_SESSION_
	fprintf(stderr,"reset sym %s %s to %s, id %d\n", 
		cntxt->curprg->name, getFunctionId(p), nme, findVariable(mb,nme) );
	fprintf(stderr,"vtop %d\n", mb->vtop);
	if( mb->vtop)
	fprintf(stderr,"first var %s\n", mb->var[0].id);
#endif

	setModuleId(p, mod);
	setFunctionId(p, nme);
	if( findVariable(mb,nme) < 0)
		p->argv[0] = newVariable(mb, nme, strlen(nme), TYPE_void);

	setVarType(mb, findVariable(mb, nme), TYPE_void);
	/* remove any MAL history */
	if (mb->history) {
		freeMalBlk(mb->history);
		mb->history = 0;
	}
	return MAL_SUCCEED;
}

/*
 * Create a new container block
 */

str
MSinitClientPrg(Client cntxt, str mod, str nme)
{
	if (cntxt->curprg  && idcmp(nme, cntxt->curprg->name) == 0)
		return MSresetClientPrg(cntxt, putName(mod), putName(nme));
	cntxt->curprg = newFunction(putName(mod), putName(nme), FUNCTIONsymbol);
	if( strcmp(mod,"user")==0 && strcmp(nme,"main")==0)
		setVarType(cntxt->curprg->def, findVariable(cntxt->curprg->def,"main"), TYPE_void);
	if( cntxt->curprg == 0)
		throw(MAL, "initClientPrg", MAL_MALLOC_FAIL);
	
	if (cntxt->glb == NULL )
		cntxt->glb = newGlobalStack(MAXGLOBALS + cntxt->curprg->def->vsize);
	if( cntxt->glb == NULL)
		throw(MAL,"initClientPrg", MAL_MALLOC_FAIL);
	assert(cntxt->curprg->def != NULL);
	assert(cntxt->curprg->def->vtop >0);
	return MAL_SUCCEED;
}

/*
 * The default method to interact with the database server is to connect
 * using a port number. The first line received should contain
 * authorization information, such as user name.
 *
 * The scheduleClient receives a challenge response consisting of
 * endian:user:password:lang:database:
 */
static void
exit_streams( bstream *fin, stream *fout )
{
	if (fout && fout != GDKstdout) {
		mnstr_flush(fout);
		close_stream(fout);
	}
	if (fin)
		bstream_destroy(fin);
}

const char* mal_enableflag = "mal_for_all";

void
MSscheduleClient(str command, str challenge, bstream *fin, stream *fout)
{
	char *user = command, *algo = NULL, *passwd = NULL, *lang = NULL;
	char *database = NULL, *s, *dbname;
	Client c;

	/* decode BIG/LIT:user:{cypher}passwordchal:lang:database: line */

	/* byte order */
	s = strchr(user, ':');
	if (s) {
		*s = 0;
		mnstr_set_byteorder(fin->s, strcmp(user, "BIG") == 0);
		user = s + 1;
	} else {
		mnstr_printf(fout, "!incomplete challenge '%s'\n", user);
		exit_streams(fin, fout);
		GDKfree(command);
		return;
	}

	/* passwd */
	s = strchr(user, ':');
	if (s) {
		*s = 0;
		passwd = s + 1;
		/* decode algorithm, i.e. {plain}mypasswordchallenge */
		if (*passwd != '{') {
			mnstr_printf(fout, "!invalid password entry\n");
			exit_streams(fin, fout);
			GDKfree(command);
			return;
		}
		algo = passwd + 1;
		s = strchr(algo, '}');
		if (!s) {
			mnstr_printf(fout, "!invalid password entry\n");
			exit_streams(fin, fout);
			GDKfree(command);
			return;
		}
		*s = 0;
		passwd = s + 1;
	} else {
		mnstr_printf(fout, "!incomplete challenge '%s'\n", user);
		exit_streams(fin, fout);
		GDKfree(command);
		return;
	}

	/* lang */
	s = strchr(passwd, ':');
	if (s) {
		*s = 0;
		lang = s + 1;
	} else {
		mnstr_printf(fout, "!incomplete challenge, missing language\n");
		exit_streams(fin, fout);
		GDKfree(command);
		return;
	}

	/* database */
	s = strchr(lang, ':');
	if (s) {
		*s = 0;
		database = s + 1;
		/* we can have stuff following, make it void */
		s = strchr(database, ':');
		if (s)
			*s = 0;
	}

	dbname = GDKgetenv("gdk_dbname");
	if (database != NULL && database[0] != '\0' &&
		strcmp(database, dbname) != 0)
	{
		mnstr_printf(fout, "!request for database '%s', "
						   "but this is database '%s', "
						   "did you mean to connect to monetdbd instead?\n",
				database, dbname);
		/* flush the error to the client, and abort further execution */
		exit_streams(fin, fout);
		GDKfree(command);
		return;
	} else {
		str err;
		oid uid;
		sabdb *stats = NULL;
		Client root = &mal_clients[0];

		/* access control: verify the credentials supplied by the user,
		 * no need to check for database stuff, because that is done per
		 * database itself (one gets a redirect) */
		err = AUTHcheckCredentials(&uid, root, user, passwd, challenge, algo);
		if (err != MAL_SUCCEED) {
			mnstr_printf(fout, "!%s\n", err);
			exit_streams(fin, fout);
			GDKfree(err);
			GDKfree(command);
			return;
		}

		err = SABAOTHgetMyStatus(&stats);
		if (err != MAL_SUCCEED) {
			/* this is kind of awful, but we need to get rid of this
			 * message */
			fprintf(stderr, "!SABAOTHgetMyStatus: %s\n", err);
			freeException(err);
			mnstr_printf(fout, "!internal server error, "
						 "please try again later\n");
			exit_streams(fin, fout);
			GDKfree(command);
			return;
		}
		if (stats->locked == 1) {
			if (uid == 0) {
				mnstr_printf(fout, "#server is running in "
							 "maintenance mode\n");
			} else {
				mnstr_printf(fout, "!server is running in "
							 "maintenance mode, please try again later\n");
				exit_streams(fin, fout);
				SABAOTHfreeStatus(&stats);
				GDKfree(command);
				return;
			}
		}
		SABAOTHfreeStatus(&stats);

		c = MCinitClient(uid, fin, fout);
		if (c == NULL) {
			if ( MCshutdowninprogress())
				mnstr_printf(fout, "!system shutdown in progress, please try again later\n");
			else
				mnstr_printf(fout, "!maximum concurrent client limit reached "
								   "(%d), please try again later\n", MAL_MAXCLIENTS);
			exit_streams(fin, fout);
			GDKfree(command);
			return;
		}
		/* move this back !! */
		if (c->usermodule == 0) {
			c->curmodule = c->usermodule = userModule();
		}

		if ((s = setScenario(c, lang)) != NULL) {
			mnstr_printf(c->fdout, "!%s\n", s);
			mnstr_flush(c->fdout);
			GDKfree(s);
			c->mode = FINISHCLIENT;
		}
		if (!GDKgetenv_isyes(mal_enableflag) &&
				(strncasecmp("sql", lang, 3) != 0 && uid != 0)) {

			mnstr_printf(fout, "!only the 'monetdb' user can use non-sql languages. "
					           "run mserver5 with --set %s=yes to change this.\n", mal_enableflag);
			exit_streams(fin, fout);
			GDKfree(command);
			return;
		}
	}

	(void) MSinitClientPrg(c, "user", "main");

	GDKfree(command);

	/* NOTE ABOUT STARTING NEW THREADS
	 * At this point we have conducted experiments (Jun 2012) with
	 * reusing threads.  The implementation used was a lockless array of
	 * semaphores to wake up threads to do work.  Experimentation on
	 * Linux, Solaris and Darwin showed no significant improvements, in
	 * most cases no improvements at all.  Hence the following
	 * conclusion: thread reuse doesn't save up on the costs of just
	 * forking new threads.  Since the latter means no difficulties of
	 * properly maintaining a pool of threads and picking the workers
	 * out of them, it is favourable just to start new threads on
	 * demand. */

	/* fork a new thread to handle this client */
	mnstr_settimeout(c->fdin->s, 50, GDKexiting);
	MSserveClient(c);
}

/*
 * After the client initialization has been finished, we can start the
 * interaction protocol. This involves parsing the input in the context
 * of an already defined procedure and upon success, its execution.
 *
 * In essence, this calls for an incremental parsing operation, because
 * we should wait until a complete basic block has been detected.  Test,
 * first collect the instructions before we take them all.
 *
 * In interactive mode, we should remove the instructions before
 * accepting new ones. The function signature remains the same and the
 * symbol table should also not be affected.  Aside from removing
 * instruction, we should also condense the variable stack, i.e.
 * removing at least the temporary variables, but maybe everything
 * beyond a previous defined point.
 *
 * Beware that we have to cleanup the global stack as well. This to
 * avoid subsequent calls to find garbage information.  However, this
 * action is only required after a successful execution.  Otherwise,
 * garbage collection is not needed.
 */
void
MSresetInstructions(MalBlkPtr mb, int start)
{
	int i;
	InstrPtr p;

	for (i = start; i < mb->ssize; i++) {
		p = getInstrPtr(mb, i);
		if (p)
			freeInstruction(p);
		mb->stmt[i] = NULL;
	}
	mb->stop = start;
}

/*
 * Determine the variables being used and clear non-used onces.
 */
void
MSresetVariables(Client cntxt, MalBlkPtr mb, MalStkPtr glb, int start)
{
	int i;

#ifdef _DEBUG_SESSION_
	fprintf(stderr,"resetVarables %d  vtop %d errors %s\n", start, mb->vtop,mb->errors);
#endif
	if( start <= mb->vtop)
		for (i = 0; i < start ; i++)
			setVarUsed(mb,i);
	if (mb->errors == MAL_SUCCEED)
		for (i = start; i < mb->vtop; i++) {
			if (isVarUsed(mb,i) || !isTmpVar(mb,i)){
				assert(!mb->var[i].value.vtype || isVarConstant(mb, i));
				setVarUsed(mb,i);
			}
			if (glb && !isVarUsed(mb,i)) {
				if (isVarConstant(mb, i))
					garbageElement(cntxt, &glb->stk[i]);
				/* clean stack entry */
				glb->stk[i].vtype = TYPE_int;
				glb->stk[i].len = 0;
				glb->stk[i].val.pval = 0;
			}
		}

#ifdef _DEBUG_SESSION_
	fprintf(stderr,"resetVar %s %d\n", getFunctionId(mb->stmt[0]), mb->var[mb->stmt[0]->argv[0]].used);
#endif
	if (mb->errors == MAL_SUCCEED)
		trimMalVariables_(mb, glb);
#ifdef _DEBUG_SESSION_
	fprintf(stderr,"after trim %s %d\n", getFunctionId(mb->stmt[0]), mb->vtop);
#endif
}

/*
 * This is a phtread started function.  Here we start the client. We
 * need to initialize and allocate space for the global variables.
 * Thereafter it is up to the scenario interpreter to process input.
 */
str
MSserveClient(void *dummy)
{
	MalBlkPtr mb;
	Client c = (Client) dummy;
	str msg = 0;

	if (!isAdministrator(c) && MCinitClientThread(c) < 0) {
		MCcloseClient(c);
		return MAL_SUCCEED;
	}
	/*
	 * A stack frame is initialized to keep track of global variables.
	 * The scenarios are run until we finally close the last one.
	 */
	mb = c->curprg->def;
	if (c->glb == NULL)
		c->glb = newGlobalStack(MAXGLOBALS + mb->vsize);
	if (c->glb == NULL) {
		c->mode = RUNCLIENT;
		throw(MAL, "serveClient", MAL_MALLOC_FAIL);
	} else {
		c->glb->stktop = mb->vtop;
		c->glb->blk = mb;
	}

	if (c->scenario == 0)
		msg = defaultScenario(c);

	if (msg) {
		c->mode = RUNCLIENT;
		return msg;
	} else {
		do {
			do {
				msg = runScenario(c);
				freeException(msg);
				if (c->mode == FINISHCLIENT)
					break;
				resetScenario(c);
			} while (c->scenario && !GDKexiting());
		} while (c->scenario && c->mode != FINISHCLIENT && !GDKexiting());
	}
	/* pre announce our exiting: cleaning up may take a while and we
	 * don't want to get killed during that time for fear of
	 * deadlocks */
	MT_exiting_thread();
	/*
	 * At this stage we should clean out the MAL block
	 */
    if (c->backup) {
        freeSymbol(c->backup);
        c->backup = 0;
    }
	if (c->curprg) {
		Symbol s = c->curprg;
		assert(0);
		c->curprg = 0;
		freeSymbol(s);
	}
	if (c->usermodule) {
		assert(0);
	}

	if (c->mode > FINISHCLIENT) {
		if (isAdministrator(c) /* && moreClients(0)==0 */) {
			if (c->scenario) {
				exitScenario(c);
			}
		}
	}
	if (!isAdministrator(c))
		MCcloseClient(c);
	if (c->usermodule && strcmp(c->usermodule->name, "user") == 0) {
		GDKfree(c->usermodule->space);
		GDKfree(c->usermodule);
		c->usermodule = NULL;
	}
	return MAL_SUCCEED;
}

/*
 * The stages of processing user requests are controlled by a scenario.
 * The routines below are the default implementation.  The main issues
 * to deal after parsing it to clean out the Admin.main function from
 * any information added erroneously.
 *
 * Ideally this involves resetting the state of the client 'main'
 * function, i.e. the symbol table is reset and any instruction added
 * should be cleaned. Beware that the instruction table may have grown
 * in size.
 */
str
MALinitClient(Client c)
{
	assert(c->state[0] == NULL);
	c->state[0] = c;
	return NULL;
}

str
MALexitClient(Client c)
{
	Module m = c->usermodule;
	if (c->glb && c->curprg->def->errors == MAL_SUCCEED)
		garbageCollector(c, c->curprg->def, c->glb, TRUE);
	c->mode = FINISHCLIENT;
    if (c->backup) {
        freeSymbol(c->backup);
        c->backup = 0;
    }
	c->curprg = NULL;
	c->usermodule = NULL;
	// only clear out the private module
	// Beware the parser may choosen another target
	if (m && strcmp(m->name,"user")== 0){
		freeModule(m);
	}
	return NULL;
}

/* 
 * All MAL instructions logically are organized per line(s)
 * and terminated with a ';' 
 * Comments run until the end of the line and are discarded.
 * The MALreader collects such a single extended line.
 */
str
MALreader(Client c)
{	str s,l;
	int string = 0;
	int blocked;

	// First eat away any left over input
	if( c->linefill)
		return MAL_SUCCEED;
	if( c->fdin == 0)
		throw(MAL,"mal.reader","missing input");
	do{
		blocked = isa_block_stream(c->fdin->s);
		if(c->fdin->pos >= c->fdin->len ){
			ssize_t nr = 0;
			if(c->fdin->eof && c->prompt ){
				if (!blocked)
					mnstr_write(c->fdout, c->prompt, strlen(c->prompt), 1);
				mnstr_flush(c->fdout);
				c->fdin->eof = 0;
			} else
			if( ! blocked && c->prompt){
					mnstr_write(c->fdout, c->prompt, strlen(c->prompt), 1);
				mnstr_flush(c->fdout);
			}
			nr = bstream_next(c->fdin);
			if(nr < 0 || (!blocked && c->fdin->eof)){
		alternative:
				if (c->bak){
#ifdef _DEBUG_SESSION_
					fprintf(stderr,"Pop the input stream for client %d\n", c->idx);
#endif
					MCpopClientInput(c);
				} else{
					// if we have unprocessed data we should return and await its consumption
					if(c->line && *c->line){
						return MAL_SUCCEED;
					}
					MT_lock_set(&mal_contextLock);
					c->mode = FINISHCLIENT;
					MT_lock_unset(&mal_contextLock);
				}
				return MAL_SUCCEED;
			}
			if (!nr && blocked ){
				nr = bstream_next(c->fdin); // check for eof 
				if (c->fdin->eof)
					goto alternative;
			} 
			if( !nr)
				continue;

			// Handle very long lines
			if ( c->fdin->len >= c->linesize){
				l = GDKrealloc(c->line, c->linesize + c->fdin->len + 512);
				if( l == NULL)
					throw(MAL,"mal.readline", MAL_MALLOC_FAIL);
				c->line = l;
				c->linesize += c->fdin->len + 512;
			}
		}
		// read until you find a complete MAL unit
		s = c->line + c->linefill;
		l = c->fdin->buf + c->fdin->pos; 
		
		// special case when we are in debugging mode.
		if( c->prompt && strncmp(c->prompt,"mdb>",4)== 0){
			for( ; c->fdin->pos < c->fdin->len; l++){
				c->fdin->pos++;
				if( *l == '\n') {
					l--;
					break;
				}
				*s++ = *l;
				c->linefill++;
			}
			*s = 0;
			return MAL_SUCCEED;
		}

		for( ; c->fdin->pos < c->fdin->len ; l++){
			// skip string literals
			if ( *l == '"' ){
				if ( string == 0)
					string = 1;
				else
				if ( string && *(l-1) != '\\')
					string = !string;
			}
			*s++ = *l;
			c->linefill++;
			c->fdin->pos++;
			if ( c->listing)
				mnstr_printf(c->fdout,"%c", *l);
			if( string)
				continue;
			if ( *l == ';' ){
				*s = 0;
				return MAL_SUCCEED;
			}
			
			if ( *l == '#' ){
				c->linefill--;
				s--;
				*s = 0;
				// eat everything away until end of line
				for( l++ ; *l && c->fdin->pos < c->fdin->len ; l++){
					if ( c->listing)
						mnstr_printf(c->fdout,"%c", *l);
					c->fdin->pos++;
					if (*l && ( *l == '\n' ||  *l == '\r' ))
						break;
				}
				return MAL_SUCCEED;
			}
		}
		*s = 0;
	} while (c->fdin->eof == 0 || blocked);
	
	return MAL_SUCCEED;
}
/*
 * The parser should parse a complete MAL unit.
 *
 * The client record always has a MALblock with the user.main signature.
 * This block is extended with parsed instructions.
 * This is either a MAL function block, guarded block, or single statement
 * Syntax checking is performed on line by line basis and sent to the output channel.
 *
 * Type checking is the last step in this process and involves handling new function admin.
 */
str
MALparser(Client cntxt)
{
	int vtop;
	str msg = MAL_SUCCEED;
	int finalize;

	cntxt->curprg->def->errors = MAL_SUCCEED;
	vtop = cntxt->curprg->def->vtop;

	if( cntxt->linefill == 0)
		return msg;
	// parse a single MAL instruction, add it to the container cntxt->curprg->def
	finalize= parseMAL(cntxt);

	assert(cntxt->line);
	cntxt->linefill = 0;
	cntxt->lineptr = cntxt->line;
	*cntxt->line = 0;
	// Any error encountered should reset the function under construction
	msg = cntxt->curprg->def->errors;
	if( msg != MAL_SUCCEED){
		cntxt->curprg->def->errors = MAL_SUCCEED;
		MSresetVariables(cntxt, cntxt->curprg->def, cntxt->glb, vtop);
		resetMalBlk(cntxt->curprg->def, 1);
		return msg;
	}

	// Handle compound MAL blocks before execution
	// There are three cases: a MAL function block, a barrier block or single statement
	// nested blocks are recognized by the blkmode depth
	/* empty blocks should be skipped as well */
	if (cntxt->blkmode)
		return MAL_SUCCEED;

	if (cntxt->curprg->def->stop == 1)
		return MAL_SUCCEED;

	/* wrap up the parsing of a single line MAL block */
	if( finalize == 0){
		pushEndInstruction(cntxt->curprg->def);
		// A compound block is ready for execution once it has been type checked.
		chkProgram(cntxt->usermodule, cntxt->curprg->def);
		if (cntxt->curprg->def->errors) {
			msg = cntxt->curprg->def->errors;
			cntxt->curprg->def->errors = MAL_SUCCEED;
			MSresetVariables(cntxt, cntxt->curprg->def, cntxt->glb, vtop);
			resetMalBlk(cntxt->curprg->def, 1);
		}
	} else {
		// check a compound MAL function and store it in the symbol table
		// insert the symbol first, otherwise we can not check recursive calls.
		// Erroneous functions are not destroyed, because the user may want to inspect them
		// They will never be executed though
#ifdef _DEBUG_SESSION_
		fprintf(stderr,"insert symbol %s.%s in %s\n", 	
			getModuleId(getSignature(cntxt->curprg)),
			getFunctionId(getSignature(cntxt->curprg)),
			cntxt->usermodule->name);
#endif
/*
		insertSymbol(cntxt->usermodule, cntxt->curprg);
		chkProgram(cntxt->usermodule, cntxt->curprg->def);
		msg = cntxt->curprg->def->errors;
		cntxt->curprg->def->errors = 0;
		if (msg) {
		} 
		(void) MSinitClientPrg(cntxt,"user","main");
*/
	}

	return msg;
}

int
MALcommentsOnly(MalBlkPtr mb)
{
	int i;

	for (i = 1; i < mb->stop; i++)
		if (mb->stmt[i]->token != REMsymbol)
			return 0;
	return 1;
}

str
MALengine(Client c)
{
	Symbol prg;
	str msg = MAL_SUCCEED;
	int vtop= c->curprg->def->vtop;

	if (c->blkmode)
		return MAL_SUCCEED;
	prg = c->curprg;
	if (prg == NULL)
		throw(SYNTAX, "mal.engine", SYNTAX_SIGNATURE);
	if (prg->def == NULL)
		throw(SYNTAX, "mal.engine", SYNTAX_SIGNATURE);

	if (prg->def->errors ) {
		MSresetVariables(c, c->curprg->def, c->glb, vtop);
		resetMalBlk(c->curprg->def, 1);
		throw(MAL, "mal.engine", "%s", prg->def->errors);
	}
	if (prg->def->stop == 1 || MALcommentsOnly(prg->def))
		return 0;   /* empty block */
	if (c->glb) {
		if (prg->def && c->glb->stksize < prg->def->vsize){
			c->glb = reallocGlobalStack(c->glb, prg->def->vsize);
			if( c->glb == NULL)
				throw(MAL, "mal.engine", MAL_MALLOC_FAIL);
		}
		c->glb->stktop = prg->def->vtop;
		c->glb->blk = prg->def;
		c->glb->cmd = (c->itrace && c->itrace != 'C') ? 'n' : 0;
	}

	/*
	 * In interactive mode we should avoid early garbage collection of values.
	 * This can be controlled by the clean up control at the instruction level
	 * and marking all non-temporary variables as being (potentially) used.
	 */
	if (c->glb) {
		c->glb->pcup = 0;
		c->glb->keepAlive = TRUE; /* no garbage collection */
	}
	if (prg->def->errors == MAL_SUCCEED)
		msg = (str) runMAL(c, prg->def, 0, c->glb);
	if (msg) {
		/* ignore "internal" exceptions */
		if (strstr(msg, "client.quit") )
			msg = MAL_SUCCEED;
	}
	MSresetVariables(c, prg->def, c->glb, 1);
	resetMalBlk(prg->def, 1);
	if (c->glb) {
		/* for global stacks avoid reinitialization from this point */
		c->glb->stkbot = prg->def->vtop;
	}
	
	if( prg->def->errors)
		GDKfree(prg->def->errors);
	prg->def->errors = NULL;
	if (c->itrace)
		mnstr_printf(c->fdout, "mdb>#EOD\n");
	return msg;
}

