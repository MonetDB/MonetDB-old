/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2012 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * @f receptor
 * @a Martin Kersten
 * @v 1
 * @+ DataCell Receptor
 * This module is a prototype for the implementation of a DataCell receptor.  It can be used as follows.
 * @example
 * @end example
 * After this call sensors can send tuples to the
 * the stream X at the DataCell connection.
 * They are appended to the event basket with the usec clock tick included.
 *
 * Each receptor is supported by an independent thread
 * that reads the input and stores the data in a container
 * composed of a series of event baskets.
 *
 * The critical issue is that the receptor should hand over
 * the events to the main thread in a safe/secure way.
 * The overhead should be kept to an absolute minimum.
 * Currently it is regulated using a simple locking
 * scheme for the baskets.
 *
 * The event format is currently strict and relies on the tablet
 * module to parse them.
 */

#include "monetdb_config.h"
#include "receptor.h"
#include "dcsocket.h"
#include "stream_socket.h"
#include "mal_builder.h"

/* #define _DEBUG_RECEPTOR_ */

/* default settings */
#define RCHOST "localhost"
#define RCPORT 55000

typedef struct RECEPTOR {
	str name;
	str host;
	int port;
	int mode;   /* active/passive */
	int protocol;   /* event protocol UDP,TCP,CSV */
	int bskt;   /* connected to a basket */
	int status;
	int delay;  /* control the delay between attempts to connect */
	int lck;
	str scenario;   /* use a scenario file */
	int sequence;   /* repetition count */
	str modnme, fcnnme; /* generic receptor generators */
	stream * receptor;
	SOCKET sockfd;
	SOCKET newsockfd;
	str error;  /* what went wrong */
	MT_Id pid;
	lng received;
	Tablet table;   /* tuple input structure */
	struct RECEPTOR *nxt, *prv;
} RCrecord, *Receptor;

static Receptor rcAnchor = NULL;
static str rcError = NULL;
static int rcErrorEvent = 0;

static str RCstartThread(Receptor rc);
static void RCscenario(Receptor rc);
static void RCgenerator(Receptor rc);

static Receptor
RCnew(str nme)
{
	Receptor rc;

	rc = (Receptor) GDKzalloc(sizeof(RCrecord));
	if (rc == 0)
		return rc;
	rc->name = GDKstrdup(nme);
	if (rcAnchor)
		rcAnchor->prv = rc;
	rc->nxt = rcAnchor;
	rc->prv = NULL;
	rcAnchor = rc;
	return rc;
}

static Receptor
RCfind(str nme)
{
	Receptor r;
	char buf[BUFSIZ];
	for (r = rcAnchor; r; r = r->nxt)
		if (strcmp(nme, r->name) == 0)
			return r;
	snprintf(buf,BUFSIZ,"datacell.%s",nme);
	for (r = rcAnchor; r; r = r->nxt)
		if (strcmp(buf, r->name) == 0)
			return r;
	return NULL;
}
/*
 * @-
 * The MAL interface for managing the receptor pool
 * The basket should already be defined. Their order
 * is used to interpret the messages received.
 * The standard tuple layout for MonetDB interaction is used.
 */
str
DCreceptorNew(int *ret, str *tbl, str *host, int *port)
{
	Receptor rc;
	int idx, i, j, len;
	Column *fmt;
	BAT *b;

	if (RCfind(*tbl))
		throw(MAL, "receptor.new", "Duplicate receptor");
	for (i = 1; i < bsktTop; i++)
		if (baskets[i].port == *port)
			throw(MAL, "receptor.new", "Port already in use");

	rc = RCnew(*tbl);
	if (rc == 0)
		throw(MAL, "receptor.new", MAL_MALLOC_FAIL);
	rc->host = GDKstrdup(*host);
	rc->port = *port;
	rc->error = NULL;
	rc->delay = PAUSEDEFAULT;
	rc->lck = 0;
	rc->status = BSKTSTOP;
	rc->scenario = 0;
	rc->sequence = 0;
	rc->modnme = 0;
	rc->fcnnme = 0;
	rc->mode = BSKTPASSIVE;
	rc->protocol = TCP;

	rc->bskt = idx = BSKTlocate(*tbl);
	if (idx == 0) /* should not happen */
		throw(MAL, "receptor.new", "Basket not found");
	len = BSKTmemberCount(*tbl);
	fmt = rc->table.format = GDKzalloc(sizeof(Column) * len);

	baskets[idx].kind = "receptor";
	baskets[idx].status = BSKTSTOP;
	baskets[idx].host = GDKstrdup(*host);
	baskets[idx].port = *port;
	for (j = 0, i = 0; i < baskets[idx].colcount; i++) {
		b = baskets[idx].primary[j];
		if (b == NULL) {
			rc->table.nr_attrs = j;   /* ensure a consistent structure*/
			throw(MAL, "receptor.new", "Could not access descriptor");
		}
		BBPincref(b->batCacheid, TRUE);
		fmt[j].c[0] = b;
		fmt[j].name = GDKstrdup(baskets[idx].cols[i]);
		fmt[j].sep = GDKstrdup(",");
		fmt[j].seplen = 1;
		fmt[j].type = GDKstrdup(ATOMname(b->ttype));
		fmt[j].adt = (b)->ttype;
		fmt[j].tostr = &TABLETadt_toStr;
		fmt[j].frstr = &TABLETadt_frStr;
		fmt[j].extra = fmt + j;
		fmt[j].len = fmt[j].nillen =
						 ATOMlen(fmt[j].adt, ATOMnilptr(fmt[j].adt));
		fmt[j].data = GDKmalloc(fmt[j].len);
		fmt[j].nullstr = GDKmalloc(fmt[j].len + 1);
		j++;
	}
	rc->table.nr_attrs = j;

#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Instantiate a new receptor %d fields\n", j);
#endif
	(void) ret;
	return MAL_SUCCEED;
}

str DCreceptorPause(int *ret, str *nme)
{
	Receptor rc;

	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.resume", "Receptor not defined");
	if (rc->status != BSKTLISTEN)
		throw(MAL, "receptor.resume", "Receptor not started");
	rc->status = BSKTPAUSE;
	baskets[rc->bskt].status=BSKTPAUSE;

#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Pause a receptor\n");
#endif
	(void) ret;
	return MAL_SUCCEED;
}

str DCreceptorResume(int *ret, str *nme)
{
	Receptor rc;

	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.resume", "Receptor not defined");
	if (rc->status == BSKTSTOP) {
		if (MT_create_thread(&rc->pid, (void (*)(void *))RCstartThread, rc, MT_THR_DETACHED) != 0) {
			throw(MAL, "receptor.start", "Receptor initiation failed");
		}
	} else if (rc->status != BSKTPAUSE)
		throw(MAL, "receptor.resume", "Receptor not paused");

	rc->status = BSKTLISTEN;
	baskets[rc->bskt].status=BSKTLISTEN;

#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Resume a receptor\n");
#endif
	(void) ret;
	return MAL_SUCCEED;
}

str
RCpause(int *ret)
{
	Receptor rc;
	for (rc = rcAnchor; rc; rc = rc->nxt)
		if (rc->status != BSKTSTOP)
			DCreceptorPause(ret, &rc->name);
	return MAL_SUCCEED;
}

str
RCresume(int *ret)
{
	Receptor rc;
	for (rc = rcAnchor; rc; rc = rc->nxt)
		if (rc->status == BSKTSTOP)
			DCreceptorResume(ret, &rc->name);
	return MAL_SUCCEED;
}

str RCdrop(int *ret, str *nme)
{
	Receptor rc, rb;

	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.drop", "Receptor not defined");
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Drop a receptor\n");
#endif
	(void) ret;
	if (rcAnchor == rc)
		rcAnchor = rc->nxt;
	rb = rc->prv;
	if (rc->nxt)
		rc->nxt->prv = rc->prv;
	if (rb)
		rb->nxt = rc->nxt;
	rc->status = BSKTDROP;
	if (rc->lck)
		BSKTunlock(&rc->lck, &rc->name);
	MT_join_thread(rc->pid);
	return MAL_SUCCEED;
}

str RCreset(int *ret)
{
	Receptor r, o;
	for (r = rcAnchor; r; r = o) {
		o = r->nxt;
		RCdrop(ret, &r->name);
	}
	return MAL_SUCCEED;
}

str DCscenario(int *ret, str *nme, str *fname, int *seq)
{
	Receptor rc;
	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.scenario", "Receptor not defined");
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Define receptor scenario\n");
#endif
	(void) ret;
	rc->scenario = GDKstrdup(*fname);
	rc->sequence = *seq;
	return MAL_SUCCEED;
}

str RCmode(int *ret, str *nme, str *arg)
{
	Receptor rc;
	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.mode", "Receptor not defined");
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Define receptor mode\n");
#endif
	(void) ret;
	if (strcmp(*arg, "passive") == 0)
		rc->mode = BSKTPASSIVE;
	else if (strcmp(*arg, "active") == 0)
		rc->mode = BSKTACTIVE;
	else
		throw(MAL, "receptor.mode", "Must be either passive/active");
	return MAL_SUCCEED;
}

str RCprotocol(int *ret, str *nme, str *mode)
{
	Receptor rc;
	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.protocol", "Receptor not defined");
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Define receptor protocol\n");
#endif
	(void) ret;
	if (strcmp(*mode, "udp") == 0)
		rc->protocol = UDP;
	else if (strcmp(*mode, "tcp") == 0)
		rc->protocol = TCP;
	else if (strcmp(*mode, "csv") == 0)
		rc->protocol = CSV;
	else
		throw(MAL, "receptor.protocol", "Must be either udp/tcp/csv");
	return MAL_SUCCEED;
}

str DCgenerator(int *ret, str *nme, str *modnme, str *fcnnme)
{
	Receptor rc;
	rc = RCfind(*nme);
	if (rc == NULL)
		throw(MAL, "receptor.generator", "Receptor not defined");
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Define receptor generator\n");
#endif
	(void) ret;
	rc->modnme = GDKstrdup(*modnme);
	rc->modnme = GDKstrdup(*fcnnme);
	return MAL_SUCCEED;
}

/*
 * @-
 * The hard part starts here. Each receptor is turned into
 * a separate thread that reads the channel and prepares
 * the containers for the continuous queries.
 * The receptor body should continously read until the socket is closed.
 */

static void
RCreconnect(Receptor rc)
{
	do {
		rc->error = NULL;
		if (rc->mode == BSKTACTIVE)
			rc->error = socket_client_connect(&rc->newsockfd, rc->host, rc->port);
		if (rc->error) {
			mnstr_printf(RCout, "#Receptor connect fails: %s\n", rc->error);
			MT_sleep_ms(rc->delay);
		} 
	} while (rc->error);
}


static void
RCbody(Receptor rc)
{
	char buf[MYBUFSIZ + 1];
	char tuplesINbuffer[5];
	int counter = 0;
	int cnt;
	size_t j;
	str e, he;
	str line = "\0";
	int i, k, n;
#ifdef _DEBUG_RECEPTOR_
	int m = 0;
#endif
	buf[MYBUFSIZ] = 0; /* ensure null terminated string */

	if (rc->scenario) {
		RCscenario(rc);
		return;
	}
	if (rc->modnme && rc->fcnnme) {
		RCgenerator(rc);
		return;
	}
	/* ADD YOUR FAVORITE RECEPTOR CODE HERE */

bodyRestart:
	/* create the channel the first time or when connection was lost. */
	if (rc->mode == BSKTACTIVE && rc->protocol == UDP)
		rc->receptor = udp_rastream(rc->host, rc->port, rc->name);
	else
		rc->receptor = socket_rastream(rc->newsockfd, rc->name);
	if (rc->receptor == NULL) {
		perror("Receptor: Could not open stream");
		mnstr_printf(RCout, "#stream %s.%d.%s\n", rc->host, rc->port, rc->name);
		socket_close(rc->newsockfd);
#ifdef _DEBUG_RECEPTOR_
		mnstr_printf(RCout, "#Terminate RCbody loop\n");
#endif
		return;
	}

	/*
	 * @-
	 * Consume each event and store the result.
	 * If the thread is suspended we sleep for at least one second.
	 * In case of a locked basket we sleep for a millisecond.
	 */

	for (n = 1; n > 0;) {
		while (rc->status == BSKTPAUSE && rc->delay) {
#ifdef _DEBUG_RECEPTOR_
			mnstr_printf(RCout, "#pause receptor\n");
#endif
			MT_sleep_ms(rc->delay);
		}

		if (rc->status == BSKTSTOP)
			break;
		if (rc->status == BSKTDROP) {
			mnstr_close(rc->receptor);
			for (j = 0; j < rc->table.nr_attrs; j++) {
				GDKfree(rc->table.format[j].sep);
				GDKfree(rc->table.format[j].name);
				GDKfree(rc->table.format[j].data);
				GDKfree(rc->table.format[j].nullstr);
				BBPdecref(rc->table.format[j].c[0]->batCacheid, TRUE);
			}
			shutdown(rc->newsockfd, SHUT_RDWR);
			GDKfree(rc);
			rc = NULL;
			break;
		}

#ifdef _DEBUG_RECEPTOR_
		mnstr_printf(RCout, "#wait for data read m: %d\n", m);
#endif

/* actually we should switch here based on the event syntax protocol */

/*Batch Processing
   The Datadriver (see linear road benchmark) or the Sensor tools, are connected through TCP/IP
   connection to the receptor module and	feed the DataCell with tuples,
   Both tools are able to send batches of tuples to the stream engine
   The first line of each batch always contains the number of tuples that the receptor is going to read (i.e.,#number)
   When the receptor reads the first line of the incoming message, it immediately LOCKS the bats (that constitute the basket)
   and keeps the lock until the end of the reading/writting procedure
   When the receptor reads all the tuples UNLOCKS the bats, and then the Factories/Queries that are waiting for these data are able to read it*/

		if ((n = (int) mnstr_readline(rc->receptor, buf, MYBUFSIZ)) > 0) {
			buf[n + 1] = 0;
#ifdef _DEBUG_RECEPTOR_
			mnstr_printf(RCout, "#Receptor buf [%d]:%s \n", n, buf);
			m = 0;
#endif
			/* use trivial concurrency measure */
			line = buf;

			BSKTlock(&rc->lck, &rc->name, &rc->delay);
			/* BATs may be replaced in the meantime */
			for (i = 0; i < baskets[rc->bskt].colcount; i++)
				rc->table.format[i].c[0] = baskets[rc->bskt].primary[i];

			cnt = 0;
			he = strchr(line, '#');
			if (he != 0) {
				strcpy(tuplesINbuffer, line + 1);
				counter = atoi(tuplesINbuffer);
				*he = 0;
			} else {
				/* we got the line already */
				goto parse;
			}

/* this code should be optimized for block-based reads */
			while (cnt < counter) {
				if ((n = (int) mnstr_readline(rc->receptor, buf, MYBUFSIZ)) > 0) {
					buf[n + 1] = 0;
#ifdef _DEBUG_RECEPTOR_
					mnstr_printf(RCout, "#Receptor buf [%d]:%s \n", n, buf);
#endif
parse:
					if (rc->status != BSKTLISTEN)
						break;
					do {
						line = buf;
						e = strchr(line, '\n');
						if (e == 0) {
							/* only keep the last errorenous event for analysis */
							if (rcError)
								GDKfree(rcError);
							rcError = (char *) GDKmalloc(k = strlen(line) + 100);
							if (rcError)
								snprintf(rcError, k, "newline missing:%s", line);
							rcErrorEvent = cnt;
							cnt--;
							break;
						}
						*e = 0;
#ifdef _DEBUG_RECEPTOR_
						mnstr_printf(RCout, "#insert line :%s \n", line);
#endif
						if (insert_line(&rc->table, line, NULL, 0, rc->table.nr_attrs) < 0) {
							if (baskets[rc->bskt].errors)
								BUNappend(baskets[rc->bskt].errors, line, TRUE);
							/* only keep the last errorenous event for analysis */
							if (rcError)
								GDKfree(rcError);
							rcError = (char *) GDKmalloc(k = strlen(line) + 100);
							if (rcError)
								snprintf(rcError, k, "parsing error:%s", line);
							rcErrorEvent = cnt;
							break;
						}
						rc->received++;
						e++;
						line = e;
					} while (*e);
				}
				cnt++;
			}
			BSKTunlock(&rc->lck, &rc->name);
			if (rc->table.error) {
				mnstr_printf(GDKerr, "%s", rc->table.error);
				rc->table.error = 0;
			}
		}
	}
	/* only when reading fails we attempt to reconnect */
	mnstr_close(rc->receptor);
	if (rc->mode == BSKTACTIVE) {
		/* try to reconnect */
		RCreconnect(rc);
		goto bodyRestart;
	}
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Terminate RCbody loop\n");
#endif
}
/*
 * @-
 * A short cut is to generate the events based upon the interpretation
 * of a scenario file. Much like the one used in the sensor.
 * It is processed multiple times. The header is the delay imposed.
 * Make sure you use a complete path.
 */
void
RCscenario(Receptor rc)
{
	char buf[MYBUFSIZ + 1], *tuple;
	lng tick;
	lng previoustsmp = 0;
	FILE *fd;
	int snr;
	int newdelay = 0;

	if (rc->scenario == 0) {
		mnstr_printf(RCout, "Scenario missing\n");
		return;
	}
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Execute the scenario '%s'\n", rc->scenario);
#endif

	snr = 0;
	do {
		fd = fopen(rc->scenario, "r");
		if (fd == NULL) {
			mnstr_printf(RCout, "Could not open file '%s'\n", rc->scenario);
			return;
		}

		/* read the event requests and sent when the becomes */
		while (fgets(buf, MYBUFSIZ, fd) != 0) {
			newdelay = (int) atol(buf);
			tuple = buf;

			if (newdelay > 0) {
				/* wait */
				tuple = strchr(buf, '[');
				if (tuple == 0)
					tuple = buf;
				MT_sleep_ms(newdelay);
			} else if (rc->delay > 0) {
				/* wait */
				MT_sleep_ms(rc->delay);
			}
#ifdef _DEBUG_RECEPTOR_
			mnstr_printf(RCout, "#%s", tuple);
#endif
			do {
				tick = usec();
			} while (tick == previoustsmp);

			previoustsmp = tick;

			BSKTlock(&rc->lck, &rc->name, &rc->delay);
			if (rc->status != BSKTLISTEN) {
				snr = rc->sequence;
				break;
			}
			if (insert_line(&rc->table, tuple + 1 /*ignore '[' */, (ptr) & tick, 0, rc->table.nr_attrs) < 0) {
				mnstr_printf(RCout, "failed insert_line %s\n", tuple);
				BSKTunlock(&rc->lck, &rc->name);
				break;
			}
			BSKTunlock(&rc->lck, &rc->name);
		}
		fclose(fd);
		snr++;
	} while (snr < rc->sequence);
}
/*
 * @
 * The last option is to simply associate a MAL function/factory
 * with an receptor. Its body can be used to encode
 * arbitrary complex generators. The easiest one is
 * a metronome.
 * Its implementation similar to the petrinet engine.
 */
static void
RCgenerator(Receptor rc)
{
	Symbol s;
	InstrPtr p;
	MalStkPtr glb;
	MalBlkPtr mb;
	Client cntxt = &mal_clients[0];  /* FIXME: should this be the active user? */
	int pc;

	if (rc->modnme == 0 || rc->fcnnme) {
		mnstr_printf(RCout, "Factory missing\n");
		return;
	}
	s = newFunction("user", "rcController", FACTORYsymbol);
	p = getSignature(s);
	getArg(p, 0) = newTmpVariable(mb = s->def, TYPE_void);
	/* create an execution environment */
	p = newFcnCall(mb, rc->modnme, rc->fcnnme);
	pc = getPC(mb, p);
	pushEndInstruction(mb);
	chkProgram(cntxt->fdout, cntxt->nspace, mb);
	if (mb->errors) {
		mnstr_printf(RCout, "Receptor Controller found errors\n");
		return;
	}

	newStack(glb, mb->vtop);
	memset((char *) glb, 0, stackSize(mb->vtop));
	glb->stktop = mb->vtop;
	glb->blk = mb;

#ifdef _DEBUG_RECEPTOR_
	printFunction(RCout, mb, 0, LIST_MAL_ALL);
#endif
	for (;;)
		switch (rc->status) {
		case BSKTPAUSE:
			MT_sleep_ms(1);
			break;
		case BSKTSTOP:
		case BSKTERROR:
			return;
		default:
			reenterMAL(cntxt, mb, pc, pc + 1, glb, 0, 0);
		}

}


/*
 * @-
 * The receptor thread manages the connections. Both as a active and
 * in passive mode.  The UDP channel part is not our focus right now.
 */
str
RCstartThread(Receptor rc)
{
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Receptor body %s starts at %s:%d\n", rc->name, rc->host, rc->port);
#endif

	/* Handle a server mode protocol */
#ifdef _DEBUG_RECEPTOR_
	mnstr_printf(RCout, "#Start the receptor thread, protocol=%d\n", rc->protocol);
#endif
	if (rc->mode == BSKTPASSIVE &&
		(rc->error = socket_server_connect(&rc->sockfd, rc->port))) {
		rc->status = BSKTERROR;
		mnstr_printf(RCout, "Failed to start receptor server:%s\n", rc->error);
		/* in this case there is nothing more we can do but terminate */
		return NULL;
	}
	/* the receptor should continously attempt to either connect the
	   remote site for new events or listing for the next request */
	do {
		if (rc->status == BSKTSTOP)
			break;
		if (rc->mode == BSKTPASSIVE) {
			/* in server mode you should expect new connections */
#ifdef _DEBUG_RECEPTOR_
			mnstr_printf(RCout, "#Receptor listens\n");
#endif
			rc->error = socket_server_listen(rc->sockfd, &rc->newsockfd);
			if (rc->error) {
				mnstr_printf(RCout, "Receptor listen fails: %s\n", rc->error);
				rc->status = BSKTERROR;
			}
#ifdef _DEBUG_RECEPTOR_
			mnstr_printf(RCout, "#Receptor connection request received \n");
#endif
			if (MT_create_thread(&rc->pid, (void (*)(void *))RCbody, rc, MT_THR_DETACHED) != 0) {
				mnstr_close(rc->receptor);
				throw(MAL, "receptor.start", "Process creation failed");
			}
		} else if (rc->mode == BSKTACTIVE) {
			/* take the initiative to connect to sensor */
			RCreconnect(rc);
			RCbody(rc);
		}
	} while (rc->status != BSKTSTOP);
	shutdown(rc->newsockfd, SHUT_RDWR);
	return MAL_SUCCEED;
}

static void
dumpReceptor(Receptor rc)
{
	mnstr_printf(GDKout, "#receptor %s at %s:%d protocol=%s mode=%s status=%s delay=%d \n",
			rc->name, rc->host, rc->port, protocolname[rc->protocol], modename[rc->mode], statusname[rc->status], rc->delay);
}

str
RCdump(void)
{
	Receptor rc = rcAnchor;
	for (; rc; rc = rc->nxt)
		dumpReceptor(rc);
	if (rcError)
		mnstr_printf(GDKout, "#last error event %d:%s\n", rcErrorEvent, rcError);
	return MAL_SUCCEED;
}
