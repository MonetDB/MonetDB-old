/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "mal_exception.h"
#include "mal_interpreter.h"
#include "mal_instruction.h"
#include "mal_weld.h"
#include "weld.h"

#define STR_BUF_SIZE 4096

#define getOrSetStructMemberImpl(ADDR, TYPE, VALUE, OP)                 \
	if ((long)*ADDR % sizeof(TYPE) != 0)                                \
		*ADDR += sizeof(TYPE) - (long)*ADDR % sizeof(TYPE); /* aling */ \
	if (OP == OP_GET)                                                   \
		*(TYPE *)VALUE = *(TYPE *)(*ADDR); /* get */                    \
	else                                                                \
		*(TYPE *)(*ADDR) = *(TYPE *)VALUE; /* set */                    \
	*ADDR += sizeof(TYPE);				   /* increase */

struct weldMinMax {
	char i8min, i8max;
	int i32min, i32max;
	long i64min, i64max;
	float f32min, f32max;
	double f64min, f64max;
} weldMinMaxInst = {SCHAR_MIN, SCHAR_MAX, INT_MIN, INT_MAX, LLONG_MIN,
					LLONG_MAX, FLT_MIN,   FLT_MAX, DBL_MIN, DBL_MAX};

struct progCache {
	str prog;
	weld_module_t module;
};

struct progCache *progCache = NULL;
int progCacheSize = 0;
int progCacheMaxSize = 0;

static void addToProgCache(str prog, weld_module_t module) {
	int i;
	for (i = 0; i < progCacheSize; i++) {
		if (strcmp(progCache[i].prog, prog) == 0) {
			return;
		}
	};
	if (progCacheSize == progCacheMaxSize) {
		if (progCacheMaxSize == 0) {
			progCacheMaxSize = 10;
		} else {
			progCacheMaxSize *= 2;
		}
		progCache = realloc(progCache, progCacheMaxSize * sizeof(struct progCache));
	}
	progCache[progCacheSize].prog = strdup(prog);
	progCache[progCacheSize].module = module;
	progCacheSize++;
}

static weld_module_t* getCachedModule(str prog) {
	int i;
	for (i = 0; i < progCacheSize; i++) {
		if (strcmp(progCache[i].prog, prog) == 0) {
			return &progCache[i].module;
		}
	}
	return NULL;
}

void getOrSetStructMember(char **addr, int type, const void *value, int op) {
	if (type == TYPE_bte) {
		getOrSetStructMemberImpl(addr, char, value, op);
	} else if (type == TYPE_int) {
		getOrSetStructMemberImpl(addr, int, value, op);
	} else if (type == TYPE_lng) {
		getOrSetStructMemberImpl(addr, long, value, op);
	} else if (type == TYPE_flt) {
		getOrSetStructMemberImpl(addr, float, value, op);
	} else if (type == TYPE_dbl) {
		getOrSetStructMemberImpl(addr, double, value, op);
	} else if (type == TYPE_str) {
		getOrSetStructMemberImpl(addr, char*, value, op);
	} else if (type == TYPE_ptr) {
		/* TODO - will assume that all pointers have the same size */
		getOrSetStructMemberImpl(addr, char*, value, op);
	} else if (ATOMstorage(type) != type) {
		return getOrSetStructMember(addr, ATOMstorage(type), value, op);
	}
}

str getWeldType(int type) {
	if (type == TYPE_bte)
		return "i8";
	if (type == TYPE_sht)
		return "i16";
	else if (type == TYPE_int)
		return "i32";
	else if (type == TYPE_lng)
		return "i64";
	else if (type == TYPE_flt)
		return "f32";
	else if (type == TYPE_dbl)
		return "f64";
	else if (type == TYPE_str)
		return "vec[i8]";
	else if (ATOMstorage(type) != type)
		return getWeldType(ATOMstorage(type));
	else
		return NULL;
}

str getWeldTypeSuffix(int type) {
	if (type == TYPE_bte)
		return "c";
	if (type == TYPE_sht)
		return "si";
	else if (type == TYPE_int)
		return "";
	else if (type == TYPE_lng)
		return "L";
	else if (type == TYPE_flt)
		return "f";
	else if (type == TYPE_dbl)
		return "";
	else if (ATOMstorage(type) != type)
		return getWeldType(ATOMstorage(type));
	else
		return NULL;
}

static str getWeldUTypeFromWidth(int width) {
	if (width == 1)
		return "u8";
	else if (width == 2)
		return "u16";
	else if (width == 4)
		return "u32";
	else
		return "u64";
}

/*
static int getMalTypeFromWidth(int width) {
	if (width == 1)
		return TYPE_bte;
	else if (width == 2)
		return TYPE_sht;
	else if (width == 4)
		return TYPE_int;
	else
		return TYPE_lng;
}
*/

void dumpWeldProgram(str program, FILE *f) {
	int i, j, tabs = 0, print_tabs = 0, print_before = 1;
	for (i = 0; i < (int)strlen(program); i++) {
		char curr = program[i];
		char prev = i > 0 ? program[i - 1] : '\0';
		if (curr == '(' || (curr == '|' && prev != ' ')) {
			++tabs;
			print_tabs = 1;
		} else if (curr == ';') {
			print_tabs = 1;
		} else if (curr == ')') {
			--tabs;
			print_before = 0;
			print_tabs = 1;

		}
		if (print_before)
			fputc(curr, f);
		if (print_tabs) {
			fputc('\n', f);
			for (j = 0; j < tabs; j++) {
				fputc('\t', f);
			}
		}
		if (!print_before)
			fputc(curr, f);
		print_tabs = 0;
		print_before = 1;
	}
}


static void dumpProgram(MalBlkPtr mb, str program) {
	FILE *f = fopen(tmpnam(NULL), "w");
	int i;
	for (i = 0; i < mb->stop; i++) {
		fprintInstruction(f, mb, NULL, mb->stmt[i], LIST_MAL_ALL);
	}
	fprintf(f, "\n\n\n\n");
	dumpWeldProgram(program, f);
	fclose(f);
}

static long getTimeNowMs(void) {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
}

static BAT*
BATundense(BAT *b) {
	/* A dense BAT should always be of TYPE_oid ? */
	BAT *newb = COLnew(b->hseqbase, TYPE_oid, BATcount(b), TRANSIENT);
	oid *base = (oid*)Tloc(newb, 0);
	for (oid i = b->tseqbase; i < b->tseqbase + BATcount(b); i++) {
		*base = i;
		base++;
	}
	BATsetcount(newb, BATcount(b));
	return newb;
}

str
WeldRun(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	(void)cntxt;

	str programBody = *getArgReference_str(stk, pci, pci->retc);
	str program = malloc(strlen(programBody) + 2 * STR_BUF_SIZE);
	int *names = (int*)*getArgReference_ptr(stk, pci, pci->retc + 1);
	int i, j, headerLen = 0;

	/* Build the input stmt, e.g.: |in13:i32, in50:vec[i8]| */
	for (i = pci->retc + 2; i < pci->argc; i++) { /* skip wstate and names */
		int type = getArgType(mb, pci, i);
		int namesIdx = i - (pci->retc + 2);
		if (isaBatType(type) && getBatType(type) != TYPE_str) {
			headerLen += sprintf(program + headerLen, " in%d:vec[%s],", names[namesIdx],
								 getWeldType(getBatType(type)));
		} else if (isaBatType(type) && getBatType(type) == TYPE_str) {
			bat bid = *getArgReference_bat(stk, pci, i);
			BAT *b = BATdescriptor(bid);
			if (b == NULL) throw(MAL, "weld.run", SQLSTATE(HY002) RUNTIME_OBJECT_MISSING": %d", getArg(pci, i));
			headerLen += sprintf(
				program + headerLen, " in%d:vec[%s], in%dstr:vec[i8], in%dstroffset:i64,",
				names[namesIdx], getWeldUTypeFromWidth(b->twidth), names[namesIdx], names[namesIdx]);
		} else {
			headerLen +=
				sprintf(program + headerLen, " in%d:%s,", names[namesIdx], getWeldType(type));
		}
	}
	headerLen += sprintf(program + headerLen,
						"i8MIN:i8, i8MAX:i8, i32MIN:i32, i32MAX:i32, i64MIN:i64, i64MAX:i64, "
						"f32MIN:f32, f32MAX:f32, f64MIN:f64, f64MAX:f64,");
	headerLen += sprintf(program + headerLen,
						 "i8nil:i8, i32nil:i32, oidnil:i64, i64nil:i64, f32nil:f32, f64nil:f64,");

	program[0] = '|';
	program[headerLen - 1] = '|';
	program = strcat(program, programBody);

	weld_error_t e = weld_error_new();
	weld_conf_t conf = weld_conf_new();
	(void)dumpProgram; /* supress the unused warning */
#ifdef WELD_DEBUG
	dumpProgram(mb, program);
	weld_conf_set(conf, "weld.compile.dumpCode", "true");
	weld_conf_set(conf, "weld.compile.dumpCodeDir", "/tmp");
#endif
	char nrThreads[8], memLimit[64];
	sprintf(nrThreads, "%d", GDKnr_threads);
	sprintf(memLimit, "%ld", 256L * 1L << 30); /* 256 GB */
	weld_conf_set(conf, "weld.threads", nrThreads);
	weld_conf_set(conf, "weld.memory.limit", memLimit);
	weld_conf_set(conf,"weld.optimization.passes", "infer-size,short-circuit-booleans,predicate,vectorize,fix-iterate");
	weld_module_t *cachedModule = getCachedModule(program);
	weld_module_t m;
	if (cachedModule != NULL) {
		m = *cachedModule;
		fprintf(stderr, "0,");
	} else {
		long start = getTimeNowMs();
		m = weld_module_compile(program, conf, e);
		if (weld_error_code(e)) {
			throw(MAL, "weld.run", PROGRAM_GENERAL ": %s", weld_error_message(e));
		}
		long elapsed = getTimeNowMs() - start;
		fprintf(stderr, "%ld,", elapsed);
		addToProgCache(program, m);
	}

	/* Prepare the input for Weld. We're building an array that has the layout of a struct */
	/* Max possible size is when we only have string bats: 2 ptrs for theap and tvheap and 4 lngs
	 * for batCount, hseqbase, stroffset and tvheap->size. */
	char *inputStruct = malloc((pci->argc - pci->retc) * (2 * sizeof(void *) + 3 * sizeof(lng)));
	char *inputPtr = inputStruct;
	for (i = pci->retc + 2; i < pci->argc; i++) { /* skip wstate and names */
		int type = getArgType(mb, pci, i);
		if (isaBatType(type)) {
			bat bid = *getArgReference_bat(stk, pci, i);
			BAT *b = BATdescriptor(bid);
			if (b == NULL) throw(MAL, "weld.run", SQLSTATE(HY002) RUNTIME_OBJECT_MISSING": %d", getArg(pci, i));
			if (BATtdense(b)) {
				b = BATundense(b);
			}
			getOrSetStructMember(&inputPtr, TYPE_ptr, &b->theap.base, OP_SET);
			getOrSetStructMember(&inputPtr, TYPE_lng, &b->batCount, OP_SET);
			if (getBatType(type) == TYPE_str) {
				getOrSetStructMember(&inputPtr, TYPE_str, &b->tvheap->base, OP_SET);
				getOrSetStructMember(&inputPtr, TYPE_lng, &b->tvheap->size, OP_SET);
				lng offset = b->twidth <= 2 ? GDK_VAROFFSET : 0;
				getOrSetStructMember(&inputPtr, TYPE_lng, &offset, OP_SET);
			}
		} else {
			getOrSetStructMember(&inputPtr, type, getArgReference(stk, pci, i), OP_SET);
			if (type == TYPE_str) {
				long len = strlen(*getArgReference_str(stk, pci, i));
				getOrSetStructMember(&inputPtr, TYPE_lng, &len, OP_SET);
			}
		}
	}
	getOrSetStructMember(&inputPtr, TYPE_bte, &weldMinMaxInst.i8min, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_bte, &weldMinMaxInst.i8max, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_int, &weldMinMaxInst.i32min, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_int, &weldMinMaxInst.i32max, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_lng, &weldMinMaxInst.i64min, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_lng, &weldMinMaxInst.i64max, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_flt, &weldMinMaxInst.f32min, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_flt, &weldMinMaxInst.f32max, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_dbl, &weldMinMaxInst.f64min, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_dbl, &weldMinMaxInst.f64max, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_bte, &bte_nil, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_int, &int_nil, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_lng, &lng_nil, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_oid, &oid_nil, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_flt, &flt_nil, OP_SET);
	getOrSetStructMember(&inputPtr, TYPE_dbl, &dbl_nil, OP_SET);

	/* Run the weld program */
	weld_value_t arg = weld_value_new(inputStruct);
	weld_value_t result = weld_module_run(m, conf, arg, e);

	/* Retrieve the output */
	char *outputStruct = weld_value_data(result);
	for (i = 0; i < pci->retc; i++) {
		int type = getArgType(mb, pci, i);
		if (isaBatType(type)) {
			BAT *b = NULL;
			char *base = NULL;
			long size = 0;
			getOrSetStructMember(&outputStruct, TYPE_ptr, &base, OP_GET);
			getOrSetStructMember(&outputStruct, TYPE_lng, &size, OP_GET);
			if (getBatType(type) == TYPE_str) {
				char *strbase = NULL;
				char *strsize = 0;
				getOrSetStructMember(&outputStruct, TYPE_str, &strbase, OP_GET);
				getOrSetStructMember(&outputStruct, TYPE_lng, &strsize, OP_GET);
				/* Find the matching vheap from the input bats */
				for (j = pci->retc; j < pci->argc; j++) {
					int inputType = getArgType(mb, pci, j);
					if (isaBatType(inputType) && getBatType(inputType) == TYPE_str) {
						bat inid = *getArgReference_bat(stk, pci, j);
						BAT *in = BATdescriptor(inid);
						if (in == NULL)
							throw(MAL, "weld.run", SQLSTATE(HY002) RUNTIME_OBJECT_MISSING ": %d",
								  getArg(pci, j));
						if (in->tvheap->base == strbase) {
							BBPshare(in->tvheap->parentid);
							b = COLnew(0, TYPE_lng, size, TRANSIENT);
							b->tsorted = b->trevsorted = 0;
							b->tvheap = in->tvheap;
							b->ttype = in->ttype;
							b->tsorted = in->tsorted;
							b->tvarsized = 1;
							memcpy(b->theap.base, base, size * BATatoms[TYPE_lng].size);
							break;
						}
					}
				}
			} else {
				b = COLnew(0, getBatType(type), size, TRANSIENT);
				b->tsorted = b->trevsorted = 0;
				memcpy(b->theap.base, base, size * b->twidth);
			}
			BATsetcount(b, size);
			BBPkeepref(b->batCacheid);
			*getArgReference_bat(stk, pci, i) = b->batCacheid;
		} else {
			/* TODO handle strings */
			getOrSetStructMember(&outputStruct, type, getArgReference(stk, pci, i), OP_GET);
		}
	}

	weld_error_free(e);
	weld_value_free(arg);
	weld_value_free(result);
	weld_conf_free(conf);
	free(program);
	free(inputStruct);
	return MAL_SUCCEED;
}
