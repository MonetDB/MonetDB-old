/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

/* authors M. Kersten ("projection2candidate"), S. Manegold ("projection2candidate")
 *
 * This optimizer is intended to eliminate projections (algebra.projection())
 * by using the candidate list (first argument of algebra.projection()
 * directly in the MAL operation that consumes the projection result,
 * i.e., replace pattern like
 *  X_20 := algebra.projection(C_10, X_10);
 *  (..) := mod.func(.., X_20, ..);
 * by
 *  (..) := mod.func(.., X_10, .., C_10, ..);
 * when/where ever the variant of mod.func() with candidate list exists.
 *
 * In fact, for actual elimination of no longer used algebra.projection()
 * calls, this optimizer should be followed by the "deadcode" optimizer!
 *
 * This optimizer might merely be a proof-of-concept prototype
 * that eventually gets replaced by having the SQL-2-MAL code generator
 * no longer emit algebra.projection() calls, but rather mod.func() calls
 * that directly use the candidate list (and perform the implicite
 * projection on-the-fly).
 */

#include "monetdb_config.h"
#include "mal_builder.h"
#include "opt_projection2candidate.h"

#if 0
#define OPTDEBUGprojection2candidate(CODE) { CODE }
#else
#define OPTDEBUGprojection2candidate(CODE)
#endif

str
OPTprojection2candidateImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	int i,actions = 0;
	int limit = mb->stop;
	InstrPtr p, *old = mb->stmt;
	char buf[256];
	lng usec = GDKusec();
	str msg = MAL_SUCCEED;

	(void) stk;
	(void) cntxt;
	(void) pci;

	OPTDEBUGprojection2candidate(
		fprintf(stderr, "#Optimize projection2candidate\n");
		fprintFunction(stderr, mb, 0, LIST_MAL_DEBUG);
	)

	setVariableScope(mb);
	if ( newMalBlkStmt(mb, mb->ssize) < 0)
		throw(MAL,"optimizer.projection2candidate", SQLSTATE(HY001) MAL_MALLOC_FAIL);

	/* peephole optimization */
	for (i = 0; i < limit; i++) {
		p = old[i];

		if (p->token == ENDsymbol){
			for(; i<limit; i++)
				if (old[i])
					pushInstruction(mb,old[i]);
			break;
		}
#if 0
// Problems / errors:
// gdk/gdk_calc.c:5561: BATcalccstsub: Assertion `cand < candend' failed.
//  with TPCH q14
// inputs not the same size.
//  with TPCH q01, q03, q05, q07, q15, q19
		/* case 1
		 * X_527 := algebra.projection(C_353, X_329);
		 * X_535 := batcalc.-(100:lng, X_527); 
		 */
		/* UNCHNAGED LEGACY CODE (from former "projection2candidate" optimizer):
		 * - too strict in matching only batcalc.-(const-scalar,COL)
		 * - batcalc.*() implementation with candidate list for shrinking result
		 *   not yet implemented --- well, possibly partly in batcalc-candidates
		 *   branch ...
		 */
		if( getModuleId(p) == batcalcRef && *getFunctionId(p) == '-' && p->argc == 3 && isVarConstant(mb, getArg(p,1)) ){
			InstrPtr q= getInstrPtr(mb, getVar(mb,getArg(p,2))->updated);
			if ( q == 0)
				q= getInstrPtr(mb, getVar(mb,getArg(p,2))->declared);
			if( q && getArg(q,0) == getArg(p,2) && getModuleId(q) == algebraRef && getFunctionId(q) == projectionRef ){
				getArg(p,2)=  getArg(q,2);
				p= pushArgument(mb,p, getArg(q,1));
				actions++;
				OPTDEBUGprojection2candidate(
					fprintf(stderr, "#Optimize projection2candidate case 1\n");
					fprintInstruction(stderr, mb,0,p,LIST_MAL_DEBUG);
				)
			}
		}
#endif
#if 0
		/* case 2:
		 * replace
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := group.group[done](X_20);
		 * by
		 *  (..) := group.group[done](X_10, C_10);
		 * or
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := group.subgroup[done](X_20, X_21 [,X_22, X_23]);
		 * by
		 *  (..) := group.subgroup[done](X_10, C_10, X_21 [,X_22, X_23]));
		 *
		 * the latter fails for now, seemling resulting in an endless loop ...
		 */
		if ( getModuleId(p) == groupRef && ( FALSE
# if 0
// Problems / errors:
// gdk/gdk_group.c:521: BATgroup_internal: Assertion `cand < candend' failed.
//  with TPCH q09, q17, q21
// GDK reported error: BATgroupmin: b and g must be aligned
//  with TPCH q02
// Can not create object
//  with TPCH q11, q1, q21
// wrong result
//  with TPCH q04, q05
                     || ( ( getFunctionId(p) == groupRef    || getFunctionId(p) == groupdoneRef    ) &&   p->argc == p->retc + 1 )
# endif
# if 0
// Problems / errors:
// gdk/gdk_group.c:521: BATgroup_internal: Assertion `cand < candend' failed.
//  with TPCH q10
// endless loop
//  with TPCH q01, q03
// GDK reported error: BATproject: does not match always
//  with TPCH q16
// Can not create object
//  with TPCH q18
		     || ( ( getFunctionId(p) == subgroupRef || getFunctionId(p) == subgroupdoneRef ) && ( p->argc == p->retc + 2 || p->argc == p->retc + 4 ) )
# endif
		   ) )
		{
			/* found group.[sub]group[done]() */
			/* find statement the creates the first argument of group.[sub]group[done] */
			InstrPtr q= getInstrPtr(mb, getVar(mb,getArg(p,p->retc))->updated);
			if ( q == 0)
				q= getInstrPtr(mb, getVar(mb,getArg(p,p->retc))->declared);
			/* if found, check whether it is algebra.projection() */
			if( q && q->argc == 3 && getArg(q,0) == getArg(p,p->retc) && getModuleId(q) == algebraRef && getFunctionId(q) == projectionRef ){
				if (p->argc == p->retc + 1) {
					/* group.group[done]():
					 * - simply add candiate list as second argument
					 */
					p= pushArgument(mb,p, getArg(q,1)); /* cand */
				} else {
					/* group.subgroup[done]():
					 * - add new argument
					 * - shift all but the first argument by one
					 * - add candiate list as second argument
					 */
					int n = p->argc - 1;
					p= pushArgument(mb,p, getArg(p,n));
					for (--n; n > p->retc + 1; n--)
						getArg(p,n)=  getArg(p,n-1);
					getArg(p,p->retc + 1) = getArg(q,1); /* cand */
				}
				/* replace first argument of group.[sub]group[done()
				 * by original column, i.e., second argument of
				 * algebra.projection()
				 */
				getArg(p,p->retc) = getArg(q,2); /* org COL */
				actions++;
				OPTDEBUGprojection2candidate(
					fprintf(stderr, "#Optimize projection2candidate case 2\n");
					fprintInstruction(stderr, mb,0,p,LIST_MAL_DEBUG);
				)
			}
		}
#endif
#if 0
		/* case 3:
		 * replace
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := aggr.sub<avg|sum|prod>(X_20, X_21, X_22, X_23, X_24);
		 * by
		 *  (..) := aggr.sub<avg|sum|prod>(X_10, C_10, X_21, X_22, X_23, X_24);
		 * or
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := aggr.sub<min|max|count>(X_20, X_21, X_22, X_23);
		 * by
		 *  (..) := aggr.sub<min|max|count>(X_10, C_10, X_21, X_22, X_23);
		 * or
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := algebra.firstn(X_20, X_21, X_22, X_23);
		 * by
		 *  (..) := algebra.firstn(X_10, C_10, X_21, X_22, X_23);
		 * or
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := algebra.select(X_20, X_21, X_22, X_23, X_24, X_25);
		 * by
		 *  (..) := algebra.select(X_10, C_10, X_21, X_22, X_23, X_24, X_25);
		 * or
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  (..) := algebra.thetaselect(X_20, X_21, X_22);
		 * by
		 *  (..) := algebra.thetaselect(X_10, C_10, X_21, X_22);
		 */
		if ( FALSE
# if 0
// Problems / errors:
// Program contains errors.:(NONE).garbageCollector
//  with TPCH q02, q16
// GDK reported error: BATgroupsum: b and g must be aligned
//  with TPCH q01, q18, q20, q22
		     ||
		     ( getModuleId(p) == aggrRef && (
		       ( ( getFunctionId(p) == subavgRef || getFunctionId(p) == subsumRef || getFunctionId(p) == subprodRef )
		         && p->argc == p->retc + 5 ) ||
		       ( ( getFunctionId(p) == subminRef || getFunctionId(p) == submaxRef || getFunctionId(p) == subcountRef )
		         && p->argc == p->retc + 4 )
		     ) )
# endif
# if 0
// Problems / errors:
// wrong result
//  with TPCH q02, q18
		     || ( getModuleId(p) == algebraRef && getFunctionId(p) == firstnRef && p->argc == p->retc + 4 )
# endif
# if 0
// Problems / errors:
// wrong result
//  with TPCH q07, q16
		     || ( getModuleId(p) == algebraRef && getFunctionId(p) == selectRef && p->argc == p->retc + 6 )
# endif
# if 0
// Problems / errors:
// segfault as soon as mclient connects
		     || ( getModuleId(p) == algebraRef && getFunctionId(p) == thetaselectRef && p->argc == p->retc + 3 )
# endif
		   )
		{
			/* found aggr.sub<avg|sum|prod|min|max|count>() */
			/* find statement the creates the first argument of aggr.sub<avg|sum|prod|min|max|count>() */
			InstrPtr q= getInstrPtr(mb, getVar(mb,getArg(p,p->retc))->updated);
			if ( q == 0)
				q= getInstrPtr(mb, getVar(mb,getArg(p,p->retc))->declared);
			/* if found, check whether it is algebra.projection() */
			if( q && q->argc == 3 && getArg(q,0) == getArg(p,p->retc) && getModuleId(q) == algebraRef && getFunctionId(q) == projectionRef ){
				/* - add new argument
				 * - shift all but the first argument by one
				 * - add candiate list as second argument
				 */
				int n = p->argc - 1;
				p= pushArgument(mb,p, getArg(p,n));
				for (--n; n > p->retc + 1; n--)
					getArg(p,n)=  getArg(p,n-1);
				getArg(p,p->retc + 1) = getArg(q,1); /* cand */
				/* replace first argument of group.[sub]group[done()
				 * by original column, i.e., second argument of
				 * algebra.projection()
				 */
				getArg(p,p->retc) = getArg(q,2); /* org COL */
				actions++;
				OPTDEBUGprojection2candidate(
					fprintf(stderr, "#Optimize projection2candidate case 3\n");
					fprintInstruction(stderr, mb,0,p,LIST_MAL_DEBUG);
				)
			}
		}
#endif
#if 0
// Problems / errors:
// GDK reported error.
//  with TPCH q02, q05, q07, q10, q17, q18, q21
// Can not create object
//  with TPCH q03, q08, q09, q12, q13, q14. q16, q20
// wrong result
//  with TPCH q04, q15, q22
		/* case 4:
		 * replace
		 *  X_20 := algebra.projection(C_10, X_10);
		 *  X_21 := algebra.projection(C_11, X_11);
		 *  (..) := algebra.<|anti|like|ilike|left|outer|semi>join(X_20, X_21, nil:bat, nil:bat, X_24, X_25);
		 * by
		 *  (..) := algebra.<|anti|like|ilike|left|outer|semi>join(X_10, X_11, C_10, C_11, X_24, X_25);
		 */
/*
TODO:

thetajoin(l:bat[:any_1], r:bat[:any_1], sl:bat[:oid], sr:bat[:oid], op:int, nil_matches:bit, estimate:lng)

bandjoin(l:bat[:any_1], r:bat[:any_1], sl:bat[:oid], sr:bat[:oid], c1:any_1, c2:any_1, li:bit, hi:bit, estimate:lng)

ilikejoin(l:bat[:str], r:bat[:str], esc:str, sl:bat[:oid], sr:bat[:oid], nil_matches:bit, estimate:lng)
likejoin(l:bat[:str], r:bat[:str], esc:str, sl:bat[:oid], sr:bat[:oid], nil_matches:bit, estimate:lng)

rangejoin(l:bat[:any_1], r1:bat[:any_1], r2:bat[:any_1], sl:bat[:oid], sr:bat[:oid], li:bit, hi:bit, estimate:lng)
*/
		if ( getModuleId(p) == algebraRef && ( getFunctionId(p) == joinRef ||
		      getFunctionId(p) == antijoinRef || getFunctionId(p) == likejoinRef ||
		      getFunctionId(p) == ilikejoinRef || getFunctionId(p) == leftjoinRef ||
		      getFunctionId(p) == outerjoinRef || getFunctionId(p) == semijoinRef ) &&
		     p->retc == 2 && p->argc == p->retc + 6 &&
		     isVarConstant(mb, getArg(p,6)) && isVarConstant(mb, getArg(p,7))
		   )
		{
			/* found algebra.<|anti|like|ilike|left|outer|semi>join() */
			/* check left & right input/argument separately */
			int a; /* input: 2 = left, 3 = right */
			for (a = 2; a <= 3; a++) {
				int c = a + 2; /* candidates: 4 = left, 5 = right */
				if (isVarConstant(mb, getArg(p,c))) {
					/* nil:bat => no candidate, yet */
					/* find statement the creates the argument of algebra.<|anti|like|ilike|left|outer|semi>join() */
					InstrPtr q = getInstrPtr(mb, getVar(mb,getArg(p,a))->updated);
					if ( q == 0)
						q = getInstrPtr(mb, getVar(mb,getArg(p,a))->declared);
					/* if found, check whether it is algebra.projection() */
					if ( q && q->argc == 3 && getArg(q,0) == getArg(p,a) && getModuleId(q) == algebraRef && getFunctionId(q) == projectionRef ) {
						/* replace input by original column */
						getArg(p,a) = getArg(q,2); /* org COL */
						/* set candidate list */
						getArg(p,c) = getArg(q,1); /* cand */
						actions++;
						OPTDEBUGprojection2candidate(
							fprintf(stderr, "#Optimize projection2candidate case 4.%d\n", a);
							fprintInstruction(stderr, mb,0,p,LIST_MAL_DEBUG);
						)
					}
				}
			}
		}
#endif
		pushInstruction(mb,p);
	}

	OPTDEBUGprojection2candidate(
		chkTypes(cntxt->usermodule,mb,TRUE);
		freeException(msg);
		msg = MAL_SUCCEED;
		fprintf(stderr, "#Optimize projection2candidate done\n");
		fprintFunction(stderr, mb, 0, LIST_MAL_DEBUG);
	)

	GDKfree(old);
    /* Defense line against incorrect plans */
	chkTypes(cntxt->usermodule, mb, FALSE);
	chkFlow(mb);
	chkDeclarations(mb);
    /* keep all actions taken as a post block comment */
	usec = GDKusec()- usec;
    snprintf(buf,256,"%-20s actions=%2d time=" LLFMT " usec","projection2candidate",actions, usec);
    newComment(mb,buf);
	if( actions >= 0)
		addtoMalBlkHistory(mb);
	return msg;
}
