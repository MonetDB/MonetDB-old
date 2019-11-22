/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

/* (author) M.L. Kersten 
 */
#include "monetdb_config.h"
#include "mal_resource.h"
#include "mal_private.h"
#include "gdk_tracer.h"

/* Memory based admission does not seem to have a major impact so far. */
static lng memorypool = 0;      /* memory claimed by concurrent threads */

void
mal_resource_reset(void)
{
	memorypool = (lng) MEMORY_THRESHOLD;
}
/*
 * Running all eligible instructions in parallel creates
 * resource contention. This means we should implement
 * an admission control scheme where threads are temporarily
 * postponed if the claim for memory exceeds a threshold
 * In general such contentions will be hard to predict,
 * because they depend on the algorithm, the input sizes,
 * concurrent use of the same variables, and the output produced.
 *
 * One heuristic is based on calculating the storage footprint
 * of the operands and assuming it preferrably should fit in memory.
 * Ofcourse, there may be intermediate structures being
 * used and the size of the result is not a priori known.
 * For this, we use a high watermark on the amount of
 * physical memory we pre-allocate for the claims.
 *
 * Instructions are eligible to be executed when the
 * total footprint of all concurrent executions stays below
 * the high-watermark or it is the single expensive
 * instruction being started.
 *
 * When we run out of memory, the instruction is delayed.
 * How long depends on the other instructions to free up
 * resources. The current policy simple takes a local
 * decision by delaying the instruction based on its
 * claim of the memory.
 */

/*
 * The memory claim is the estimate for the amount of memory hold.
 * Views are consider cheap and ignored
 */
lng
getMemoryClaim(MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, int i, int flag)
{
	lng total = 0;
	BAT *b;

	(void)mb;
	if (stk->stk[getArg(pci, i)].vtype == TYPE_bat) {
		b = BATdescriptor( stk->stk[getArg(pci, i)].val.bval);
		if (b == NULL)
			return 0;
		if (flag && isVIEW(b)) {
			BBPunfix(b->batCacheid);
			return 0;
		}

		total += BATcount(b) * b->twidth;
		// string heaps can be shared, consider them as space-less views
		total += heapinfo(b->tvheap, b->batCacheid); 
		total += hashinfo(b->thash, d->batCacheid); 
		total += IMPSimprintsize(b);
		//total = total > (lng)(MEMORY_THRESHOLD ) ? (lng)(MEMORY_THRESHOLD ) : total;
		BBPunfix(b->batCacheid);
	}
	return total;
}

/*
 * A consequence of multiple threads is that they may claim more
 * space than available. This may cause GDKmalloc to fail.
 * In many cases this situation will be temporary, because
 * threads will ultimately release resources.
 * Therefore, we wait for it.
 *
 * Alternatively, a front-end can set the flow administration
 * program counter to -1, which leads to a soft abort.
 * [UNFORTUNATELY this approach does not (yet) work
 * because there seem to a possibility of a deadlock
 * between incref and bbptrim. Furthermore, we have
 * to be assured that the partial executed instruction
 * does not lead to ref-count errors.]
 *
 * The worker produces a result which will potentially unblock
 * instructions. This it can find itself without the help of the scheduler
 * and without the need for a lock. (does it?, parallel workers?)
 * It could also give preference to an instruction that eats away the object
 * just produced. THis way it need not be saved on disk for a long time.
 */
/*
 * The hotclaim indicates the amount of data recentely written.
 * as a result of an operation. The argclaim is the sum over the hotclaims
 * for all arguments.
 * The argclaim provides a hint on how much we actually may need to execute
 * The hotclaim is a hint how large the result would be.
 */
#ifdef USE_MAL_ADMISSION
static MT_Lock admissionLock = MT_LOCK_INITIALIZER("admissionLock");

int
MALadmission_claim(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, lng argclaim)
{
	(void) mb;
	(void) pci;
	if (argclaim == 0)
		return 0;

	MT_lock_set(&admissionLock);
	/* Check if we are allowed to allocate another worker thread for this client */
	/* It is somewhat tricky, because we may be in a dataflow recursion, each of which should be counted for.
	 * A way out is to attach the thread count to the MAL stacks, which just limits the level
	 * of parallism for a single dataflow graph.
	 */
	if(cntxt->workerlimit && cntxt->workerlimit < stk->workers){
		DEBUG(PAR, "Worker limit reached, %d <= %d\n", cntxt->workerlimit, stk->workers);
		MT_lock_unset(&admissionLock);
		return -1;
	}
	/* Determine if the total memory resource is exhausted, because it is overall limitation.  */
	if ( memorypool <= 0){
		// we accidently released too much memory or need to initialize
		DEBUG(PAR, "Memorypool reset\n");
		memorypool = (lng) MEMORY_THRESHOLD;
	}

	/* the argument claim is based on the input for an instruction */
	if ( memorypool > argclaim || stk->workers == 0 ) {
		/* If we are low on memory resources, limit the user if he exceeds his memory budget 
		 * but make sure there is at least one worker thread active */
		/* on hold until after experiments
		if ( 0 &&  cntxt->memorylimit) {
			if (argclaim + stk->memory > (lng) cntxt->memorylimit * LL_CONSTANT(1048576)){
				MT_lock_unset(&admissionLock);
				DEBUG(PAR, "Delayed due to lack of session memory " LLFMT " requested "LLFMT"\n", 
							stk->memory, argclaim);
				return -1;
			}
			stk->memory += argclaim;
		}
		*/
		memorypool -= argclaim;
		DEBUG(PAR, "Thread %d pool " LLFMT "claims " LLFMT "\n",
					THRgettid(), memorypool, argclaim);
		stk->workers++;
		MT_lock_unset(&admissionLock);
		return 0;
	}
	DEBUG(PAR, "Delayed due to lack of memory " LLFMT " requested " LLFMT "\n", 
			memorypool, argclaim);
	MT_lock_unset(&admissionLock);
	return -1;
}

void
MALadmission_release(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, lng argclaim)
{
	/* release memory claimed before */
	(void) cntxt;
	(void) mb;
	(void) pci;
	if (argclaim == 0 )
		return;

	MT_lock_set(&admissionLock);
	/* on hold until after experiments
	if ( 0 && cntxt->memorylimit) {
		DEBUG(PAR, "Return memory to session budget " LLFMT "\n", stk->memory);
		stk->memory -= argclaim;
	}
	*/
	memorypool += argclaim;
	if ( memorypool > (lng) MEMORY_THRESHOLD ){
		DEBUG(PAR, "Memorypool reset\n");
		memorypool = (lng) MEMORY_THRESHOLD;
	}
	stk->workers--;
	DEBUG(PAR, "Thread %d pool " LLFMT " claims " LLFMT "\n",
				THRgettid(), memorypool, argclaim);
	MT_lock_unset(&admissionLock);
	return;
}
