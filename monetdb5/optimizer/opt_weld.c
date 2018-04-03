/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2018 MonetDB B.V.
 */

/* This optimizer replaces some MAL instructions with Weld instructions. It does
 * so by finding subgraphs of Weld portable operations in the instruction dependecy graph.
 * Instructions in the subgraphs are replaced with their Weld equivalent and are surrounded
 * by MAL-Weld helper instructions :
 * weld.initstate()                          -> generates a data structure to manage the Weld state
 * weld.algebraselect(arg1, ..., weldState)  -> produces Weld code for this op.
 * ret1,... = weld.run(weldState, arg1, ...) -> runs the Weld program
 */
#include "monetdb_config.h"
#include "mal_instruction.h"
#include "opt_weld.h"

#define NUM_WELD_INSTR 36
#define UNMARKED 0
#define TEMP_MARK 1
#define PERM_MARK 2

#define addWeldInstr(modRef, fcnRef, weldFcnRef) \
	weldInstrs[i][0] = modRef;                   \
	weldInstrs[i][1] = fcnRef;                   \
	weldInstrs[i][2] = weldFcnRef;               \
	++i;

str weldInstrs[NUM_WELD_INSTR][3];

typedef struct InstrDependecy {
	InstrPtr instr;
	str weldRef;
	int subGraphID;
	int isSubGraph;
	int topoMark;
	struct InstrDependecy **inputs, **outputs;
	int numInputs, sizeInputs;
	int numOutputs, sizeOutputs;
} InstrDep;

static void initWeldInstrs(void) {
	/* When adding or removing instructions NUM_WELD_INSTR should be updated */
	int i = 0;
	addWeldInstr(aggrRef, sumRef, weldAggrSumRef);						 /* aggr.sum */
	addWeldInstr(aggrRef, subcountRef, weldAggrSubCountRef);			 /* aggr.subcount */
	addWeldInstr(aggrRef, submaxRef, weldAggrSubMaxRef);				 /* aggr.submax */
	addWeldInstr(aggrRef, subminRef, weldAggrSubMinRef);				 /* aggr.submin */
	addWeldInstr(aggrRef, subprodRef, weldAggrSubProdRef);				 /* aggr.subprod */
	addWeldInstr(aggrRef, subsumRef, weldAggrSubSumRef);				 /* aggr.subsum */
	addWeldInstr(algebraRef, differenceRef, weldAlgebraDifferenceRef);   /* algebra.difference */
	addWeldInstr(algebraRef, intersectRef, weldAlgebraIntersectRef);	 /* algebra.intersect */
	addWeldInstr(algebraRef, joinRef, weldAlgebraJoinRef);				 /* algebra.join */
	addWeldInstr(algebraRef, projectionRef, weldAlgebraProjectionRef);   /* algebra.projection */
	addWeldInstr(algebraRef, selectRef, weldAlgebraSelectRef);			 /* algebra.select */
	addWeldInstr(algebraRef, thetaselectRef, weldAlgebraThetaselectRef); /* algebra.thetaselect */
	addWeldInstr(batcalcRef, andRef, weldBatcalcAndRef);				 /* batcalc.and */
	addWeldInstr(batcalcRef, orRef, weldBatcalcOrRef);					 /* batcalc.or */
	addWeldInstr(batcalcRef, plusRef, weldBatcalcAddRef);				 /* batcalc.+ */
	addWeldInstr(batcalcRef, minusRef, weldBatcalcSubRef);				 /* batcalc.- */
	addWeldInstr(batcalcRef, mulRef, weldBatcalcMulRef);				 /* batcalc.* */
	addWeldInstr(batcalcRef, divRef, weldBatcalcDivRef);				 /* batcalc./ */
	addWeldInstr(batcalcRef, modRef, weldBatcalcModRef);				 /* batcalc.% */
	addWeldInstr(batcalcRef, eqRef, weldBatcalcEqRef);					 /* batcalc.== */
	addWeldInstr(batcalcRef, ltRef, weldBatcalcLtRef);					 /* batcalc.< */
	addWeldInstr(batcalcRef, leRef, weldBatcalcLeRef);					 /* batcalc.<= */
	addWeldInstr(batcalcRef, gtRef, weldBatcalcGtRef);					 /* batcalc.> */
	addWeldInstr(batcalcRef, geRef, weldBatcalcGeRef);					 /* batcalc.>= */
	addWeldInstr(batcalcRef, neRef, weldBatcalcNeRef);					 /* batcalc.!= */
	addWeldInstr(batcalcRef, identityRef, weldBatcalcIdentityRef);		 /* batcalc.identity */
	addWeldInstr(batcalcRef, isnilRef, weldBatcalcIsNilRef);			 /* batcalc.isnil */
	addWeldInstr(batRef, mergecandRef, weldBatMergeCandRef);			 /* bat.mergecand */
	addWeldInstr(batRef, mirrorRef, weldBatMirrorRef);					 /* bat.mirror */
	addWeldInstr(batmtimeRef, yearRef, weldBatMtimeYearRef);			 /* batmtime.year */
	addWeldInstr(languageRef, passRef, weldLanguagePassRef);			 /* language.pass */
	addWeldInstr(groupRef, groupRef, weldGroupRef);						 /* group.group*/
	addWeldInstr(groupRef, subgroupRef, weldGroupRef);					 /* group.subgroup */
	addWeldInstr(groupRef, groupdoneRef, weldGroupRef);					 /* group.groupdone */
	addWeldInstr(groupRef, subgroupdoneRef, weldGroupRef);				 /* group.subgroupdone */
	addWeldInstr(sqlRef, projectdeltaRef, weldSqlProjectDeltaRef);		 /* sql.projectdelta */
}

static str getWeldRef(MalBlkPtr mb, InstrPtr instr) {
	(void) mb;
	int i;
	for (i = 0; i < NUM_WELD_INSTR; i++) {
		if (getModuleId(instr) == weldInstrs[i][0] && getFunctionId(instr) == weldInstrs[i][1]) {
			return weldInstrs[i][2];
		}
	}
	return NULL;
}

static void addInputInstrDep(InstrDep *instrDep, InstrDep *input) {
	int i;
	if (input == NULL) return;
	for (i = 0; i < instrDep->numInputs; i++) {
		if (instrDep->inputs[i] == input) {
			return;
		}
	}
	if (instrDep->numInputs >= instrDep->sizeInputs) {
		instrDep->sizeInputs += 8;
		instrDep->inputs = realloc(instrDep->inputs, instrDep->sizeInputs * sizeof(InstrDep *));
	}
	instrDep->inputs[instrDep->numInputs++] = input;
}

static void addOutputInstrDep(InstrDep *instrDep, InstrDep *output) {
	int i;
	if (instrDep == NULL || output == NULL) return;
	for (i = 0; i < instrDep->numOutputs; i++) {
		if (instrDep->outputs[i] == output) {
			return;
		}
	}
	if (instrDep->numOutputs >= instrDep->sizeOutputs) {
		instrDep->sizeOutputs += 8;
		instrDep->outputs = realloc(instrDep->outputs, instrDep->sizeOutputs * sizeof(InstrDep *));
	}
	instrDep->outputs[instrDep->numOutputs++] = output;
}

static void removeSubGraph(InstrDep **instrList, int size, int subGraphID) {
	int i;
	for (i = 0; i < size; i++) {
		if (instrList[i] != NULL && instrList[i]->subGraphID == subGraphID) {
			instrList[i]->subGraphID = 0;
			instrList[i]->weldRef = NULL;
		}
	}
}

/* Find weld subGraphIDs by marking nodes with a subGraph id. */
static void marksubGraph(InstrDep *instrDep, int *nextsubGraphID) {
	int i;
	if (instrDep == NULL || instrDep->weldRef == NULL)
		return;
	if (instrDep->subGraphID == 0) {
		++*nextsubGraphID;
		instrDep->subGraphID = *nextsubGraphID;
	}
	for (i = 0; i < instrDep->numInputs; i++) {
		InstrDep *input = instrDep->inputs[i];
		if (input->weldRef != NULL && input->subGraphID == 0) {
			input->subGraphID = instrDep->subGraphID;
			marksubGraph(input, nextsubGraphID);
		}
	}
	for (i = 0; i < instrDep->numOutputs; i++) {
		InstrDep *output = instrDep->outputs[i];
		if (output->weldRef != NULL && output->subGraphID == 0) {
			output->subGraphID = instrDep->subGraphID;
			marksubGraph(output, nextsubGraphID);
		}

	}
}

/* All nodes that belong to subgraph=sourceID and have an input path leading
 * to instrDep will now become part of subgraph=newID */
static void partitionSubGraph(InstrDep *instrDep, int sourceID, int newID) {
	if (instrDep == NULL)
		return;
	int i;
	for (i = 0; i < instrDep->numOutputs; i++) {
		InstrDep *output = instrDep->outputs[i];
		if (output->subGraphID == sourceID) {
			output->subGraphID = newID;
			partitionSubGraph(output, sourceID, newID);
		}
	}
}

static void changeSubGraphID(InstrDep **instrList, int size, int sourceID, int newID) {
	int i;
	for (i = 0; i < size; i++) {
		if (instrList[i] != NULL && instrList[i]->subGraphID == sourceID) {
			instrList[i]->subGraphID = newID;
		}
	}
}

/* Find cycles that would result in the DAG obtained from collapsing subgraphs.
 * It checkes whether when starting from a node not part of the subgraph there's an input dependency
 * path that leads back to a Weld subgraph with subGraphID = sourceID */
static int findWeldCycle(InstrDep *instrDep, int sourceID) {
	if (instrDep == NULL)
		return 0;
	int i;
	for (i = 0; i < instrDep->numInputs; i++) {
		InstrDep *input = instrDep->inputs[i];
		if (input->subGraphID == sourceID) {
			return 1;
		} else {
			if (findWeldCycle(input, sourceID))
				return 1;
		}
	}
	return 0;
}

static void topoSort(InstrDep *instrDep, int subGraphID, InstrDep **result, int *resultIdx) {
	int i;
	if (instrDep == NULL || instrDep->topoMark == PERM_MARK ||
		(instrDep->subGraphID != subGraphID && !instrDep->isSubGraph)) {
		return;
	}
	assert(instrDep->topoMark != TEMP_MARK);
	instrDep->topoMark = TEMP_MARK;
	for (i = 0; i < instrDep->numInputs; i++) {
		topoSort(instrDep->inputs[i], subGraphID, result, resultIdx);
	}
	instrDep->topoMark = PERM_MARK;
	result[*resultIdx] = instrDep;
	++*resultIdx;
}

/* Generate the MAL-Weld instruction that creates the state ptr */
static void initWeldState(MalBlkPtr mb, int *wstateVar) {
	InstrPtr wstate;
	*wstateVar = newVariable(mb, "wstate", 6, TYPE_ptr);
	wstate = newInstruction(0, weldRef, weldInitStateRef);
	wstate = pushReturn(mb, wstate, *wstateVar);
	pushInstruction(mb, wstate);
}

/* Create a MAL-Weld instruction */
static void convertToWeld(MalBlkPtr mb, InstrDep *instrDep, int wstateVar) {
	int i;
	InstrPtr weldInstr = newInstruction(0, weldRef, instrDep->weldRef);
	for (i = 0; i < instrDep->instr->retc; i++) {
		weldInstr = pushReturn(mb, weldInstr, getArg(instrDep->instr, i));
	}
	for (i = instrDep->instr->retc; i < instrDep->instr->argc; i++) {
		weldInstr = pushArgument(mb, weldInstr, getArg(instrDep->instr, i));
	}
	weldInstr = pushArgument(mb, weldInstr, wstateVar);
	pushInstruction(mb, weldInstr);
}

/* Add the vars produced by non-weld instrs or Weld instrs belonging to a different subgraph */
static void addWeldArgs(MalBlkPtr mb, InstrPtr *weldRun, InstrDep *instrDep, InstrDep **varInstrMap) {
	int i, j;
	for (i = instrDep->instr->retc; i < instrDep->instr->argc; i++) {
		int arg = getArg(instrDep->instr, i);
		if (varInstrMap[arg] == NULL || varInstrMap[arg]->subGraphID != instrDep->subGraphID) {
			int alreadyAdded = 0;
			for (j = (*weldRun)->retc; j < (*weldRun)->argc; j++) {
				if (arg == getArg(*weldRun, j)) {
					alreadyAdded = 1;
					break;
				}
			}
			if (!alreadyAdded) {
				*weldRun = pushArgument(mb, *weldRun, arg);
			}
		}
	}
}

/* Retrieve results produced by Weld instructions that are later needed by non-weld instrs */
static void getWeldResults(MalBlkPtr mb, InstrPtr *weldRun, InstrDep *instrDep,
						   InstrDep **instrList, int stop) {
	int i, j, k, l;
	for (i = 0; i < instrDep->instr->retc; i++) {
		int ret = getArg(instrDep->instr, i);
		for (j = 0; j < stop; j++) {
			InstrDep *outputDep = instrList[j];
			if (outputDep != NULL && outputDep->subGraphID != instrDep->subGraphID &&
				!outputDep->isSubGraph) {
				for (k = outputDep->instr->retc; k < outputDep->instr->argc; k++) {
					if (ret == getArg(outputDep->instr, k)) {
						/* If "ret" is an argument for an instr form a different subgraph or a mal
						 * instr */
						int alreadyAdded = 0;
						for (l = 0; l < (*weldRun)->retc; l++) {
							if (ret == getArg(*weldRun, l)) {
								alreadyAdded = 1;
								break;
							}
						}
						if (!alreadyAdded) {
							*weldRun = pushReturn(mb, *weldRun, ret);
						}
					}
				}
			}
		}
	}
}

str OPTweldImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p) {
	int i, j, fcnEnd, actions = 0, vtop = mb->vtop, stop = mb->stop, subGraphIDs, change;
	InstrPtr instr, *old = mb->stmt;
	InstrDep **instrList = calloc(stop, sizeof(InstrDep *));
	InstrDep **varInstrMap = calloc(vtop, sizeof(InstrDep *));
	InstrDep *subGraphReps; /* Nodes that will represent a weld subGraphID */
	InstrDep **ordInstr = calloc(stop, sizeof(InstrDep *));
	InstrDep **ordSubgraph = calloc(stop, sizeof(InstrDep *));
	int ordInstrSize, ordSubgraphSize, *subgraphCounts, numSubGraphs;
	lng usec = GDKusec();
	char buf[256];
	initWeldInstrs();
	(void)cntxt;
	(void)stk;
	(void)p;

	/* Build the dependency graph */
	for (i = 1; i < stop; i++) {
		instr = old[i];
		if (instr->token == ENDsymbol)
			break;
		InstrDep *instrDep = calloc(1, sizeof(InstrDep));
		instrDep->instr = instr;
		instrDep->weldRef = getWeldRef(mb, instr);
		instrList[i] = instrDep;
		/* Mark that the output vars depend on the current instr */
		for (j = 0; j < instr->retc; j++) {
			varInstrMap[getArg(instr, j)] = instrDep;
		}
		/* Find the instrs on which the current one depends */
		for (j = instr->retc; j < instr->argc; j++) {
			InstrDep *input = varInstrMap[getArg(instr, j)];
			addInputInstrDep(instrDep, input);
			addOutputInstrDep(input, instrDep);
		}
	}
	fcnEnd = i;

	/* Find the weld subgraphs */
	subGraphIDs = 0;
	for (i = stop - 1; i >= 0; i--) {
		marksubGraph(instrList[i], &subGraphIDs);
	}
	/* Merge adjacent subgraphs that depend on the same input MAL node */
	change = 1;
	while (change) {
		change = 0;
		for (i = stop - 1; i >= 0; i--) {
			if (instrList[i] != NULL && instrList[i]->subGraphID == 0) {
				int minOutputSubGraphID = INT_MAX;
				for (j = 0; j < instrList[i]->numOutputs; j++) {
					InstrDep *output = instrList[i]->outputs[j];
					if (output->subGraphID != 0 && output->subGraphID < minOutputSubGraphID) {
						minOutputSubGraphID = output->subGraphID;
					}
				}
				if (minOutputSubGraphID != INT_MAX) {
					for (j = 0; j < instrList[i]->numOutputs; j++) {
						InstrDep *output = instrList[i]->outputs[j];
						if (output->subGraphID != 0 && output->subGraphID != minOutputSubGraphID) {
							changeSubGraphID(instrList, stop, output->subGraphID, minOutputSubGraphID);
							change = 1;
						}
					}
				}
			}
		}
	}
	/* Remove weld dependency cycles */
	change = 1;
	while (change) {
		change = 0;
		for (i = stop - 1; i >= 0; i--) {
			if (instrList[i] != NULL && instrList[i]->subGraphID != 0) {
				for (j = 0; j < instrList[i]->numInputs; j++) {
					InstrDep *input = instrList[i]->inputs[j];
					if (input->subGraphID != instrList[i]->subGraphID) {
						if (findWeldCycle(input, instrList[i]->subGraphID)) {
							change = 1;
							++subGraphIDs;
							partitionSubGraph(input, instrList[i]->subGraphID, subGraphIDs);
							break;
						}
					}
				}
			}
		}
	}
	/* Count the number of nodes in each subgraph */
	numSubGraphs = subGraphIDs + 1;
	subgraphCounts = calloc(numSubGraphs, sizeof(int));
	for (i = 0; i < stop; i++) {
		if (instrList[i] != NULL) {
			subgraphCounts[instrList[i]->subGraphID]++;
		}
	}
	/* Remove subgraphs that have fewer than 2 nodes */
	for (i = 1; i < numSubGraphs; i++) {
		if (subgraphCounts[i] < 2) {
			removeSubGraph(instrList, stop, i);
			subgraphCounts[i] = 0;
		}
	}
	subGraphReps = calloc(numSubGraphs, sizeof(InstrDep));
	for (i = 1; i < numSubGraphs; i++) {
		if (subgraphCounts[i] != 0) {
			subGraphReps[i].subGraphID = i;
			subGraphReps[i].isSubGraph = 1;
		}
	}

	/* Collapse subgraphs into their representative nodes */
	for (i = 0; i < stop; i++) {
		InstrDep *instrDep = instrList[i];
		if (instrDep == NULL)
			continue;
		for (j = 0; j < instrDep->numInputs; j++) {
			InstrDep *input = instrDep->inputs[j];
			if (input->subGraphID != 0 && input->subGraphID != instrDep->subGraphID) {
				/* A node depends on a subgraph */
				instrDep->inputs[j] = &subGraphReps[input->subGraphID];
			}
			if (instrDep->subGraphID != 0) {
				InstrDep *subGraphRep = &subGraphReps[instrDep->subGraphID];
				if (input->subGraphID == 0) {
					/* A subgraph depends on a MAL node */
					addInputInstrDep(subGraphRep, input);
				} else if (input->subGraphID != subGraphRep->subGraphID) {
					/* A subgraph depends on another subgraph */
					addInputInstrDep(subGraphRep, &subGraphReps[input->subGraphID]);
				}
			}
		}
	}

	/* Topological sort on the new graph */
	ordInstrSize = 0;
	for (i = stop - 1; i >= 0; i--) {
		topoSort(instrList[i], 0 /*subGraphID*/, ordInstr, &ordInstrSize);
	}
	for (i = 1; i < numSubGraphs; i++) {
		if (subGraphReps[i].isSubGraph)
			topoSort(&subGraphReps[i], 0 /*subGraphID*/, ordInstr, &ordInstrSize);
	}

	if (newMalBlkStmt(mb, mb->ssize) < 0) {
		throw(MAL, "optimizer.weld", SQLSTATE(HY001) MAL_MALLOC_FAIL);
	}

	if (stop > 0) {
		pushInstruction(mb, old[0]);
	}
	for (i = 0; i < ordInstrSize; i++) {
		if (ordInstr[i]->isSubGraph) {
			int wstateVar;
			InstrPtr weldRun = newInstruction(0, weldRef, weldRunRef);
			initWeldState(mb, &wstateVar);
			ordSubgraphSize = 0;
			for (j = stop - 1; j >= 0; j--) {
				topoSort(instrList[j], ordInstr[i]->subGraphID, ordSubgraph, &ordSubgraphSize);
			}
			for (j = 0; j < ordSubgraphSize; j++) {
				getWeldResults(mb, &weldRun, ordSubgraph[j], instrList, stop);
			}
			weldRun = pushArgument(mb, weldRun, wstateVar);
			for (j = 0; j < ordSubgraphSize; j++) {
				++actions;
				addWeldArgs(mb, &weldRun, ordSubgraph[j], varInstrMap);
				convertToWeld(mb, ordSubgraph[j], wstateVar);
			}
			pushInstruction(mb, weldRun);
		} else {
			pushInstruction(mb, ordInstr[i]->instr);
		}
	}

	for (i = fcnEnd; i < stop; i++) {
		pushInstruction(mb, old[i]);
	}

	/* Clean up */
	for (i = 0; i < stop; i++) {
		if (instrList[i] != NULL) {
			free(instrList[i]->inputs);
			free(instrList[i]->outputs);
			free(instrList[i]);
		}
	}
	free(instrList);
	free(varInstrMap);
	free(subGraphReps);
	free(ordInstr);
	free(ordSubgraph);

	/* keep all actions taken as a post block comment and update statics */
	usec = GDKusec() - usec;
	snprintf(buf, 256, "%-20s actions=%2d time=" LLFMT " usec", "weld", actions, usec);
	newComment(mb, buf);
	addtoMalBlkHistory(mb);
	return MAL_SUCCEED;
}
