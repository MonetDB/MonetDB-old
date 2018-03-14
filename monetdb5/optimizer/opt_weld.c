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

#define NUM_WELD_INSTR 12
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
	struct InstrDependecy **inputs;
	int numInputs, sizeInputs;
} InstrDep;

static void initWeldInstrs(void) {
	/* When adding or removing instructions NUM_WELD_INSTR should be updated */
	int i = 0;
	addWeldInstr(aggrRef, sumRef, weldAggrSumRef);						 /* aggr.sum */
	addWeldInstr(algebraRef, projectionRef, weldAlgebraProjectionRef);   /* algebra.projection*/
	addWeldInstr(algebraRef, selectRef, weldAlgebraSelectRef);			 /* algebra.select */
	addWeldInstr(algebraRef, thetaselectRef, weldAlgebraThetaselectRef); /* algebra.thetaselect */
	addWeldInstr(batcalcRef, plusRef, weldBatcalcAddRef);				 /* batcalc.+ */
	addWeldInstr(batcalcRef, minusRef, weldBatcalcSubRef);				 /* batcalc.- */
	addWeldInstr(batcalcRef, mulRef, weldBatcalcMulRef);				 /* batcalc.* */
	addWeldInstr(languageRef, passRef, weldLanguagePassRef);			 /* language.pass */
	addWeldInstr(groupRef, groupRef, weldGroupRef);						 /* group.group*/
	addWeldInstr(groupRef, subgroupRef, weldGroupRef);					 /* group.subgroup */
	addWeldInstr(groupRef, groupdoneRef, weldGroupRef);					 /* group.groupdone */
	addWeldInstr(groupRef, subgroupdoneRef, weldGroupRef);				 /* group.subgroupdone */
}

static str getWeldRef(InstrPtr instr) {
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

static void removeSubGraph(InstrDep **instrList, int size, int subGraphID) {
	int i;
	for (i = 0; i < size; i++) {
		if (instrList[i] != NULL && instrList[i]->subGraphID == subGraphID) {
			instrList[i]->subGraphID = 0;
			instrList[i]->weldRef = NULL;
		}
	}
}

/* Find weld subGraphIDs by marking nodes with a subGraph id.
 * If two nodes are connected but have different subGraphIDs, the smallest
 * id is kept. Returns 1 if a change was made */
static int marksubGraphID(InstrDep *instrDep, int *nextsubGraphID) {
	if (instrDep == NULL || instrDep->weldRef == NULL) return 0;
	InstrDep *input;
	int i, change = 0;
	if (instrDep->subGraphID == 0) {
		++*nextsubGraphID;
		instrDep->subGraphID = *nextsubGraphID;
		change = 1;
	}

	for (i = 0; i < instrDep->numInputs; i++) {
		input = instrDep->inputs[i];
		if (input->weldRef == NULL) {
			continue;
		} else if (input->subGraphID == 0 || input->subGraphID > instrDep->subGraphID) {
			input->subGraphID = instrDep->subGraphID;
			marksubGraphID(input, nextsubGraphID);
			change = 1;
		} else if (input->subGraphID < instrDep->subGraphID) {
			instrDep->subGraphID = input->subGraphID;
			change = 1;
		}
	}
	return change;
}

/* Find cycles that would result in the DAG obtained from collapsing subgraphs
 */
static int findWeldCycle(InstrDep *instrDep, int sourceID) {
	if (instrDep == NULL) return 0;
	int i, result = 0;
	for (i = 0; i < instrDep->numInputs; i++) {
		InstrDep *input = instrDep->inputs[i];
		/* We started from a node in the subgraph, we're now visiting a node
		 * that is not part of the subgraph but depends on a node from the
		 * subgraph */
		if (instrDep->subGraphID != sourceID && input->subGraphID == sourceID) {
			return 1;
		} else {
			result |= findWeldCycle(input, sourceID);
		}
	}
	return result;
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

/* Add the vars produced by non-weld instrs to the weldRun instr */
static void addWeldArgs(MalBlkPtr mb, InstrPtr *weldRun, InstrPtr instr, InstrDep **varInstrMap) {
	int i, j;
	for (i = instr->retc; i < instr->argc; i++) {
		int arg = getArg(instr, i);
		if (varInstrMap[arg] == NULL || varInstrMap[arg]->weldRef == NULL) {
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
static void getWeldResults(MalBlkPtr mb, InstrPtr *weldRun, InstrPtr instr, InstrDep **instrList,
						   int stop) {
	int i, j, k, l;
	for (i = 0; i < instr->retc; i++) {
		int ret = getArg(instr, i);
		for (j = 0; j < stop; j++) {
			InstrDep *instrDep = instrList[j];
			if (instrDep != NULL && instrDep->weldRef == NULL && !instrDep->isSubGraph) {
				for (k = instrDep->instr->retc; k < instrDep->instr->argc; k++) {
					/* If "ret" is an argument for a non-weld instr */
					if (ret == getArg(instrDep->instr, k)) {
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
	int i, j, k, fcnEnd, actions = 0, vtop = mb->vtop, stop = mb->stop, subGraphIDs, change;
	InstrPtr instr, *old = mb->stmt;
	InstrDep **instrList = calloc(stop, sizeof(InstrDep *));
	InstrDep **varInstrMap = calloc(vtop, sizeof(InstrDep *));
	InstrDep *subGraphRep; /* Nodes that will represent a weld subGraphID */
	InstrDep **ordInstr = calloc(stop, sizeof(InstrDep *));
	InstrDep **ordSubgraph = calloc(stop, sizeof(InstrDep *));
	int ordInstrSize, ordSubgraphSize, *subgraphCounts, subgraphCountsSize;
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
		instrDep->weldRef = getWeldRef(instr);
		instrList[i] = instrDep;
		/* Mark that the output vars depend on the current instr */
		for (j = 0; j < instr->retc; j++) {
			varInstrMap[getArg(instr, j)] = instrDep;
		}
		/* Find the instrs on which the current one depends */
		for (j = instr->retc; j < instr->argc; j++) {
			InstrDep *input = varInstrMap[getArg(instr, j)];
			addInputInstrDep(instrDep, input);
		}
	}
	fcnEnd = i;

	/* Find the weld subgraphs */
	subGraphIDs = 0;
	change = 1;
	while (change) {
		change = 0;
		for (i = stop - 1; i >= 0; i--) {
			change |= marksubGraphID(instrList[i], &subGraphIDs);
		}
	}
	/* Remove weld dependency cycles */
	for (i = stop - 1; i >= 0; i--) {
		if (instrList[i] != NULL && instrList[i]->subGraphID != 0) {
			if (findWeldCycle(instrList[i], instrList[i]->subGraphID)) {
				/* TODO - we remove the whole graph at this time */
				removeSubGraph(instrList, stop, instrList[i]->subGraphID);
			}
		}
	}

	/* Get the number of subGraphIDs */
	subgraphCountsSize = subGraphIDs + 1;
	subgraphCounts = calloc(subgraphCountsSize, sizeof(int));
	for (i = 0; i < stop; i++) {
		if (instrList[i] != NULL) {
			subgraphCounts[instrList[i]->subGraphID]++;
		}
	}
	/* Remove subgraphs that have fewer than 2 nodes */
	for (i = 1; i < subgraphCountsSize; i++) {
		if (subgraphCounts[i] < 2) {
			removeSubGraph(instrList, stop, i);
			subgraphCounts[i] = 0;
		}
	}
	subGraphIDs = 0;
	/* Start at 1 because id = 0 is not a subgraph */
	for (i = 1; i < subgraphCountsSize; i++) {
		if (subgraphCounts[i] != 0) {
			subGraphIDs++;
		}
	}
	subGraphRep = calloc(subGraphIDs, sizeof(InstrDep));
	subGraphIDs = 0;
	for (i = 1; i < subgraphCountsSize; i++) {
		if (subgraphCounts[i] != 0) {
			subGraphRep[subGraphIDs].subGraphID = i;
			subGraphRep[subGraphIDs].isSubGraph = 1;
			subGraphIDs++;
		}
	}

	/* Collapse a subgraph into its representative node */
	for (i = 0; i < subGraphIDs; i++) {
		/* For each node connected to the subgraph but not part of it */
		for (j = stop - 1; j >= 0; j--) {
			InstrDep *instrDep = instrList[j];
			if (instrDep != NULL && instrDep->subGraphID == 0) {
				/* For each input node from the subgraph */
				for (k = 0; k < instrDep->numInputs; k++) {
					InstrDep *input = instrDep->inputs[k];
					if (input->subGraphID == subGraphRep[i].subGraphID) {
						/* Replace the node with the subgraph representative */
						instrDep->inputs[k] = &subGraphRep[i];
					}
				}
			} else if (instrDep != NULL && instrDep->subGraphID == subGraphRep[i].subGraphID) {
				/* Add the dependency of non-weld inputs */
				for (k = 0; k < instrDep->numInputs; k++) {
					InstrDep *input = instrDep->inputs[k];
					if (input->subGraphID == 0) {
						/* Replace the node with the subgraph representative */
						instrDep->inputs[k] = &subGraphRep[i];
						addInputInstrDep(&subGraphRep[i], input);
					}
				}
			}
		}
	}

	/* Topological sort on the new graph */
	ordInstrSize = 0;
	for (i = stop - 1; i >= 0; i--) {
		topoSort(instrList[i], 0 /*subGraphID*/, ordInstr, &ordInstrSize);
	}
	for (i = 0; i < subGraphIDs; i++) {
		topoSort(&subGraphRep[i], 0 /*subGraphID*/, ordInstr, &ordInstrSize);
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
			for (j = 0; j < stop; j++) {
				topoSort(instrList[j], ordInstr[i]->subGraphID, ordSubgraph, &ordSubgraphSize);
			}
			for (j = 0; j < ordSubgraphSize; j++) {
				getWeldResults(mb, &weldRun, ordSubgraph[j]->instr, instrList, stop);
			}
			weldRun = pushArgument(mb, weldRun, wstateVar);
			for (j = 0; j < ordSubgraphSize; j++) {
				++actions;
				addWeldArgs(mb, &weldRun, ordSubgraph[j]->instr, varInstrMap);
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
			free(instrList[i]);
		}
	}
	free(instrList);
	free(varInstrMap);
	free(subGraphRep);
	free(ordInstr);
	free(ordSubgraph);

	/* keep all actions taken as a post block comment and update statics */
	usec = GDKusec() - usec;
	snprintf(buf, 256, "%-20s actions=%2d time=" LLFMT " usec", "weld", actions, usec);
	newComment(mb, buf);
	addtoMalBlkHistory(mb);
	return MAL_SUCCEED;
}
