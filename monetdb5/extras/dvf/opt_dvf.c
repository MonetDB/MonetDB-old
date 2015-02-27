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
 * Copyright August 2008-2015 MonetDB B.V.
 * All Rights Reserved.
 */

/*
 * @a Yagiz Kargin
 * @- Data Vault Framework Optimizer
 *
 * This optimizer is for "indirect" rewriting of MAL plans that access the 'data'
 * tables of a data vault in the framework.
 *
 * Its indirectness comes from the fact that this optimizer is to inject only
 * one MAL line (a pattern call) somewhere in the MAL plan, and that pattern call
 * will do the real rewriting in the execution time.
 *
 * So, the aim of this optimizer is to find the right place to inject the pattern
 * call.
 *
 * The right place for injection is right after we get to know about the files
 * which have to be mounted (or possibly brought). It should also be before any
 * 'data' table is bound.
 *
 * For that purpose, the optimizer's task is to recognize MAL code
 * patterns like
 *
 *  ...
 *  v3 := sql.bind(..., schema_name, meta_table_name, "file_id", 0);
 *  ...
 *  v4 := algebra.leftjoin(v2, v3);
 *  .
 *  .
 *  .
 *  v6 := algebra.leftjoin(v5, v4);
 *  ...
 *  v7 := sql.bind(..., schema_name, data_table_name, ..., ...);
 *  ...
 *
 * WHERE the same pattern does not occur between assignments of v6 and v7.
 * Namely, the subpattern of v3, v4, ..., v6 should not occur in '...' before v7.
 * (Note that from v4 to v6 always the similar leftjoin pattern occurs.)
 *
 * And transforms them into
 *
 *  ...
 *  v3 := sql.bind(..., schema_name, meta_table_name, "file_id", 0);
 *  ...
 *  v4 := algebra.leftjoin(v2, v3);
 *  .
 *  .
 *  .
 *  v6 := algebra.leftjoin(v5, v4);
 *  (t1, t2) := group.done(v6);
 *  t3 := bat.mirror(t1);
 *  t4 := algebra.leftjoin(t3, v6);
 *  dvf.plan_modifier(schema_name, t4);
 *  ...
 *  v7 := sql.bind(..., schema_name, data_table_name, ..., ...);
 *  ...
 *
 * The optimizer looks for this pattern because v2 is supposed to have the oids
 * of meta_table_name with all the selections on that meta_table_name are applied
 * (it is all selections because of the WHERE clause) and v5 is supposed to have
 * the oids of (an)other meta_table_name with all the selections on that
 * meta_table_name are applied. Then v6 is supposed to provide the required
 * file_locations. Line with v7 is there, because the required file_locations
 * should be collected before the first bind to a data_table happens.
 *
 * The first 3 lines directly after assignment to v6 are to get the distinct
 * file_locations.
 *
 *
 * WARNING: This optimizer should run without the dataflow optimizations!
*/



#include "monetdb_config.h"
#include "opt_dvf.h"
#include "dvf.h"
#include "mal_interpreter.h"
#include "opt_statistics.h"

static int
OPTdvfImplementation(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci, int mode)
{
	//TODO: Replace these with a proper (global) constants
	str sys_schema_name = "sys";
	str schema_name = "mseed";
	str data_table_identifier = "data";
	str file_identifier_str = "file_id";
// 	str files_table_name = "files";
// 	str file_location_str = "file_location";
	str mountRef = putName("mount", 5);
	str miniseedRef = putName("miniseed", 8);
	str dvfRef = putName("dvf", 3);
	str planmodifierRef = putName("plan_modifier", 13);
	str fetchfileidsandlocationsRef = putName("fetch_file_ids_and_locations", 28);
	str barrier_to_defend_against_dataflow_identifier = "dataflow_defender";
	
	str fetchRef = putName("fetch", 5);

	//states of finding the pattern
	int state = 0; //0: start, 1:found v3, 2:found v5, 3:done with injection;

	//state variables (instruction index) numbered with state
	int i1 = 0, i2 = 0;
	
	//

	InstrPtr *old = NULL, q = NULL, r = NULL, t = NULL, b = NULL, bar = NULL, m = NULL, e = NULL, f = NULL, ffiali = NULL, *ps_iter = NULL;
	int i, limit, which_column, actions = 0;
	int last_bind_return_var_id = -1;
	int last_data_tid_return_var_id = -1;
	int last_insert_bind_return_var_id = -1;
	int last_subselect_return_var_id = -1;
	int last_update_bind_second_return_var_id = -1;
	int last_subdelta_return_var_id = -1;
	
	int var_sql_mvc = 0;
	int barrier_to_defend_against_dataflow = 0;

	stk = stk; //to escape 'unused' parameter error.
	pci = pci; //to escape 'unused' parameter error.
	cntxt = cntxt; //to escape 'unused' parameter error.

	/* check for logical error: mb must never be NULL */
	assert (mb != NULL);

	/* save the old stage of the MAL block */
	old = mb->stmt;
	limit= mb->stop;
	
	printf("mode:%d\n", mode);
	
	/* iterate over the instructions of the input MAL program */
	for (i = 1; i < limit; i++, limit = mb->stop) /* the plan signature can be skipped safely */
	{
		InstrPtr p = old[i];

		/* check for
		 * v3 := sql.bind(..., schema_name, meta_table_name, "file_id", 0);
		 */
		if(getModuleId(p) == sqlRef &&
			getFunctionId(p) == bindRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			strcmp(getVarConstant(mb, getArg(p, 2)).val.sval, sys_schema_name) != 0 &&
			strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, data_table_identifier) != 0 &&
			strcmp(getVarConstant(mb, getArg(p, 4)).val.sval, file_identifier_str) == 0 &&
			getVarConstant(mb, getArg(p, 5)).val.ival == 0 &&
			state < 3)
		{
			i1 = i;
			state = 1;
			var_sql_mvc = getArg(p, 1);
		}
		/* check for
		 * v4 := algebra.leftjoin(v2, v3);
		 */
		else if((state == 1 || state == 2) &&
			getModuleId(p) == algebraRef &&
			getFunctionId(p) == leftfetchjoinRef &&
			p->argc == 3 &&
			p->retc == 1 &&
			((getModuleId(old[i-1]) == sqlRef &&
			getFunctionId(old[i-1]) == projectdeltaRef &&
			getArg(p, 2) == getArg(old[i-1], 0) &&
			getArg(old[i-1], 2) == getArg(old[i1], 0)) ||
			(getModuleId(old[i-1]) == sqlRef &&
			getFunctionId(old[i-1]) == deltaRef &&
			getArg(p, 2) == getArg(old[i-1], 0) &&
			getArg(old[i-1], 1) == getArg(old[i1], 0))))
		{
			i2 = i;
			state = 2;

			/* check for
			 * v6 := algebra.leftjoin(v5, v4);
			 * or series thereof.
			 */
			for(i = i+1; i < limit; i++)
			{
				p = old[i];

				if(getModuleId(p) == algebraRef &&
					getFunctionId(p) == leftfetchjoinRef &&
					p->argc == 3 &&
					p->retc == 1 &&
					getArg(p, 2) == getArg(old[i-1], 0))
				{
					i2 = i;
				}
				else
				{
					i = i-1;
					break;
				}
			}
		}
		/* check for
		 * v7 := sql.bind(..., schema_name, data_table_name, ..., ...);
		 */
		else if((state == 1 || state == 2 || state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == bindRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			strcmp(getVarConstant(mb, getArg(p, 2)).val.sval, getVarConstant(mb, getArg(old[i1], 2)).val.sval) == 0 &&
			strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, data_table_identifier) == 0 &&
			getVarConstant(mb, getArg(p, 5)).val.ival == 0)
		{

			switch(state)
			{
				case 1:
					/* Error! What to do? This shouldn't happen! */
					return -1;
					//throw(MAL,"optimizer.DVframework", "Schema of %s vault is not well-organized.\n", getVarConstant(mb, getArg(p, 2)).val.sval);
				case 2:
					/* pattern found! What to do */
					
					if(mode == 1)
					{
						/* v7:bat[:oid,:int] := bat.new(:oid,:int); #BAT for file_id in data
						 * v8:bat[:oid,:int] := bat.new(:oid,:int); #BAT for seq_no in data
						 * v9:bat[:oid,:int] := bat.new(:oid,:timestamp); #BAT for sample_time in data
						 * ... #BAT for sample_value in data
						 * (t1, t2) := group.done(v6);
						 *  t3 := bat.mirror(t1);
						 *  t4 := algebra.leftjoin(t3, v6);
						 * 
						 * (X_21,X_22) := dvf.fetch_file_ids_and_locations("mseed",t4);
						 * 
						 * barrier (o, fileId) := iterator.new(X_21);
						 * fileLocation:= algebra.fetch(X_22,o);
						 * (v71:bat[:oid,:int], v81:bat[:oid,:int], v91:bat[:oid,:timestamp], ...) :=miniseed.mount(fileId,fileLocation);
						 * v7 := sql.append(v7,v71);
						 * v8 := sql.append(v8,v81);
						 * v9 := sql.append(v9,v91);
						 * ...
						 * redo (o, fileId) := iterator.next(X_21);
						 * exit (o,fileId);
						 */
						
						int a = 0, type;
						ps_iter = (InstrPtr*)GDKmalloc(2*NUM_RET_MOUNT*sizeof(InstrPtr)); /* for the bat.new and bat.append commands. */
						
						/* create group.done instruction */
						q = newInstruction(mb, ASSIGNsymbol);
						setModuleId(q, groupRef);
						setFunctionId(q, subgroupRef);
						q = pushReturn(mb, q, newTmpVariable(mb, TYPE_bat));
						q = pushReturn(mb, q, newTmpVariable(mb, TYPE_bat));
						q = pushReturn(mb, q, newTmpVariable(mb, TYPE_bat));
						q = pushArgument(mb, q, getArg(old[i2], 0));
						
						
						/* create bat.mirror instruction */
// 						s = newInstruction(mb, ASSIGNsymbol);
// 						setModuleId(s, batRef);
// 						setFunctionId(s, mirrorRef);
// 						s = pushReturn(mb, s, newTmpVariable(mb, TYPE_bat));
// 						s = pushArgument(mb, s, getArg(q, 0));
						
						/* create algebra.leftjoin instruction */
						t = newInstruction(mb, ASSIGNsymbol);
						setModuleId(t, algebraRef);
						setFunctionId(t, leftfetchjoinRef);
						t = pushReturn(mb, t, newTmpVariable(mb, TYPE_bat)); /* t4 */
						t = pushArgument(mb, t, getArg(q, 1));
						t = pushArgument(mb, t, getArg(old[i2], 0));
						
						/* create dvf.fetch_file_ids_and_locations instruction */
						ffiali = newInstruction(mb, ASSIGNsymbol);
						setModuleId(ffiali, dvfRef);
						setFunctionId(ffiali, fetchfileidsandlocationsRef);
						ffiali = pushReturn(mb, ffiali, newTmpVariable(mb, TYPE_bat)); /* X_21 */
						ffiali = pushReturn(mb, ffiali, newTmpVariable(mb, TYPE_bat)); /* X_22 */
						ffiali = pushArgument(mb, ffiali, var_sql_mvc);
						ffiali = pushStr(mb, ffiali, schema_name); /* schema_name */
						ffiali = pushArgument(mb, ffiali, getArg(t, 0)); /* t4 */
						
						
						/* create barrier instruction */
						b = newInstruction(mb, ASSIGNsymbol);
						setModuleId(b, iteratorRef);
						setFunctionId(b, newRef);
						b->barrier = BARRIERsymbol;
						b = pushReturn(mb, b, newTmpVariable(mb, TYPE_any)); /* o */
						b = pushReturn(mb, b, newTmpVariable(mb, TYPE_any)); /* fileId iterator */
						b = pushArgument(mb, b, getArg(ffiali, 0)); /* X_21 */
						
						/* create redo instruction */
						r = newInstruction(mb, ASSIGNsymbol);
						setModuleId(r, iteratorRef);
						setFunctionId(r, nextRef);
						r->barrier = REDOsymbol;
						r = pushReturn(mb, r, getArg(b, 0)); /* o */
						r = pushReturn(mb, r, getArg(b, 1)); /* fileId iterator */
						r = pushArgument(mb, r, getArg(ffiali, 0)); /* X_21 */
						
						/* create algebra.fetch instruction */
						f = newInstruction(mb, ASSIGNsymbol);
						setModuleId(f, algebraRef);
						setFunctionId(f, fetchRef);
						f = pushReturn(mb, f, newTmpVariable(mb, TYPE_str)); /* fileLocation iterator */
						f = pushArgument(mb, f, getArg(ffiali, 1));  /* X_22 */
						f = pushArgument(mb, f, getArg(b, 0)); /* o */
						
						/* create mount instruction */
						m = newInstruction(mb, ASSIGNsymbol);
						setModuleId(m, miniseedRef);
						setFunctionId(m, mountRef);
						
						for(; a < NUM_RET_MOUNT; a++)
						{
							type = get_column_type(schema_name, data_table_identifier, a);
							if(type < 0)
							{
								printf("dvf.get_column_type is not defined yet for schema: %s and table: %s and column: %d.\n", schema_name, data_table_identifier, a);
								return -1;
							}
							
							type = newBatType(TYPE_oid, type);
							
							/* create bat.new instructions */
							ps_iter[a] = newInstruction(mb, ASSIGNsymbol);
							setModuleId(ps_iter[a], batRef);
							setFunctionId(ps_iter[a], newRef);
							ps_iter[a] = pushReturn(mb, ps_iter[a], newTmpVariable(mb, type)); /* v7, v8, ... */
							ps_iter[a] = pushType(mb, ps_iter[a], getHeadType(type));
							ps_iter[a] = pushType(mb, ps_iter[a], getTailType(type));
							
							/* push returns of mount instruction */
							m = pushReturn(mb, m, newTmpVariable(mb, type));
							
							/* create sql.append instructions */
							ps_iter[a+NUM_RET_MOUNT] = newInstruction(mb, ASSIGNsymbol);
							setModuleId(ps_iter[a+NUM_RET_MOUNT], batRef);
							setFunctionId(ps_iter[a+NUM_RET_MOUNT], appendRef);
							ps_iter[a+NUM_RET_MOUNT] = pushReturn(mb, ps_iter[a+NUM_RET_MOUNT], getArg(ps_iter[a], 0));
							ps_iter[a+NUM_RET_MOUNT] = pushArgument(mb, ps_iter[a+NUM_RET_MOUNT], getArg(ps_iter[a], 0));
							ps_iter[a+NUM_RET_MOUNT] = pushArgument(mb, ps_iter[a+NUM_RET_MOUNT], getArg(m, a));
						}
						
						/* push args of mount instruction */
						m = pushArgument(mb, m, getArg(b, 1)); /* fileId iterator */
						m = pushArgument(mb, m, getArg(f, 0)); /* fileLocation iterator */
						
						/* create exit instruction */
						e = newInstruction(mb, ASSIGNsymbol);
						e->barrier = EXITsymbol;
						e = pushReturn(mb, e, getArg(b, 0)); /* o */
						e = pushReturn(mb, e, getArg(b, 1)); /* fileId iterator */
						
						/* insert the new instructions in pc i2+1 */
						insertInstruction(mb, e, i2+1);
						insertInstruction(mb, r, i2+1);
						for(a = NUM_RET_MOUNT-1; a >= 0; a--)
						{
							insertInstruction(mb, ps_iter[a+NUM_RET_MOUNT], i2+1);
						}
						insertInstruction(mb, m, i2+1);
						insertInstruction(mb, f, i2+1);
						insertInstruction(mb, b, i2+1);
						
						insertInstruction(mb, ffiali, i2+1);
						
						insertInstruction(mb, t, i2+1);
// 						insertInstruction(mb, s, i2+1);
						insertInstruction(mb, q, i2+1);
						
						for(a = NUM_RET_MOUNT-1; a >= 0; a--)
						{
							insertInstruction(mb, ps_iter[a], i2+1);
						}
						
						i += 8 + NUM_RET_MOUNT * 2 - 1; /* -1 because this bind should be reconsidered with state=3 in the next iteration. */
						actions += 8 + NUM_RET_MOUNT * 2;
						state = 3;
						
					}
					else
					{
						/* (t1, t2) := group.done(v6);
						*  t3 := bat.mirror(t1);
						*  t4 := algebra.leftjoin(t3, v6);
						* 
						* (X_21,X_22) := dvf.fetch_file_ids_and_locations(var_sql_mvc,schema_name,t4);
						* 
						*  dvf.plan_modifier(schema_name, X_21, X_22, mode);
						*/

						/* create group.done instruction */
						r = newInstruction(mb, ASSIGNsymbol);
						setModuleId(r, groupRef);
						setFunctionId(r, subgroupRef);
						r = pushReturn(mb, r, newTmpVariable(mb, TYPE_bat));
						r = pushReturn(mb, r, newTmpVariable(mb, TYPE_bat));
						r = pushReturn(mb, r, newTmpVariable(mb, TYPE_bat));
						r = pushArgument(mb, r, getArg(old[i2], 0));


						/* create bat.mirror instruction */
// 						s = newInstruction(mb, ASSIGNsymbol);
// 						setModuleId(s, batRef);
// 						setFunctionId(s, mirrorRef);
// 						s = pushReturn(mb, s, newTmpVariable(mb, TYPE_bat));
// 						s = pushArgument(mb, s, getArg(r, 0));

						/* create algebra.leftjoin instruction */
						t = newInstruction(mb, ASSIGNsymbol);
						setModuleId(t, algebraRef);
						setFunctionId(t, leftfetchjoinRef);
						t = pushReturn(mb, t, newTmpVariable(mb, TYPE_bat)); /* t4 */
						t = pushArgument(mb, t, getArg(r, 1));
						t = pushArgument(mb, t, getArg(old[i2], 0));
						
						/* create dvf.fetch_file_ids_and_locations instruction */
						ffiali = newInstruction(mb, ASSIGNsymbol);
						setModuleId(ffiali, dvfRef);
						setFunctionId(ffiali, fetchfileidsandlocationsRef);
						ffiali = pushReturn(mb, ffiali, newTmpVariable(mb, TYPE_bat)); /* X_21 */
						ffiali = pushReturn(mb, ffiali, newTmpVariable(mb, TYPE_bat)); /* X_22 */
						ffiali = pushArgument(mb, ffiali, var_sql_mvc);
						ffiali = pushStr(mb, ffiali, schema_name); /* schema_name */
						ffiali = pushArgument(mb, ffiali, getArg(t, 0)); /* t4 */

						/* create dvf.plan_modifier instruction */
						q = newInstruction(mb, ASSIGNsymbol);
						setModuleId(q, dvfRef);
						setFunctionId(q, planmodifierRef);
						q = pushReturn(mb, q, newTmpVariable(mb, TYPE_void));
						q = pushArgument(mb, q, getArg(p, 2)); /* schema_name */
						q = pushArgument(mb, q, getArg(ffiali, 0)); /* X_21 */
						q = pushArgument(mb, q, getArg(ffiali, 1)); /* X_22 */
						if(mode == 2)
							q = pushInt(mb, q, 0);
						else
							q = pushInt(mb, q, 1);

						/* create barrier instruction */
						bar = newInstruction(mb, ASSIGNsymbol);
						bar->barrier = BARRIERsymbol;
						bar = pushReturn(mb, bar, newTmpVariable(mb, TYPE_any));
						bar = pushStr(mb, bar, barrier_to_defend_against_dataflow_identifier);
						
						
						/* insert the new instructions in pc i2+1 */
						insertInstruction(mb, bar, i2+1);
						barrier_to_defend_against_dataflow = 1;
						insertInstruction(mb, q, i2+1);
						
						insertInstruction(mb, ffiali, i2+1);
						
						insertInstruction(mb, t, i2+1);
// 						insertInstruction(mb, s, i2+1);
						insertInstruction(mb, r, i2+1);

						actions += 5;
						state = 3;
					}
					
					break;
				case 3:
					/* injection is done.*/
					last_bind_return_var_id = getArg(p, 0);
					
					if(mode == 1)
					{
						/* Now it is time to replace return bats of sql.binds for data table with appended bats */
						which_column = get_column_num(schema_name, getVarConstant(mb, getArg(p, 3)).val.sval, 
										getVarConstant(mb, getArg(p, 4)).val.sval);
						if(which_column < 0)
						{
							printf("dvf.get_column_num is not defined yet for schema: %s and table: %s and column: %s.",
							schema_name, getVarConstant(mb, getArg(p, 3)).val.sval, getVarConstant(mb, getArg(p, 4)).val.sval);
							return -1;
						}
						
						r = newInstruction(mb, ASSIGNsymbol);
						r = pushReturn(mb, r, getArg(p, 0));
						r = pushArgument(mb, r, getArg(ps_iter[which_column+NUM_RET_MOUNT], 0));
						
						insertInstruction(mb, r, i+1);
						removeInstruction(mb, p);
						
						actions += 2;
					}
					
					
			}
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == tidRef &&
			p->argc == 4 &&
			p->retc == 1 &&
			strcmp(getVarConstant(mb, getArg(p, 2)).val.sval, getVarConstant(mb, getArg(old[i1], 2)).val.sval) == 0 &&
			strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, data_table_identifier)  == 0)
		{
			last_data_tid_return_var_id = getArg(p, 0);
			removeInstruction(mb, p);
			i--;
			actions++;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == bindRef &&
			p->argc == 7 &&
			p->retc == 2 &&
			strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, getVarConstant(mb, getArg(old[i1], 2)).val.sval) == 0 &&
			strcmp(getVarConstant(mb, getArg(p, 4)).val.sval, data_table_identifier) == 0 &&
			getVarConstant(mb, getArg(p, 6)).val.ival == 2)
		{
			last_update_bind_second_return_var_id = getArg(p, 1);
			removeInstruction(mb, p);
			i--;
			actions++;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == bindRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			strcmp(getVarConstant(mb, getArg(p, 2)).val.sval, getVarConstant(mb, getArg(old[i1], 2)).val.sval) == 0 &&
			strcmp(getVarConstant(mb, getArg(p, 3)).val.sval, data_table_identifier) == 0 &&
			getVarConstant(mb, getArg(p, 5)).val.ival == 1)
		{
			last_insert_bind_return_var_id = getArg(p, 0);
			removeInstruction(mb, p);
			i--;
			actions++;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == algebraRef &&
			getFunctionId(p) == subselectRef &&
			p->argc == 8 &&
			p->retc == 1 &&
			getArg(p, 1) == last_bind_return_var_id)
		{
			last_subselect_return_var_id = getArg(p, 0);
			
			r = newInstruction(mb, ASSIGNsymbol);
			setModuleId(r, algebraRef);
			setFunctionId(r, subselectRef);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, getArg(p, 1));
			r = pushArgument(mb, r, getArg(p, 3));
			r = pushArgument(mb, r, getArg(p, 4));
			r = pushArgument(mb, r, getArg(p, 5));
			r = pushArgument(mb, r, getArg(p, 6));
			r = pushArgument(mb, r, getArg(p, 7));
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == algebraRef &&
			getFunctionId(p) == thetasubselectRef &&
			p->argc == 5 &&
			p->retc == 1 &&
			getArg(p, 1) == last_bind_return_var_id)
		{
			last_subselect_return_var_id = getArg(p, 0);
			
			r = newInstruction(mb, ASSIGNsymbol);
			setModuleId(r, algebraRef);
			setFunctionId(r, thetasubselectRef);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, getArg(p, 1));
			r = pushArgument(mb, r, getArg(p, 3));
			r = pushArgument(mb, r, getArg(p, 4));
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == algebraRef &&
			((getFunctionId(p) == subselectRef && (p->argc == 7 || p->argc == 8)) || 
			(getFunctionId(p) == thetasubselectRef && (p->argc == 5 || p->argc == 4))) &&
			p->retc == 1 &&
			(getArg(p, 1) == last_update_bind_second_return_var_id || getArg(p, 1) == last_insert_bind_return_var_id))
		{
			removeInstruction(mb, p);
			i--;
			actions++;
		}
		else if((state == 2 || state == 3) &&
			getModuleId(p) == algebraRef &&
			getFunctionId(p) == subselectRef &&
			p->argc == 8 &&
			p->retc == 1 &&
			getArg(p, 2) == last_data_tid_return_var_id)
		{
			last_subselect_return_var_id = getArg(p, 0);
			
			r = newInstruction(mb, ASSIGNsymbol);
			setModuleId(r, algebraRef);
			setFunctionId(r, subselectRef);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, getArg(p, 1));
			r = pushArgument(mb, r, getArg(p, 3));
			r = pushArgument(mb, r, getArg(p, 4));
			r = pushArgument(mb, r, getArg(p, 5));
			r = pushArgument(mb, r, getArg(p, 6));
			r = pushArgument(mb, r, getArg(p, 7));
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == projectdeltaRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			last_bind_return_var_id == getArg(p, 2))
		{
			r = newInstruction(mb, ASSIGNsymbol);
			
			if(getArg(p, 1) == last_data_tid_return_var_id)
			{
				r = pushReturn(mb, r, getArg(p, 0));
				r = pushArgument(mb, r, last_bind_return_var_id);
			}
			else
			{
				setModuleId(r, algebraRef);
				setFunctionId(r, leftfetchjoinRef);
				r = pushReturn(mb, r, getArg(p, 0));
				r = pushArgument(mb, r, getArg(p, 1));
				r = pushArgument(mb, r, last_bind_return_var_id);
			}
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == projectdeltaRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			last_subdelta_return_var_id == getArg(p, 1))
		{
			r = newInstruction(mb, ASSIGNsymbol);
			setModuleId(r, algebraRef);
			setFunctionId(r, leftfetchjoinRef);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, getArg(p, 1));
			r = pushArgument(mb, r, getArg(p, 2));
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 3) &&
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == deltaRef &&
			p->argc == 5 &&
			p->retc == 1 &&
			last_bind_return_var_id == getArg(p, 1))
		{
			r = newInstruction(mb, ASSIGNsymbol);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, last_bind_return_var_id);
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if((state == 3) && 
			getModuleId(p) == sqlRef &&
			getFunctionId(p) == subdeltaRef &&
			p->argc == 6 &&
			p->retc == 1 &&
			last_subselect_return_var_id == getArg(p, 1))
		{
			last_subdelta_return_var_id = getArg(p, 0);
			
			r = newInstruction(mb, ASSIGNsymbol);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, last_subselect_return_var_id);
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			
			actions += 2;
		}
		else if(getModuleId(p) == algebraRef &&
			getFunctionId(p) == leftfetchjoinRef &&
			p->argc == 3 &&
			p->retc == 1 &&
			last_data_tid_return_var_id == getArg(p, 1) &&
			getModuleId(old[i-1]) == sqlRef &&
			getFunctionId(old[i-1]) == deltaRef &&
			old[i-1]->argc == 5 &&
			old[i-1]->retc == 1 &&
			last_bind_return_var_id == getArg(old[i-1], 1) &&
			getArg(p, 2) == getArg(old[i-1], 0))
		{
			r = newInstruction(mb, ASSIGNsymbol);
			r = pushReturn(mb, r, getArg(p, 0));
			r = pushArgument(mb, r, last_bind_return_var_id);
			
			insertInstruction(mb, r, i+1);
			removeInstruction(mb, p);
			removeInstruction(mb, old[i-1]);
			
			actions += 3;
		}
		
		/* put the exit instruction of the dataflow defender barrier */
		else if(p->token == ENDsymbol && barrier_to_defend_against_dataflow)
		{
			/* create exit instruction */
			e = newInstruction(mb, ASSIGNsymbol);
			e->barrier = EXITsymbol;
			e = pushReturn(mb, e, getArg(bar, 0)); /* o */
			insertInstruction(mb, e, i);
			i++;
			barrier_to_defend_against_dataflow = 0;
		}
			
		
	}

	return actions;

}

str OPTdvfIterative(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p)
{
	str msg= MAL_SUCCEED;
	lng t,clk= GDKusec();
	int actions = 0;
	
	optimizerInit();
	if( p )
		removeInstruction(mb, p);

	if( mb->errors ){
		/* when we have errors, we still want to see them */
		addtoMalBlkHistory(mb,"dvf");
		return MAL_SUCCEED;
	}
	actions= OPTdvfImplementation(cntxt, mb,stk,p, 1);
	msg= optimizerCheck(cntxt, mb, "optimizer.DVframework", actions, t=(GDKusec() - clk),OPT_CHECK_ALL);
	OPTDEBUGdvf {
		mnstr_printf(cntxt->fdout,"=FINISHED opt_dvf %d\n",actions);
		printFunction(cntxt->fdout,mb,0,LIST_MAL_STMT | LIST_MAPI);
	}
	DEBUGoptimizers
	mnstr_printf(cntxt->fdout,"#opt_dvf: " LLFMT " ms\n",t);
	QOTupdateStatistics("dvf",actions,t);
	addtoMalBlkHistory(mb,"dvf");
	return msg;
}

str OPTdvfSemiparallel(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p)
{
	str msg= MAL_SUCCEED;
	lng t,clk= GDKusec();
	int actions = 0;
	
	optimizerInit();
	if( p )
		removeInstruction(mb, p);
	
	if( mb->errors ){
		/* when we have errors, we still want to see them */
		addtoMalBlkHistory(mb,"dvf");
		return MAL_SUCCEED;
	}
	actions= OPTdvfImplementation(cntxt, mb,stk,p, 2);
	msg= optimizerCheck(cntxt, mb, "optimizer.DVframework", actions, t=(GDKusec() - clk),OPT_CHECK_ALL);
	OPTDEBUGdvf {
		mnstr_printf(cntxt->fdout,"=FINISHED opt_dvf %d\n",actions);
		printFunction(cntxt->fdout,mb,0,LIST_MAL_STMT | LIST_MAPI);
	}
	DEBUGoptimizers
	mnstr_printf(cntxt->fdout,"#opt_dvf: " LLFMT " ms\n",t);
	QOTupdateStatistics("dvf",actions,t);
	addtoMalBlkHistory(mb,"dvf");
	return msg;
}

str OPTdvfParallel(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr p)
{
	str msg= MAL_SUCCEED;
	lng t,clk= GDKusec();
	int actions = 0;
	
	optimizerInit();
	if( p )
		removeInstruction(mb, p);
	
	if( mb->errors ){
		/* when we have errors, we still want to see them */
		addtoMalBlkHistory(mb,"dvf");
		return MAL_SUCCEED;
	}
	actions= OPTdvfImplementation(cntxt, mb,stk,p, 3);
	msg= optimizerCheck(cntxt, mb, "optimizer.DVframework", actions, t=(GDKusec() - clk),OPT_CHECK_ALL);
	OPTDEBUGdvf {
		mnstr_printf(cntxt->fdout,"=FINISHED opt_dvf %d\n",actions);
		printFunction(cntxt->fdout,mb,0,LIST_MAL_STMT | LIST_MAPI);
	}
	DEBUGoptimizers
	mnstr_printf(cntxt->fdout,"#opt_dvf: " LLFMT " ms\n",t);
	QOTupdateStatistics("dvf",actions,t);
	addtoMalBlkHistory(mb,"dvf");
	return msg;
}
