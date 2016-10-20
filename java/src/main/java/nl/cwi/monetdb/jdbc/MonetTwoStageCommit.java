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
 * Copyright August 2008-2014 MonetDB B.V.
 * All Rights Reserved.
 *
 * Acknowledgement
 * ===============
 * 
 * The research leading to this code has been partially funded by the European
 * Commission under FP7 programme project #611068.
 */

package nl.cwi.monetdb.jdbc;

import java.sql.SQLException;
import java.sql.Savepoint;

/**
 * Support for two-stage commit. Pre-commit first writes the transaction only to the write-ahead log.
 * The following persist call writes the data to the persistent store.
 * Since the two-stage commit relies on {@link nl.cwi.monetdb.jdbc.MonetSavepoint savepoints},
 * make sure {@link nl.cwi.monetdb.jdbc.MonetConnection#setAutoCommit(boolean autoCommit) autocommit} is disabled.
 * 
 * @author dnedev <Dimitar.Nedev@MonetDBSolutions.com>
 * 
 */
public class MonetTwoStageCommit {
	/**
	 * Pre-commits the transaction, storing it in the write-ahead log and
	 * creates a {@link nl.cwi.monetdb.jdbc.MonetSavepoint savepoint}.
	 * This method should be used only when auto-commit mode has been disabled.
	 * 
	 * @param name A name for the savepoint
	 * @param connection The JDBC connection to the database
	 * @return The newly created savepoint object
	 * @throws SQLException
	 */
	public Savepoint preCommit(String name, MonetConnection connection) throws SQLException {
		Savepoint savepoint;
		
		if (connection.getAutoCommit()) {
			throw new SQLException("Cannot execute preCommit - autocommit enabled", "3B000");
		}
		savepoint = connection.setSavepoint(name);
		connection.sendIndependentCommand("precommit()");
		
		return savepoint;
	}
	
	/**
	 * Persists a given {@link nl.cwi.monetdb.jdbc.MonetSavepoint savepoint},
	 * writing transaction to the persistent database store.
	 * Following that the savepoint is released.
	 * 
	 * @param savepoint The savepoint which to persists
	 * @param connection The JDBC connection to the database
	 * @throws SQLException
	 */
	public void persistCommit(Savepoint savepoint, MonetConnection connection) throws SQLException {
		connection.sendIndependentCommand("presistcommit()");
		connection.releaseSavepoint(savepoint);
		return;
	}
	
	/**
	 * Rolls-back a given savepoint.
	 * 
	 * @param savepoint The savepoint which to rollback
	 * @param connection The JDBC connection to the database
	 * @throws SQLException
	 */
	public void rollbackCommit(Savepoint savepoint, MonetConnection connection) throws SQLException {
		// no need to do anything else, rollback bring the store back to the previous commited transaction
		connection.rollback(savepoint);
		return;
	}
}
