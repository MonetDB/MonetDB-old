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
 */

package nl.cwi.monetdb.jdbc;

import java.sql.SQLException;
import java.sql.Savepoint;

/**
 * Support for two-stage commit. Pre-commit first writes the transaction only to the write-ahead log.
 * The following persist call writes the data to the persistent store.
 * Since the two-stage commit relies on {@link nl.cwi.monetdb.jdbc.MonetSavepoint Savepoints},
 * make sure {@link nl.cwi.monetdb.jdbc.MonetConnection#setAutoCommit(boolean autoCommit) autocommit} is disabled.
 * 
 * @author dnedev <Dimitar.Nedev@MonetDBSolutions.com>
 * 
 */
public class MonetTwoStageCommit {
	private final MonetConnection connection;
	
	MonetTwoStageCommit(MonetConnection connection) {
		this.connection = connection;
	}
	
	public Savepoint preCommit(String name) throws SQLException {
		Savepoint savepoint;
		
		if (connection.getAutoCommit()) {
			throw new SQLException("Cannot execute preCommit - autocommit enabled", "3B000");
		}
		savepoint = connection.setSavepoint(name);
		connection.sendIndependentCommand("pre_commit");
		
		return savepoint;
	}
	
	public void persistCommit(Savepoint savepoint) throws SQLException {
		connection.sendIndependentCommand("presist_commit");
		connection.releaseSavepoint(savepoint);
		return;
	}
	
	public void rollbackCommit(Savepoint savepoint) throws SQLException {
		connection.rollback(savepoint);
		return;
	}
}
