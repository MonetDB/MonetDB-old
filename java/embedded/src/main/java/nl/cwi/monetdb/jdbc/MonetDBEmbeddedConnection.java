/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package nl.cwi.monetdb.jdbc;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;

import org.monetdb.embedded.MonetDBEmbedded;

/**
 * A JDBC connection for the embedded MonetDB.
 *
 */
public class MonetDBEmbeddedConnection extends MonetConnection {
	private final String databaseLocationString;
	private final MonetDBEmbedded database;

	public MonetDBEmbeddedConnection(Properties props) throws SQLException, IllegalArgumentException {
		super(props.getProperty("database"));
		this.databaseLocationString = props.getProperty("database");
		if (databaseLocationString == null || databaseLocationString.isEmpty()) {
			throw new IllegalArgumentException("Database location is not set.");
		}
		File databaseLocation = new File(databaseLocationString);
		database = new MonetDBEmbedded(databaseLocation);
		try {
			database.start();
		} catch (IOException e) {
			throw new SQLException(e);
		}
	}

	@Override
	public Statement createStatement(int resultSetType, int resultSetConcurrency, int resultSetHoldability) throws SQLException {
		Statement ret = new MonetDBEmbeddedStatement(this);
		statements.put(ret, null);
		return ret;
	}

	@Override
	public String getJDBCURL() {
		return "jdbc:monetdb://" + databaseLocationString;
	}

	public MonetDBEmbedded getDatabase() {
		return database;
	}
	
	@Override
	public void close() {
		try {
			database.close();
		} catch (IOException e) {
			// Do nothing. We can't throw it up
		}
	}
}
