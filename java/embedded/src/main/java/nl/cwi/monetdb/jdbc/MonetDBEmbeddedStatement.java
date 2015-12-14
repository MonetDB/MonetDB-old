/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package nl.cwi.monetdb.jdbc;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;

import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

/**
 * A JDBC statement for the embedded MonetDB.
 */
public class MonetDBEmbeddedStatement extends MonetStatement {
	private final MonetDBEmbedded database;
	private EmbeddedQueryResult resultSet;
	
	public MonetDBEmbeddedStatement(MonetDBEmbeddedConnection connection)
			throws SQLException, IllegalArgumentException {
		super(connection, ResultSet.CONCUR_READ_ONLY, ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.HOLD_CURSORS_OVER_COMMIT);
		this.database = connection.getDatabase();
	}
	
	/**
	 * Execute the SQL query.
	 * Need to call {@link nl.cwi.monetdb.jdbc.MonetDBEmbeddedStatement#getResultSet()}
	 * to get the result set back, like in JDBC.
	 * <br />
	 * <strong>Note</strong>:Currently supports only single queries.
	 * 
	 * @param sql Query string
	 */
	@Override
	protected boolean internalExecute(String sql) throws SQLException {
		resultSet = database.query(sql);
		return false;
	}
	
	/**
	 * Get the JDBC result set from previously executed query.
	 */
	@Override
	public ResultSet getResultSet() throws SQLException {
		if (resultSet != null ) {
			return resultSet.getJDBCResultSet();			
		}
		return null;
	}
	
	@Override
	public int[] executeBatch() throws SQLException {
		throw new SQLFeatureNotSupportedException("Method executeBatch not implemented in the embedded MonetDB");
	}
	
	@Override
	public Connection getConnection() {
		// There is no connection to return.
		// And we can't return the the embedded database (since it's not a Connection type) 
		return null;
	}
}
