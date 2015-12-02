package nl.cwi.monetdb.jdbc;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;

import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

public class MonetDBEmbeddedStatement extends MonetStatement {
	private final MonetDBEmbedded database;
	private EmbeddedQueryResult resultSet;
	
	public MonetDBEmbeddedStatement(MonetDBEmbedded database)
			throws SQLException, IllegalArgumentException {
		super(null, ResultSet.CONCUR_READ_ONLY, ResultSet.TYPE_SCROLL_INSENSITIVE, ResultSet.HOLD_CURSORS_OVER_COMMIT);
		this.database = database;
	}
	
	@Override
	protected boolean internalExecute(String sql) throws SQLException {
		resultSet = database.query(sql);
		return false;
	}
	
	@Override
	public ResultSet getResultSet() throws SQLException{
		return resultSet.getJDBCResultSet();
	}
	
	@Override
	public int[] executeBatch() throws SQLException {
		throw new SQLFeatureNotSupportedException("Method executeBatch not implemented in the embedded MonetDB");
	}
	
	@Override
	public Connection getConnection() {
		return null;
	}
}
