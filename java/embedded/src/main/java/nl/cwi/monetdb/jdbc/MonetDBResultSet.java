/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package nl.cwi.monetdb.jdbc;

import java.io.IOException;
import java.sql.SQLException;

import org.monetdb.embedded.result.EmbeddedQueryResult;

/**
 * This class overrides {@link nl.cwi.monetdb.jdbc.MonetResultSet} to allow for handling embedded query results.
 * It needs to be in the {@code nl.cwi.monetdb.jdbc} module to access the constructor.
 */
public class MonetDBResultSet extends MonetResultSet {
	private final EmbeddedQueryResult resultSet;

	public MonetDBResultSet(EmbeddedQueryResult resultSet) {
		super(resultSet.getColumnNames(), resultSet.getColumnTypes(), resultSet.getNumberOfColumns());
		this.resultSet = resultSet;
	}

	@Override
	public String getString(int columnIndex) throws SQLException {
		String ret = resultSet.getColumn(columnIndex).getValue(curRow).toString();
//		String ret = tlp.values[columnIndex - 1];
		lastColumnRead = columnIndex - 1;
		return ret;
	}

	@Override
	public void close() {
		try {
			resultSet.close();
		} catch (IOException e) {
			// can't throw an exception now
		}
		super.close();
	}
}
