/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result;

import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Array;
import java.sql.Blob;
import java.sql.SQLException;
import java.sql.Timestamp;

public class QueryResult implements Closeable {
	private Array[] columns;
	private String[] columnNames;
	private String[] columnTypes;

	/**
	 * Returns the number of columns in a result set.
	 * 
	 * @return Number of columns
	 */
	public int numberOfColumns() {
		return columns.length;
	}

	/**
	 * Get a column from the result set by index.
	 * 
	 * @param id Column index (starting from 0)
	 * @return The columns as an array, {@code null} if index not in bounds
	 * @throws SQLException
	 */
	public Array getColumn(int index) throws SQLException, IndexOutOfBoundsException {
		if (index < 0 || index >= columns.length) {
			return null;
		}
		return getColumn(columns[index], columnTypes[index]);
	}

	/**
	 * Get a column from the result set by name.
	 * 
	 * @param name Column name
	 * @return The columns as an array, {@code null} if not found
	 * @throws SQLException
	 */
	public Array getColumn(String name) throws SQLException {
		int index = 0;
		for (String columnName : columnNames) {
			if (name.equals(columnName)) {
				return getColumn(columns[index], columnTypes[index]);
			}
			index++;
		}
		return null;
	}

	private static Array getColumn(Array column, String typeString) throws SQLException {
		Class<?> type;
		switch(typeString) {
		case "integer":
			type = Integer.TYPE;
			break;
		case "string":
			type = String.class;
			break;
		case "double":
			type = Double.TYPE;
			break;
		case "float":
			type = Float.TYPE;
			break;
		case "long":
			type = Long.TYPE;
			break;
		case "timestamp":
			type = Timestamp.class;
			break;
		case "binary":
			type = Blob.class;
			break;
		default:
			throw new SQLException("Unsupported column type " + typeString);
		}
		Array result = (Array) Array.newInstance(type, Array.getLength(column));
		result = column;
		return result;
	}

	@Override
	public void close() throws IOException {
		// TODO: Close the underlying C-level temp BATs, if any
	}
}
