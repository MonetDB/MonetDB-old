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
import java.util.Iterator;

import org.monetdb.embedded.result.column.Column;

import nl.cwi.monetdb.jdbc.MonetDBResultSet;
import nl.cwi.monetdb.jdbc.MonetResultSet;

/**
 * Embedded MonetDB query result.
 * The query result columns are not eagerly copied from the native code to Java.
 * Instead, they are kept around at MonetDB native C-level, materialised in Java 
 * on demand and freed on {@code close()}.
 *
 */
public class EmbeddedQueryResult implements Closeable, Iterable<Column<?>> {
	/**
	 * The names of the columns in the query result.
	 */
	private String[] columnNames;
	/**
	 * The types of the columns in the query result.
	 */
	private String[] columnTypes;
	/**
	 * The number of columns in the query result.
	 */
	private int numberOfColumns;
	/**
	 * The number of rows in the query result.
	 */
	private int numberOfRows;
	/**
	 * Pointer to the native result set.
	 * We need to keep it around for getting columns.
	 * The native result set is kept until the {@link close()} is called.
	 */
	private long resultPointer;

	public EmbeddedQueryResult(String[] columnNames, String[] columnTypes, int numberOfColumns, int numberOfRows, long resultPointer) {
		this.columnNames = columnNames;
		this.columnTypes = columnTypes;
		this.numberOfColumns = numberOfColumns;
		this.numberOfRows = numberOfRows;
		this.resultPointer = resultPointer;
	}

	/**
	 * Returns the number of columns in the result set.
	 * 
	 * @return Number of columns
	 */
	public int getNumberOfColumns() {
		return numberOfColumns;
	}

	/**
	 * Get a column from the result set by index.
	 * 
	 * @param index Column index (starting from 0)
	 * @return The column, {@code null} if index not in bounds
	 */
	public Column<?> getColumn(int index) {
		if (index >= numberOfColumns || index < 0) {
			return null;
		}
		return getColumnWrapper(resultPointer, index);
	}

	/**
	 * Get a column from the result set by name.
	 * 
	 * @param name Column name
	 * @return The column, {@code null} if not found
	 */
	public Column<?> getColumn(String name) {
		int index = 0;
		for (String columnName : columnNames) {
			if (name.equals(columnName)) {
				return getColumn(index);
			}
			index++;
		}
		return null;
	}

	/**
	 * A native C function that returns a {@code Column} object.
	 * 
	 * @param resultPointerWrapper Pointer to the C-level result structure
	 * @param index Column index (starting from 0) 
	 * @return
	 */
	private native Column<?> getColumnWrapper(long resultPointerWrapper, int index);

	public MonetResultSet getJDBCResultSet() {
		return new MonetDBResultSet(columnNames, columnTypes, numberOfRows);
	}

	@Override
	public Iterator<Column<?>> iterator() {
		return new Iterator<Column<?>>() {
			private int currentIndex = 0;

			@Override
			public boolean hasNext() {
				return (currentIndex < getNumberOfColumns());
			}

			@Override
			public Column<?> next() {
				try {
					return getColumn(currentIndex++);
				} catch (IndexOutOfBoundsException e) {
					// We can't throw it really, so just return null
					// XXX: Find a more elegant solution
				}
				return null;
			}
		};
	}

	@Override
	public void close() throws IOException {
		cleanupResult(resultPointer);
	}

	/** 
	 * Free the C-level result structure.
	 * 
	 * @param resultPointerWrapper Pointer to the C-level result structure 
	 */
	private native void cleanupResult(long resultPointerWrapper);
}
