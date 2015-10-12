/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result.column;

import java.util.Iterator;

/**
 *  Am abstract class for accessing, 
 *  materialised (Java-level) query result columns.
 *
 * @param <T> A primitive or String type
 * TODO: To be extended for more complex types.
 */
public abstract class Column<T> implements Iterable<T> {
	/**
	 * The size/length of the column.
	 */
	private int columnSize;
	/**
	 * An index with boolean flags if certain column value is {@code null} or not.
	 * We need this since we are returning primary-type array from C to Java,
	 * which cannot contain nulls. At the same time. MonetDB columns can contain
	 * null values.  
	 */
	private boolean[] nullIndex;

	public Column(int columnSize, boolean[] nullIndex) {
		this.columnSize = columnSize;
		this.nullIndex = nullIndex;
	}

	/** 
	 * Get a (non-primary-type) value at index of a column.
	 *  
	 * @param index Column index for the value
	 * @return Value, cloud be {@code null}
	 */
	public abstract T getVaule(int index);

	/**
	 * Get the size of a column.
	 * 
	 * @return Column size
	 */
	public int columnSize() {
		return columnSize;
	}

	/**
	 * Check if the value at that index is {@code null} or not.
	 * 
	 * @param index Column index for the value
	 * @return {@code True} if it is {@code null}, {@code False} otherwise. 
	 */
	public boolean isNullValue(int index) {
		return nullIndex[index];
	}

	@Override
	public Iterator<T> iterator() {
		return new Iterator<T>() {
			private int currentIndex = 0;

			@Override
			public boolean hasNext() {
				return (currentIndex < columnSize());
			}

			@Override
			public T next() {
				return getVaule(currentIndex++);
			}
		};
	}
}
