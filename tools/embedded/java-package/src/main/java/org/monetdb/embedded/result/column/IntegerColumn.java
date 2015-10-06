/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result.column;

/** 
 * An integer column.
 *
 */
public class IntegerColumn extends Column<Integer> {
	private int[] array;

	public IntegerColumn(int[] array, int columnSize, boolean[] nullIndex) {
		super(columnSize, nullIndex);
		this.array = array;
	}

	@Override
	public Integer getVaule(int index) {
		if (isNullValue(index) || index <= columnSize()) {
			return null;
		}
		return Integer.valueOf(array[index]);
	}
}
