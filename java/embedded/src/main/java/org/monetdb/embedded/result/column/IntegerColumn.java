/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result.column;

/** 
 * An {@code Integer} column.
 *
 */
public class IntegerColumn extends Column<Integer> {
	private int[] values;

	public IntegerColumn(int[] values, int columnSize, boolean[] nullIndex) {
		super(columnSize, nullIndex);
		this.values = values;
	}

	@Override
	public Integer getValue(int index) {
		if (isNullValue(index) || index < 0 || index >= columnSize()) {
			return null;
		}
		return Integer.valueOf(values[index]);
	}
}
