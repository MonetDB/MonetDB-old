/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result.column;

/** 
 * A {@code Short} column.
 *
 */
public class ShortColumn extends Column<Short> {
	private short[] values;

	public ShortColumn(short[] values, int columnSize, boolean[] nullIndex) {
		super(columnSize, nullIndex);
		this.values = values;
	}

	@Override
	public Short getValue(int index) {
		if (isNullValue(index) || index < 0 || index >= columnSize()) {
			return null;
		}
		return Short.valueOf(values[index]);
	}
}
