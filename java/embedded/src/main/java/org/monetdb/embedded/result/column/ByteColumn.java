/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result.column;

/** 
 * An {@code Boolean} column.
 *
 */
public class ByteColumn extends Column<Byte> {
	private byte[] values;

	public ByteColumn(byte[] values, int columnSize, boolean[] nullIndex) {
		super(columnSize, nullIndex);
		this.values = values;
	}

	@Override
	public Byte getValue(int index) {
		if (isNullValue(index) || index < 0 || index >= columnSize()) {
			return null;
		}
		return Byte.valueOf(values[index]);
	}
}
