/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package nl.cwi.monetdb.jdbc;

/**
 * This class serves only to expose the {@link nl.cwi.monetdb.jdbc.MonetResultSet} constructor
 * outside the {@code nl.cwi.monetdb.jdbc}.
 */
public class MonetDBResultSet extends MonetResultSet {
	public MonetDBResultSet(String[] columnNames, String[] columnTypes, int numberOfRows) {
		super(columnNames, columnTypes, numberOfRows);
	}
}
