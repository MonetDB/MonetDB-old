/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.result;

import java.lang.reflect.Array;
import java.sql.Blob;
import java.sql.SQLException;
import java.sql.Timestamp;

public class Result {
	public String name;
	public String type;

//	public Array column() throws SQLException {
//		Class<?> typeClass;
//		switch(type) {
//		case "integer":
//			typeClass = Integer.TYPE;
//			break;
//		case "string":
//			typeClass = String.class;
//			break;
//		case "double":
//			typeClass = Double.TYPE;
//			break;
//		case "float":
//			typeClass = Float.TYPE;
//			break;
//		case "long":
//			typeClass = Long.TYPE;
//			break;
//		case "timestamp":
//			typeClass = Timestamp.class;
//			break;
//		case "binary":
//			typeClass = Blob.class;
//			break;
//		default:
//			throw new SQLException("Unsupported column type " + type);
//		}
//		
//		Array result = (Array) Array.newInstance(typeClass, Array.getLength(column));
//		result = column;
//		
//		return result;
//	}

//	public Integer[] getInteget() throws SQLException {
//		if (!"integer".equals(type)) {
//			throw new SQLException("Type mismatch, the column is of type " + type);
//		}
//		//		Integer[] result = Arrays.copyOf(column, Array.getLength(column), Integer[].class); 
//		Integer[] result = (Integer[]) Array.newInstance(Integer.TYPE, Array.getLength(column));
//		result  = column.get
//		return result;
//	}
}
