/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded.test;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Iterator;
import java.util.Properties;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;
import org.monetdb.embedded.result.column.Column;

import nl.cwi.monetdb.jdbc.MonetDBEmbeddedConnection;

public class EmbeddedTest {
	static File datbaseDirectory;
	static MonetDBEmbedded db;

	static Integer[][] testValues = {{Integer.valueOf(0), Integer.valueOf(1), Integer.valueOf(2), Integer.valueOf(3)}, 
			{Integer.valueOf(10), Integer.valueOf(20), Integer.valueOf(30), null}};

	static Object[] numbericTypeTestValues = new Object[]{
			Byte.valueOf((byte)12),
			Short.valueOf((short)23),
			Integer.valueOf(34),
			Long.valueOf(45l),
			Float.valueOf(5.6f),
			Double.valueOf(6.7)
	};

	static Object[] charTypeTestValues = new Object[]{
			"a string"
	};

	static Object[] booleanTypeTestValues = new Object[]{
			Boolean.valueOf(true),
			Boolean.valueOf(false)
	};

	@BeforeClass
	public static void createTestDB() throws IOException, SQLException {
		final Path directoryPath = Files.createTempDirectory("monetdbtest");
		datbaseDirectory = directoryPath.toFile();

		db = new MonetDBEmbedded(datbaseDirectory);
		//	    TimeUnit.SECONDS.sleep(15);
		db.start();

		db.query("CREATE TABLE test (id integer, val integer);");
		db.query("INSERT INTO test VALUES (" + testValues[0][0] + ", " + testValues[1][0] + "), (" + testValues[0][1] + ", " + testValues[1][1] + 
				"), (" + testValues[0][2] + ", " + testValues[1][2] + "), (" + testValues[0][3] + ", " + testValues[1][3] + ");");

		db.query("CREATE TABLE numeric_types_test (fbyte tinyint, fshort smallint, fint integer, flong bigint, freal real, fdouble double);");
		db.query("INSERT INTO numeric_types_test VALUES (" + numbericTypeTestValues[0] + ", " + numbericTypeTestValues[1] + ", " + numbericTypeTestValues[2] + ", " 
				+ numbericTypeTestValues[3] + ", " + numbericTypeTestValues[4] + ", " + numbericTypeTestValues[5] + ");");
		db.query("INSERT INTO numeric_types_test VALUES (null, null, null, null, null, null);");

		db.query("CREATE TABLE char_types_test (fstring string, fvarchar varchar(10));");
		db.query("INSERT INTO char_types_test VALUES ('" + charTypeTestValues[0] + "', '" + charTypeTestValues[0] + "');");
		db.query("INSERT INTO char_types_test VALUES (null, null);");

		db.query("CREATE TABLE boolean_types_test (fboolean boolean);");
		db.query("INSERT INTO boolean_types_test VALUES (" + booleanTypeTestValues[0] + ");");
		db.query("INSERT INTO boolean_types_test VALUES (" + booleanTypeTestValues[1] + ");");
		db.query("INSERT INTO boolean_types_test VALUES (null);");
	}

	@Test
	public void rowIteratorTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {

			int i = 0;
			Iterator<?> iterator = result.getColumn(1).iterator();
			while (iterator.hasNext()) {
				assertEquals(testValues[1][i++], iterator.next());
			}
		}
	}

	@Test
	public void columnIteratorTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {

			int i = 0;
			Iterator<Column<?>> columnIterator = result.iterator();
			while (columnIterator.hasNext()) {
				int j = 0;
				Iterator<?> rowIterator = columnIterator.next().iterator();
				while (rowIterator.hasNext()) {
					assertEquals(testValues[i][j++], rowIterator.next());
				}
				i++;
			}
		}
	}

	@Test
	public void integerWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getValue(1));
			assertEquals(null, result.getColumn(1).getValue(3));
		}
	}

	@Test
	public void getColumnByNameTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM numeric_types_test;")) {
			assertEquals(6, result.getNumberOfColumns());

			// byte
			assertEquals(numbericTypeTestValues[2], result.getColumn("fint").getValue(0));
			assertEquals(null, result.getColumn("fint").getValue(1));
		}
	}

	@Test
	public void numericTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM numeric_types_test;")) {
			assertEquals(6, result.getNumberOfColumns());

			// byte
			assertEquals(numbericTypeTestValues[0], result.getColumn(0).getValue(0));
			assertEquals(null, result.getColumn(0).getValue(1));
			// short
			assertEquals(numbericTypeTestValues[1], result.getColumn(1).getValue(0));
			assertEquals(null, result.getColumn(1).getValue(1));
			// int
			assertEquals(numbericTypeTestValues[2], result.getColumn(2).getValue(0));
			assertEquals(null, result.getColumn(2).getValue(1));
			// long
			assertEquals(numbericTypeTestValues[3], result.getColumn(3).getValue(0));
			assertEquals(null, result.getColumn(3).getValue(1));
			// float
			assertEquals(numbericTypeTestValues[4], result.getColumn(4).getValue(0));
			assertEquals(null, result.getColumn(4).getValue(1));
			// double
			assertEquals(numbericTypeTestValues[5], result.getColumn(5).getValue(0));
			assertEquals(null, result.getColumn(5).getValue(1));
		}
	}

	@Test
	public void charTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM char_types_test;")) {
			assertEquals(2, result.getNumberOfColumns());

			assertEquals(charTypeTestValues[0], result.getColumn(0).getValue(0));
			assertEquals(charTypeTestValues[0], result.getColumn(1).getValue(0));
			assertEquals("", result.getColumn(0).getValue(1));
			assertEquals("", result.getColumn(1).getValue(1));
		}
	}

	@Test
	public void booleanTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM boolean_types_test;")) {
			assertEquals(1, result.getNumberOfColumns());

			assertEquals(booleanTypeTestValues[0], result.getColumn(0).getValue(0));
			assertEquals(booleanTypeTestValues[1], result.getColumn(0).getValue(1));
			assertEquals(null, result.getColumn(0).getValue(2));
		}
	}

	@Test
	public void twoQueriesTest() throws SQLException {
		EmbeddedQueryResult result1 = db.query("SELECT * FROM test WHERE id > 1;");
		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(testValues[1][2], result1.getColumn(1).getValue(0));
		assertEquals(testValues[1][3], result1.getColumn(1).getValue(1));

		EmbeddedQueryResult result2 = db.query("SELECT * FROM test WHERE id < 1;");
		assertEquals(1, result2.getColumn(1).columnSize());
		assertEquals(testValues[1][0], result2.getColumn(1).getValue(0));

		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(testValues[1][2], result1.getColumn(1).getValue(0));
		assertEquals(testValues[1][3], result1.getColumn(1).getValue(1));
	}

	@Test
	public void twoQueriesWithManualCleanupTest() throws SQLException, IOException {
		EmbeddedQueryResult result1 = db.query("SELECT * FROM test WHERE id > 1;");
		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(testValues[1][2], result1.getColumn(1).getValue(0));
		assertEquals(testValues[1][3], result1.getColumn(1).getValue(1));

		EmbeddedQueryResult result2 = db.query("SELECT * FROM test WHERE id < 1;");
		assertEquals(testValues[1][0], result2.getColumn(1).getValue(0));

		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(testValues[1][2], result1.getColumn(1).getValue(0));
		assertEquals(testValues[1][3], result1.getColumn(1).getValue(1));

		result1.close();
		result2.close();
	}

	@Test
	public void manualCleanupTest() throws IOException, SQLException {
		@SuppressWarnings("resource")
		EmbeddedQueryResult result = db.query("SELECT * FROM test;");
		assertEquals(4, result.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(20), result.getColumn(1).getValue(1));
		assertEquals(null, result.getColumn(1).getValue(3));

		result.close();
	}

	@Test
	public void dobuleManualCleanupTest() throws IOException, SQLException {
		@SuppressWarnings("resource")
		EmbeddedQueryResult result = db.query("SELECT * FROM test;");
		assertEquals(4, result.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(20), result.getColumn(1).getValue(1));
		assertEquals(null, result.getColumn(1).getValue(3));

		result.close();
		result.close();
	}

	@Test
	public void resultAccessAfterClose() throws IOException, SQLException {
		@SuppressWarnings("resource")
		EmbeddedQueryResult result = db.query("SELECT * FROM test;");
		assertEquals(4, result.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(20), result.getColumn(1).getValue(1));
		assertEquals(null, result.getColumn(1).getValue(3));

		result.close();

		// The result of any column get should be null
		assertEquals(null, result.getColumn(1));
	}

	@Test(expected=SQLException.class)
	public void captureQueryErrorTest() throws SQLException {
		db.query("SELECT FROM test;");
	}

	@Test
	public void newObjectWithSameDatabaseDirectoryTest() throws IOException, SQLException {
		MonetDBEmbedded sameDB = new MonetDBEmbedded(datbaseDirectory, false);
		// This is technically a no-op
		sameDB.start();

		try (EmbeddedQueryResult result = sameDB.query("SELECT * FROM test;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getValue(1));
			assertEquals(null, result.getColumn(1).getValue(3));
		}
	}

	@Test(expected=IOException.class)
	public void newDatabaseTest() throws IOException {
		final Path tempDirectoryPath = Files.createTempDirectory("monetdbtest_new");
		final File newDirectory = tempDirectoryPath.toFile();

		MonetDBEmbedded newDB = new MonetDBEmbedded(newDirectory);
		newDB.start();
	}

	@Test
	public void simpleResultSetJDBCTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(10), result.getColumn(1).getValue(0));
			assertEquals(10, result.getJDBCResultSet().getInt(1));
		}
	}

	@Test
	public void simpleConnectionAndCreateStatementAndResultSetJDBCTest() throws SQLException {
		Properties props = new Properties();
		props.put("database", datbaseDirectory.toString());

		try (Connection connection = new MonetDBEmbeddedConnection(props)) {
			try (Statement statement = connection.createStatement()) {
				statement.execute("SELECT * FROM test;");
				try (ResultSet result = statement.getResultSet()) {
					assertEquals(10, result.getInt(1));
				}
			}
		}
	}

	@AfterClass
	public static void cleanup() throws SQLException, IOException {
		db.query("DROP TABLE test");
		db.query("DROP TABLE numeric_types_test");
		db.query("DROP TABLE char_types_test");
		db.query("DROP TABLE boolean_types_test");
		db.close();
	}
}
