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
import java.sql.SQLException;
import java.util.Iterator;

import org.junit.BeforeClass;
import org.junit.Test;
import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

public class EmbeddedTest {
	static File datbaseDirectory;
	static MonetDBEmbedded db;

	static Integer[] testValues = {Integer.valueOf(10), Integer.valueOf(20), Integer.valueOf(30), null};

	static Object[] numbericTypeTestValues = new Object[]{
			Byte.valueOf((byte)12),
			Short.valueOf((short)23),
			Integer.valueOf(34),
			Long.valueOf(45l),
			Float.valueOf(5.6f),
			Double.valueOf(6.7),
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

		db = new MonetDBEmbedded(datbaseDirectory, false);
		db.run();

		db.query("CREATE TABLE test (id integer, val integer);");
		db.query("INSERT INTO test VALUES (0, " + testValues[0] + "), (1, " + testValues[1] + "), (2, " + testValues[2] + "), (3, " + testValues[3] + ");");

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
	public void restartExistingDatabaseTest() throws IOException, SQLException {
		MonetDBEmbedded restartedDB = new MonetDBEmbedded(datbaseDirectory, false);
		restartedDB.run();

		try (EmbeddedQueryResult result = restartedDB.query("SELECT * FROM test;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}

	@Test
	public void iteratorTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {

			int i = 0;
			Iterator<?> iterator = result.getColumn(1).iterator();
			while (iterator.hasNext()) {
				assertEquals(testValues[i++], iterator.next());
			}
		}
	}

	@Test
	public void integerWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}

	@Test
	public void getColumnByNameTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM numeric_types_test;")) {
			assertEquals(6, result.getNumberOfColumns());

			// byte
			assertEquals(numbericTypeTestValues[2], result.getColumn("fint").getVaule(0));
			assertEquals(null, result.getColumn("fint").getVaule(1));
		}
	}

	@Test
	public void numericTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM numeric_types_test;")) {
			assertEquals(6, result.getNumberOfColumns());

			// byte
			assertEquals(numbericTypeTestValues[0], result.getColumn(0).getVaule(0));
			assertEquals(null, result.getColumn(0).getVaule(1));
			// short
			assertEquals(numbericTypeTestValues[1], result.getColumn(1).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));
			// int
			assertEquals(numbericTypeTestValues[2], result.getColumn(2).getVaule(0));
			assertEquals(null, result.getColumn(2).getVaule(1));
			// long
			assertEquals(numbericTypeTestValues[3], result.getColumn(3).getVaule(0));
			assertEquals(null, result.getColumn(3).getVaule(1));
			// float
			assertEquals(numbericTypeTestValues[4], result.getColumn(4).getVaule(0));
			assertEquals(null, result.getColumn(4).getVaule(1));
			// double
			assertEquals(numbericTypeTestValues[5], result.getColumn(5).getVaule(0));
			assertEquals(null, result.getColumn(5).getVaule(1));
		}
	}

	@Test
	public void charTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM char_types_test;")) {
			assertEquals(2, result.getNumberOfColumns());

			assertEquals(charTypeTestValues[0], result.getColumn(0).getVaule(0));
			assertEquals(charTypeTestValues[0], result.getColumn(1).getVaule(0));
			assertEquals("", result.getColumn(0).getVaule(1));
			assertEquals("", result.getColumn(1).getVaule(1));
		}
	}

	@Test
	public void booleanTypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM boolean_types_test;")) {
			assertEquals(1, result.getNumberOfColumns());

			assertEquals(booleanTypeTestValues[0], result.getColumn(0).getVaule(0));
			assertEquals(booleanTypeTestValues[1], result.getColumn(0).getVaule(1));
			assertEquals(null, result.getColumn(0).getVaule(2));
		}
	}

	@Test
	public void twoQueries() throws SQLException {
		EmbeddedQueryResult result1 = db.query("SELECT * FROM test WHERE id > 1;");
		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(30), result1.getColumn(1).getVaule(0));
		assertEquals(null, result1.getColumn(1).getVaule(1));

		EmbeddedQueryResult result2 = db.query("SELECT * FROM test WHERE id < 1;");
		assertEquals(1, result2.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(10), result2.getColumn(1).getVaule(0));

		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(30), result1.getColumn(1).getVaule(0));
		assertEquals(null, result1.getColumn(1).getVaule(1));
	}

	@Test
	public void manualCleanupTest() throws IOException, SQLException {
		@SuppressWarnings("resource")
		EmbeddedQueryResult result = db.query("SELECT * FROM test;");
		assertEquals(4, result.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
		assertEquals(null, result.getColumn(1).getVaule(3));

		result.close();
	}

	@Test(expected=SQLException.class)
	public void captureQueryErrorTest() throws SQLException {
		db.query("SELECT FROM test;");
	}

	@Test
	public void newDatabaseTest() throws IOException, SQLException {
		final Path tempDirectoryPath = Files.createTempDirectory("monetdbtest_new");
		final File newDirectory = tempDirectoryPath.toFile();

		MonetDBEmbedded newDB = new MonetDBEmbedded(newDirectory, false);
		newDB.run();

		newDB.query("CREATE TABLE test (id integer, val integer);");
		newDB.query("INSERT INTO test VALUES (1, 10), (2, 20), (3, 30), (4, null);");

		try (EmbeddedQueryResult result = newDB.query("SELECT * FROM world;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}
}
