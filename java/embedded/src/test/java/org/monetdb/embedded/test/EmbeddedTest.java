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

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

public class EmbeddedTest {
	static File datbaseDirectory;
	static MonetDBEmbedded db;
	static Object[] typeValues = new Object[]{
			Short.valueOf((short)12),
			Integer.valueOf(23),
			Long.valueOf(34l),
			Float.valueOf(4.5f),
			Double.valueOf(5.6),
			"a string"
	};

	@BeforeClass
	public static void createTestDB() throws IOException, SQLException {
		final Path directoryPath = Files.createTempDirectory("monetdbtest");
		datbaseDirectory = directoryPath.toFile();

		db = new MonetDBEmbedded(datbaseDirectory);
		db.startup(false);

		db.query("CREATE TABLE world (id integer, val integer);");
		db.query("INSERT INTO world VALUES (1, 10), (2, 20), (3, 30), (4, null);");

		db.query("CREATE TABLE typestest (fshort smallint, fint integer,  flong bigint, ffloat float, fdouble double, fstring string);");
		db.query("INSERT INTO typestest VALUES (" + typeValues[0] + ", " + typeValues[1] + ", " + typeValues[2] + ", " 
				+ typeValues[3] + ", " + typeValues[4] + ", " + "'" + typeValues[5] + "'" + ");");
		db.query("INSERT INTO typestest VALUES (null, null, null, null, null, null);");
	}

	@Test
	public void restartExistingDatabaseTest() throws IOException, SQLException {
		MonetDBEmbedded restartedDB = new MonetDBEmbedded(datbaseDirectory);
		restartedDB.startup(false);

		try (EmbeddedQueryResult result = restartedDB.query("SELECT * FROM world;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}

	@Test
	public void IntegerWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM world;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}

	@Test
	public void TypesWithNullTest() throws IOException, SQLException {
		try (EmbeddedQueryResult result = db.query("SELECT * FROM typestest;")) {
			assertEquals(6, result.getNumberOfColumns());

			assertEquals(typeValues[0], result.getColumn(0).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));

			assertEquals(typeValues[1], result.getColumn(1).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));

			assertEquals(typeValues[2], result.getColumn(2).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));

			assertEquals(typeValues[3], result.getColumn(3).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));

			assertEquals(typeValues[4], result.getColumn(4).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));

			assertEquals(typeValues[5], result.getColumn(5).getVaule(0));
			assertEquals(null, result.getColumn(1).getVaule(1));
		}
	}

	@Test
	public void TwoQueries() throws IOException, SQLException {
		EmbeddedQueryResult result1 = db.query("SELECT * FROM world WHERE id > 2;");
		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(30), result1.getColumn(1).getVaule(0));
		assertEquals(null, result1.getColumn(1).getVaule(1));

		EmbeddedQueryResult result2 = db.query("SELECT * FROM world WHERE id < 2;");
		assertEquals(1, result2.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(10), result2.getColumn(1).getVaule(0));

		assertEquals(2, result1.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(30), result1.getColumn(1).getVaule(0));
		assertEquals(null, result1.getColumn(1).getVaule(1));
	}

	@Test
	public void manualCleanupTest() throws IOException, SQLException {
		@SuppressWarnings("resource")
		EmbeddedQueryResult result = db.query("SELECT * FROM world;");
		assertEquals(4, result.getColumn(1).columnSize());
		assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
		assertEquals(null, result.getColumn(1).getVaule(3));

		result.close();
	}

	@Test(expected=SQLException.class)
	public void captureQueryErrorTest() throws SQLException {
		db.query("SELECT FROM world;");
	}

	@Test
	@Ignore
	public void newDatabaseTest() throws IOException, SQLException {
		final Path tempDirectoryPath = Files.createTempDirectory("monetdbtest_new");
		final File newDirectory = tempDirectoryPath.toFile();

		MonetDBEmbedded newDB = new MonetDBEmbedded(newDirectory);
		newDB.startup(false);

		newDB.query("CREATE TABLE world (id integer, val integer);");
		newDB.query("INSERT INTO world VALUES (1, 10), (2, 20), (3, 30), (4, null);");

		try (EmbeddedQueryResult result = newDB.query("SELECT * FROM world;")) {
			assertEquals(4, result.getColumn(1).columnSize());
			assertEquals(Integer.valueOf(20), result.getColumn(1).getVaule(1));
			assertEquals(null, result.getColumn(1).getVaule(3));
		}
	}
}
