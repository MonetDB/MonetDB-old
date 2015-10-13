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

import org.junit.Ignore;
import org.junit.Test;
import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

public class EmbeddedTest {

	@Ignore
	@Test
	public void newDatabaseTest() throws IOException, SQLException {
		final Path directoryPath = Files.createTempDirectory("monetdb");
		final File directory = directoryPath.toFile();

		MonetDBEmbedded db = new MonetDBEmbedded(directory);
		db.startup(false);

		db.query("CREATE TABLE world (id integer, val integer);");
		db.query("INSERT INTO world VALUES (1, 10), (2, 20), (3, 30);");

		EmbeddedQueryResult result = db.query("SELECT * FROM world;");
		assertEquals(3, result.getColumn(1).columnSize());

		result.close();
	}

	@Test
	public void existingDatabaseTest() throws IOException, SQLException {
		final File directory = new File("src" + File.separatorChar + "test" + 
				File.separatorChar + "resources" + File.separatorChar + "monetdbtest");
		directory.mkdirs();

		MonetDBEmbedded db = new MonetDBEmbedded(directory);
		db.startup(false);

		EmbeddedQueryResult result = db.query("SELECT * FROM world;");
		assertEquals(3, result.getColumn(1).columnSize());
		assertEquals(20, result.getColumn(1).getVaule(1));

		result.close();
	}
}
