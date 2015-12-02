package org.monetdb.embedded.test;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.monetdb.embedded.MonetDBEmbedded;
import org.monetdb.embedded.result.EmbeddedQueryResult;

public class EmbeddedJDBCTest {
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

//	@BeforeClass
//	public static void createTestDB() throws IOException, SQLException, InterruptedException {
//		final Path directoryPath = Files.createTempDirectory("monetdbtest");
//		datbaseDirectory = directoryPath.toFile();
//
//		db = new MonetDBEmbedded(datbaseDirectory);
//		//	    TimeUnit.SECONDS.sleep(15);
//		db.start();
//
//		db.query("CREATE TABLE test (id integer, val integer);");
//		db.query("INSERT INTO test VALUES (" + testValues[0][0] + ", " + testValues[1][0] + "), (" + testValues[0][1] + ", " + testValues[1][1] + 
//				"), (" + testValues[0][2] + ", " + testValues[1][2] + "), (" + testValues[0][3] + ", " + testValues[1][3] + ");");
//
//		db.query("CREATE TABLE numeric_types_test (fbyte tinyint, fshort smallint, fint integer, flong bigint, freal real, fdouble double);");
//		db.query("INSERT INTO numeric_types_test VALUES (" + numbericTypeTestValues[0] + ", " + numbericTypeTestValues[1] + ", " + numbericTypeTestValues[2] + ", " 
//				+ numbericTypeTestValues[3] + ", " + numbericTypeTestValues[4] + ", " + numbericTypeTestValues[5] + ");");
//		db.query("INSERT INTO numeric_types_test VALUES (null, null, null, null, null, null);");
//
//		db.query("CREATE TABLE char_types_test (fstring string, fvarchar varchar(10));");
//		db.query("INSERT INTO char_types_test VALUES ('" + charTypeTestValues[0] + "', '" + charTypeTestValues[0] + "');");
//		db.query("INSERT INTO char_types_test VALUES (null, null);");
//
//		db.query("CREATE TABLE boolean_types_test (fboolean boolean);");
//		db.query("INSERT INTO boolean_types_test VALUES (" + booleanTypeTestValues[0] + ");");
//		db.query("INSERT INTO boolean_types_test VALUES (" + booleanTypeTestValues[1] + ");");
//		db.query("INSERT INTO boolean_types_test VALUES (null);");
//	}
//
//	@Test
//	public void simpleTest() throws IOException, SQLException {
//		try (EmbeddedQueryResult result = db.query("SELECT * FROM test;")) {
//			assertEquals(4, result.getColumn(1).columnSize());
//			assertEquals(Integer.valueOf(10), result.getColumn(1).getValue(0));
//			assertEquals(10, result.getJDBCResultSet().getInt(1));
//		}
//	}
//
//	@AfterClass
//	public static void cleanup() throws SQLException {
//		db.query("DROP TABLE test");
//		db.query("DROP TABLE numeric_types_test");
//		db.query("DROP TABLE char_types_test");
//		db.query("DROP TABLE boolean_types_test");
//	}
}
