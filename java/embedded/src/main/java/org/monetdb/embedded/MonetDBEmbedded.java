/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;

import org.monetdb.embedded.result.EmbeddedQueryResult;

/**
 * Embedded version of MonetDB.
 * Communication between Java and native C is done via JNI.
 * <br/>
 * <strong>Note</strong>: You can have only one embedded MonetDB database running per JVM process.
 */
public class MonetDBEmbedded implements Closeable {
	final private static String LIB_PATH_VAR = "java.library.path"; 
	final private static String NATIVE_LIB_PATH_IN_JAR = "src" + File.separatorChar + "main" +
			File.separatorChar + "resources" + File.separatorChar + "lib";
	final private static String NATIVE_LIB_NAME = "monetdb5";

	/** 
	 * Flag if the embedded database was already started.
	 */
	private boolean running = false;
	/** 
	 * The working directory for MonetDB.
	 */
	private final File databaseDirectory;
	/**
	 * Silent flag for database startup
	 */
	private final boolean silentFlag;

	/**
	 * The native embedded MonetDB library.
	 */
	static {
		try {
			// Try load the embedded library
			System.loadLibrary(NATIVE_LIB_NAME);
		} catch (UnsatisfiedLinkError e) {
			// Still no, then get the lib bundled in the jar
			loadLibFromJar("lib" + NATIVE_LIB_NAME + ".jnilib");
		}
	}

	private static void loadLibFromJar(String fileName) {
		String pathToLib = NATIVE_LIB_PATH_IN_JAR + File.separatorChar + fileName;
		try {
			InputStream in = MonetDBEmbedded.class.getResourceAsStream(File.separatorChar + pathToLib);
			if (in == null) {
				// OK, the input stream is null, hence no .jar
				// This was probably a test and/or in an IDE
				// Just read the files from the src/main/resources/lib dir
				in = new FileInputStream(new File(pathToLib));
			}
			// Set a temp location to extract (and load from later)
			final Path tempLibsDir = Files.createTempDirectory("monetdb-embedded-libs");
			File fileOut = new File(tempLibsDir.toString() + File.separatorChar + fileName);
			try (OutputStream out = new FileOutputStream(fileOut)) {
				byte[] buffer = new byte[in.available()];
				while (in.read(buffer) != -1) {
			        out.write(buffer);
				}
				out.flush();
				in.close();
				// Load the lib from the extracted file
				System.load(fileOut.toString());
			}
		} catch (IOException e) {
			throw new UnsatisfiedLinkError("Unable to extract native library from JAR:" + e.getMessage());
		}
	}

	/**
	 * @param databaseDirectory Database directory
	 */
	public MonetDBEmbedded(final File databaseDirectory) {
		this(databaseDirectory, true);
	}

	/**
	 * @param databaseDirectory Database directory
	 * @param silentFlag Silent flag
	 */
	public MonetDBEmbedded(final File databaseDirectory, final boolean silentFlag) {
		if (!databaseDirectory.isDirectory()) {
			throw new IllegalArgumentException(databaseDirectory + " is not a directory");
		}
		this.databaseDirectory = databaseDirectory;
		this.silentFlag = silentFlag;
	}

	/** 
	 * Check if the embedded database is running.
	 * 
	 * @return {@code True} if the was started successfully, otherwise {@code False}.
	 */
	public boolean isRunning() {
		return running;
	}

	/**
	 * Start the embedded database up. Starting in a existing database directory should restore
	 * the database's last committed state.
	 * 
	 * @return {@code True} if the was started successfully or is already running, otherwise {@code False}.
	 * @throws IOException Database startup failure
	 */
	public boolean start() throws IOException {
		if (!running) {
			if (startupWrapper(databaseDirectory.getAbsolutePath(), silentFlag)){
				running = true;
			}
		}
		return running;
	}

	/**
	 * Execute an SQL query in an embedded database.
	 * 
	 * @param query The SQL query string
	 * @return The query result object, {@code null} if the database is not running
	 * @throws SQLException
	 */
	public EmbeddedQueryResult query(String query) throws SQLException {
		if (!running) {
			return null;
		}

		String queryString = query;
		if (!queryString.endsWith(";")) {
			queryString = queryString + ";";
		}
		return queryWrapper(queryString);
	}

	/**
	 * Start the embedded database.
	 * 
	 * @param dbDirecory Database directory
	 * @param silent Silent flag
	 * @return Startup status code
	 */
	private native boolean startupWrapper(String dbDirecory, boolean silent) throws IOException;

	/**
	 * Execute an SQL query in an embedded database.
	 * 
	 * @param query The SQL query string
	 * @return The query result object, {@code null} if the database is not running
	 * @throws SQLException
	 */
	private native EmbeddedQueryResult queryWrapper(String query) throws SQLException;

	/**
	 * Shut down the embedded database.
	 */
	private native void shutdownWrapper();

	@Override
	public void close() throws IOException {
		// Avoid for now
		//		shutdownWrapper();
	}
}
