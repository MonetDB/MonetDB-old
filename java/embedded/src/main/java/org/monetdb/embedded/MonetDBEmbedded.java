/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;

import org.monetdb.embedded.result.EmbeddedQueryResult;

/**
 * Embedded version of MonetDB.
 * Communication between Java and native C is done via JNI.
 * <br/>
 * <strong>Note</strong>: You can have only one embedded MonetDB database running per JVM process.
 */
public class MonetDBEmbedded {
	static {
		// Load the embedded library
		System.loadLibrary("embedded_java");
	}
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
			if (startupWrapper(System.getProperty("java.library.path"), databaseDirectory.getAbsolutePath(), silentFlag)){
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
	 * @param libsDirectory Libraries directory
	 * @param dbDirecory Database directory
	 * @param silent Silent flag
	 * @return Startup status code
	 */
	private native boolean startupWrapper(String libsDirectory, String dbDirecory, boolean silent) throws IOException;

	/**
	 * Execute an SQL query in an embedded database.
	 * 
	 * @param query The SQL query string
	 * @return The query result object, {@code null} if the database is not running
	 * @throws SQLException
	 */
	private native EmbeddedQueryResult queryWrapper(String query) throws SQLException;
}
