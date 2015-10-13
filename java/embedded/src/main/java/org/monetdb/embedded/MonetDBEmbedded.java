/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

package org.monetdb.embedded;

import java.io.File;
import java.lang.reflect.Array;
import java.sql.SQLException;

import org.monetdb.embedded.result.EmbeddedQueryResult;

/**
 * Embedded version of MonetDB.
 * Communication between Java and native C is done via JNI.
 * 
 */
public class MonetDBEmbedded {
	static {
		System.loadLibrary("embedded_java");
	}

	/** 
	 * Flag if the embedded database was already started.
	 */
	private boolean running = false;
	/** 
	 * The working directory for MonetDB.
	 */
	private File directory;

	/**
	 * You can instantiate multiple object, 
	 * just make sure they are assigned different directories.
	 * 
	 * @param directory Database directory 
	 */
	public MonetDBEmbedded(File directory) {
		if (!directory.isDirectory()) {
			throw new IllegalArgumentException(directory + " is not a directory");
		}
		this.directory = directory;
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
	 * @param silent Silent flag to logging messages
	 * @return {@code True} if the was started successfully or is already running, otherwise {@code False}.
	 */
	public boolean startup(boolean silent) {
		if (!running) {
			if (startupWrapper(directory.getAbsolutePath(), silent) == 1) {
				running = true;
			}
		}
		return running;
	}

	/**
	 * Start the embedded database up.
	 * 
	 * @param direcrory Database directory
	 * @param silent Silent flag
	 * @return Startup status code
	 */
	private native int startupWrapper(String dir, boolean silent);

	public native EmbeddedQueryResult query(String query) throws SQLException;

	public native String append(String schema, String table, Array data) throws SQLException;
}
