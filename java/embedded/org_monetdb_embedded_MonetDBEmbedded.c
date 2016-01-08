/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "org_monetdb_embedded_MonetDBEmbedded.h"
#include "embedded.h"

#include "monetdb_config.h"
#include "res_table.h"
#include "mal_type.h"

#include "dlfcn.h"

bool running = false;
char *runningDirectory;

JNIEXPORT jboolean JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_startupWrapper
(JNIEnv *env, jobject object, jstring dbDirectoryJava, jboolean silent) {
	(void)object;
	const char *dbdir_string_tmp = (*env)->GetStringUTFChars(env, dbDirectoryJava, 0);
	char *databaseDirectory = strdup(dbdir_string_tmp);
	unsigned char silent_char = 'n';
	char *err;
	jclass exClass = (*env)->FindClass(env, "java/io/IOException");

	// Release the directories strings
	(*env)->ReleaseStringUTFChars(env, dbDirectoryJava, dbdir_string_tmp);

	// Check if we already have a running db
	if (running) {
		if (strcmp(databaseDirectory, runningDirectory) == 0) {
			return true;
		}
		// Throw a Java exception
		if (exClass == NULL) {
			// Cloud not find the exception class, just return empty object
			return false;
		}
		err = "A database is already working in another directory. Only a single embedded MonetDB process can be started in the same JVM";
		(*env)->ThrowNew(env, exClass, err);
		return false;
	}

	// Set the silent flag based on passed boolean value
	if (silent) {
		silent_char = 'y';
	}

	err = monetdb_startup(databaseDirectory, silent_char);
	// Checking for errors
	if (err != NULL) {
		if (exClass == NULL) {
			// Cloud not find the exception class, just return empty object
			return false;
		}
		(*env)->ThrowNew(env, exClass, err);
		return false;
	}

	// set the flags
	running = true;
	runningDirectory = databaseDirectory;
	return true;
}

JNIEXPORT jobject JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_queryWrapper
(JNIEnv *env, jobject object, jstring query) {
	(void)object;
	res_table *output = NULL;
	int numberOfColumns = 0;
	int numberOfRows = 0;
	const char *query_string_tmp = (*env)->GetStringUTFChars(env, query, 0);
	char *query_string = strdup(query_string_tmp);
	// Release the query string
	(*env)->ReleaseStringUTFChars(env, query, query_string_tmp);

	jobject result;
	jclass resultClass = (*env)->FindClass(env, "org/monetdb/embedded/result/EmbeddedQueryResult");
	// from Java EmbeddedQueryResult(String[] columnNames, String[] columnTypes, int numberOfColumns, int numberOfRows, long resultPointer)
	jmethodID resultConstructor = (*env)->GetMethodID(env, resultClass, "<init>", "([Ljava/lang/String;[Ljava/lang/String;IIJ)V");
	// column names and types string arrays
	jobjectArray columnNames, columnTypes = NULL;
	jclass stringClass = (*env)->FindClass(env, "java/lang/String");

	// In case we can't find the result object class
	if (resultClass == NULL) {
		return NULL;
	}

	// Execute the query
	char* err = monetdb_query(monetdb_connect(), query_string, (void**)&output);

	// Checking for errors
	if (err != NULL) {
		jclass exClass = (*env)->FindClass(env, "java/sql/SQLException");

		// Clean up the result data
		// TODO: creates a segfault, fix later
		//		monetdb_cleanup_result(output);
		if (exClass == NULL) {
			// Cloud not find the exception class, just return empty object
			return NULL;
		}
		(*env)->ThrowNew(env, exClass, err);
		return NULL;
	}

	// Collect result column names and types in string arrays
	// If we have not output, we will return them empty
	if (output) {
		numberOfColumns = output->nr_cols;
	}
	columnNames = (*env)->NewObjectArray(env, numberOfColumns, stringClass, 0);
	columnTypes = (*env)->NewObjectArray(env, numberOfColumns, stringClass, 0);

	if (numberOfColumns > 0) {
		int i;
		for (i = 0; i < numberOfColumns; i++) {
			res_col col = output->cols[i];
			BAT* b = BATdescriptor(col.b);
			char *type;

			numberOfRows = BATcount(b);

			switch (getColumnType(b->T->type)) {
			case TYPE_bit:
				type = "boolean";
				break;
			case TYPE_bte:
				type = "tinyint";
				break;
			case TYPE_sht:
				type = "smallint";
				break;
			case TYPE_int:
				type = "int";
				break;
			case TYPE_lng:
				type = "bigint";
				break;
			case TYPE_flt:
				type = "real";
				break;
			case TYPE_dbl:
				type = "double";
				break;
			case TYPE_str:
				type = "varchar";
				break;
#ifdef HAVE_HGE
			case TYPE_hge:
				type = "huge";
				break;
#endif
			default:
				type = "unknown";
			}
			// Set the meta fields in the result object
			(*env)->SetObjectArrayElement(env, columnNames, i, (*env)->NewStringUTF(env, col.name));
			(*env)->SetObjectArrayElement(env, columnTypes, i, (*env)->NewStringUTF(env, type));
		}
	}
	// Also keep a long value with the result pointer in the Java result object
	long resultTablePointer = (long)output;
	// Create the result object
	// from Java EmbeddedQueryResult(String[] columnNames, String[] columnTypes, int numberOfColumns, int numberOfRows, long resultPointer)
	result = (*env)->NewObject(env, resultClass, resultConstructor, columnNames, columnTypes, numberOfColumns, numberOfRows, resultTablePointer);

	return result;
}

JNIEXPORT void JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_shutdownWrapper
(JNIEnv *env, jobject object) {
	(void)object;
	monetdb_shutdown();
}
