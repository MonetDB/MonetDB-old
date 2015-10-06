/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "org_monetdb_embedded_MonetDBEmbedded.h"
#include "embedded.h"
#include "gdk.h"

JNIEXPORT jint JNICALL Java_org_monetdb_embedded_MonetDBLite_startupWrapper
(JNIEnv *env, jobject object, jstring directory, jboolean silent) {
	const char *dir = (*env)->GetStringUTFChars(env, directory, 0);
	unsigned char silent_char = 'n';

	// Release the directory string
	(*env)->ReleaseStringUTFChars(env, directory, dir);
	// Set the silent flag based on passed boolean value
	if (silent) {
		silent_char = 'y';
	}
	return monetdb_startup(dir, silent_char);
}

/*
 * Class:     org_monetdb_embedded_MonetDBEmbedded
 * Method:    query
 * Signature: (Ljava/lang/String;)Lorg/monetdb/embedded/result/EmbeddedQueryResult;
 */
JNIEXPORT jobject JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_query
(JNIEnv *env, jobject object, jstring query) {
	res_table* output = NULL;
	const char *query_string = (*env)->GetStringUTFChars(env, query, 0);

	jobject *result;
	jclass resultClass = (*env)->FindClass(env, "org/monetdb/embedded/result/EmbeddedQueryResult");
	// from Java EmbeddedQueryResult(String[] columnNames, String[] columnTypes, int numberOfColumns, long resultPointer)
	jmethodID resultConstructor = (*env)->GetMethodID(env, resultClass, "<init>", "([Ljava/lang/String;[Ljava/lang/String;IJ)V");
	// column names and types string arrays
	jobjectArray columnNames, columnTypes = NULL;

	// In case we can't find the result object class
	if (resultClass == NULL) {
		return NULL;
	}

	char* err = monetdb_query(query_string, (void**)&output);
	// Release the query string
	(*env)->ReleaseStringUTFChars(env, query, query_string);

	// Checking for errors
	if (err != NULL) {
		jclass exClass = (*env)->FindClass(env, "java/sql/SQLException");

		// Clean up the result data
		monetdb_cleanup_result(output);
		if (exClass == NULL) {
			// Cloud not find the exception class, just return empty object
			return NULL;
		}
		return (*env)->ThrowNew(env, exClass, err);
	}

	// Collect result column names and types in string arrays
	// If we have not output, we will return them empty
	columnNames = (jobjectArray)env->NewObjectArray(output->nr_cols, env->FindClass("java/lang/String"), env->NewStringUTF(""));
	columnTypes = (jobjectArray)env->NewObjectArray(output->nr_cols, env->FindClass("java/lang/String"), env->NewStringUTF(""));
	if (output && output->nr_cols > 0) {
		int i;
		for (i = 0; i < output->nr_cols; i++) {
			res_col col = output->cols[i];
			BAT* b = BATdescriptor(col.b);
			char *type;

			switch (ATOMstorage(getColumnType(b->T->type))) {
			case TYPE_sht:
				type = "short";
				break;
			case TYPE_int:
				type = "integer";
				break;
			case TYPE_lng:
				type = "long";
				break;
			case TYPE_flt:
				type = "float";
				break;
			case TYPE_dbl:
				type = "double";
				break;
			case TYPE_str:
				type = "string";
				break;
#ifdef HAVE_HGE
			case TYPE_hge:
				type_string = "huge";
				break;
#endif
			default:
				type_string = "unknown";
			}
			// Set the meta fields in the result object
			env->SetObjectArrayElement(columnNames, i, env->NewStringUTF(env, col.name));
			env->SetObjectArrayElement(columnTypes, i, env->NewStringUTF(env, type));
		}
	}
	// Also keep a long value with the result pointer in the Java result object
	long resultTablePointer = (long)output;
	// Create the result object
	// from Java EmbeddedQueryResult(String[] columnNames, String[] columnTypes, int numberOfColumns, long resultPointer)
	result = (*env)->NewObject(env, resultClass, resultConstructor, columnNames, columnTypes, output->nr_cols, resultTablePointer);

	return result;
}

/*
 * Class:     org_monetdb_embedded_MonetDBLite
 * Method:    append
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/reflect/Array;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_monetdb_embedded_MonetDBLite_append
(JNIEnv *env, jobject object, jstring schema_name, jstring table_name, jobject data) {
	const char *schema_name = (*env)->GetStringUTFChars(env, schema, 0);
	const char *table_name = (*env)->GetStringUTFChars(env, table, 0);

	(*env)->ReleaseStringUTFChars(env, string, str);
	return (*env)->NewStringUTF(env, strupr(cap));
}
