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

/*
 * Class:     org_monetdb_embedded_MonetDBEmbedded
 * Method:    query
 * Signature: (Ljava/lang/String;)Lorg/monetdb/embedded/result/EmbeddedQueryResult;
 */
JNIEXPORT jobject JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_query
(JNIEnv *, jobject, jstring) {
	const char *query_string = (*env)->GetStringUTFChars(env, query, 0);
	res_table* output = NULL;
	jobjectArray *result;
	jclass resultClass = (*env)->FindClass(env, "org/monetdb/embedded/Result");

	// In case we cant find the result object class
	if (resultClass == NULL) {
		return NULL;
	}

	char* err = monetdb_query(query_string, (void**)&output);
	// Release the query string
	(*env)->ReleaseStringUTFChars(env, query, query_string);

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

	// Create the result object
	result = (*env)->NewObjectArray(env, output->nr_cols, resultClass, NULL);
	if (output && output->nr_cols > 0) {
		int i;
		for (i = 0; i < output->nr_cols; i++) {
			res_col col = output->cols[i];
			BAT* b = BATdescriptor(col.b);
			char *type_string;
			int size = BATcount(b);
			int j = 0;
			jobject *array;
			jclass arrayClass = (*env)->FindClass(env, "java/lang/reflect/Array");

			// Set the Java array, depending on its type
			varvalue = bat_to_sexp(b);
			if (varvalue == NULL) {
				switch (ATOMstorage(getColumnType(b->T->type))) {
				case TYPE_int:
					type_string = "integer";
					jintArray array  = (*env)->NewIntArray(env, size);
					for (j = 0; j < size; i++) {
						b->
					}
					// move from the temp structure to the java structure
					(*env)->SetIntArrayRegion(env, result, 0, size, fill);
					break;
				case TYPE_lng:
					break;
				default:
				}
				// Set the meta fields in the result object
				jstring name = (jstring)(*env)->NewStringUTF(env, col.name);
				jstring type = (jstring)(*env)->NewStringUTF(env, type_string);

				// Construct a single result object
				jmethodID resultConstructor = (*env)->GetMethodID(env, resultClass, "<init>", "([L;LJAVA/LANG/STRING;LJAVA/LANG/STRING)V");
				jobject resultObject = (*env)->NewObject(env, resultClass, resultConstructor, array, name, type);

				// Add the result object to the result array
				(*env)->SetObjectArrayElement(env, result, i, resultObject);
			}
			return result;
		}
		return result;
	}
}

JNIEXPORT jobject JNICALL Java_org_monetdb_embedded_MonetDBEmbedded_query
(JNIEnv *, jobject, jstring) {
	jintArray array = (*env)->NewIntArray(env, size);

	BAT* b = BATdescriptor(col.b);
	int size = BATcount(b);

	int i = 0;
	int tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			tmp[i] = (int) ((int*) Tloc(b, BUNfirst(b)))[i];
		}
	}
	//	else {
	//		for (i = 0; i < size; i++) {
	//			int v = ((int*) Tloc(b, BUNfirst(b)))[i];
	//			if (v == int##_nil) {
	//				tmp[i] = naval;
	//			} else {
	//				tmp[i] = (int)v;
	//			}
	//		}
	//	}

	// Move from the tmp C array to a Java array
	(*env)->SetIntArrayRegion(env, array, 0, size, tmp);
	return array;
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
