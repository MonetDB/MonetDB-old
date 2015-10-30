/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#include "org_monetdb_embedded_result_EmbeddedQueryResult.h"
#include "embedded.h"

#include "monetdb_config.h"
#include "res_table.h"
#include "mal_type.h"

static jobject getBooleanColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jbooleanArray values = (*env)->NewBooleanArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/BooleanColumn");
	// from Java BooleanColumn(boolean[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([ZI[Z)V");

	int i = 0;
	jboolean val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (bit) ((bit*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			int v = ((bit*) Tloc(b, BUNfirst(b)))[i];
			if (v == bit_nil) {
				val_tmp[i] = 0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (bit)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetBooleanArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java BooleanColumn(boolean[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getByteColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jbyteArray values = (*env)->NewByteArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/ByteColumn");
	// from Java ByteColumn(byte[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([BI[Z)V");

	int i = 0;
	bte val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (bte) ((bte*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			bte v = ((bte*) Tloc(b, BUNfirst(b)))[i];
			if (v == bte_nil) {
				val_tmp[i] = 0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (bte)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetByteArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java ByteColumn(byte[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getShortColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jshortArray values = (*env)->NewShortArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/ShortColumn");
	// from Java ShortColumn(short[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([SI[Z)V");

	int i = 0;
	short val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (short) ((short*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			short v = ((short*) Tloc(b, BUNfirst(b)))[i];
			if (v == sht_nil) {
				val_tmp[i] = 0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (short)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetShortArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java ShortColumn(short[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getIntegerColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jintArray values = (*env)->NewIntArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/IntegerColumn");
	// from Java IntegerColumn(int[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([II[Z)V");

	int i = 0;
	int val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (int) ((int*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			int v = ((int*) Tloc(b, BUNfirst(b)))[i];
			if (v == int_nil) {
				val_tmp[i] = 0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (int)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetIntArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java IntegerColumn(int[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getLongColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jlongArray values = (*env)->NewLongArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/LongColumn");
	// from Java LongColumn(long[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([JI[Z)V");

	int i = 0;
	jlong val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (lng) ((lng*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			lng v = ((lng*) Tloc(b, BUNfirst(b)))[i];
			if (v == lng_nil) {
				val_tmp[i] = 0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (lng)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetLongArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java LongColumn(long[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getFloatColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jfloatArray values = (*env)->NewFloatArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/FloatColumn");
	// from Java FloatColumn(float[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([FI[Z)V");

	int i = 0;
	float val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (float) ((float*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			float v = ((float*) Tloc(b, BUNfirst(b)))[i];
			if (v == flt_nil) {
				val_tmp[i] = 0.0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (float)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetFloatArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java FloatColumn(float[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getDoubleColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jdoubleArray values = (*env)->NewDoubleArray(env, size);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/DoubleColumn");
	// from Java DoubleColumn(double[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([DI[Z)V");

	int i = 0;
	double val_tmp[size];
	jboolean nul_tmp[size];
	if (b->T->nonil && !b->T->nil) {
		for (i = 0; i < size; i++) {
			val_tmp[i] = (double) ((double*) Tloc(b, BUNfirst(b)))[i];
			nul_tmp[i] = false;
		}
	}
	else {
		for (i = 0; i < size; i++) {
			double v = ((double*) Tloc(b, BUNfirst(b)))[i];
			if (v == dbl_nil) {
				val_tmp[i] = 0.0;
				nul_tmp[i] = true;
			} else {
				val_tmp[i] = (double)v;
				nul_tmp[i] = false;
			}
		}
	}
	// Move from the tmp C arrays to a Java arrays
	(*env)->SetDoubleArrayRegion(env, values, 0, size, val_tmp);
	(*env)->SetBooleanArrayRegion(env, nulls, 0, size, nul_tmp);

	// Create the column object
	// from Java DoubleColumn(double[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

static jobject getStringColumn(JNIEnv *env, BAT *b) {
	int size = BATcount(b);
	// The values and nulls arrays
	jclass stringClass = (*env)->FindClass(env, "java/lang/String");
	jobjectArray values = (*env)->NewObjectArray(env, size, stringClass, 0);
	jbooleanArray nulls = (*env)->NewBooleanArray(env, size);

	jobject column;
	jclass columnClass = (*env)->FindClass(env, "org/monetdb/embedded/result/column/StringColumn");
	// from Java StringColumn(String[] values, int columnSize, boolean[] nullIndex)
	jmethodID columnConstructor = (*env)->GetMethodID(env, columnClass, "<init>", "([Ljava/lang/String;I[Z)V");

	BUN p = 0, q = 0, j = 0;
	BATiter li;
	li = bat_iterator(b);

	if (b->T->nonil && !b->T->nil) {
		BATloop(b, p, q) {
			(*env)->SetObjectArrayElement(env, values, j++, (*env)->NewStringUTF(env, (const char *) BUNtail(li, p)));
		}
	}
	else {
		BATloop(b, p, q) {
			const char *t = (const char *) BUNtail(li, p);
			if (ATOMcmp(TYPE_str, t, str_nil) == 0) {
				(*env)->SetObjectArrayElement(env, values, j++, (*env)->NewStringUTF(env, ""));
			} else {
				(*env)->SetObjectArrayElement(env, values, j, (*env)->NewStringUTF(env, t));
			}
			j++;
		}
	}
	// Create the column object
	// from Java StringColumn(String[] values, int columnSize, boolean[] nullIndex)
	column = (*env)->NewObject(env, columnClass, columnConstructor, values, size, nulls);

	return column;
}

JNIEXPORT jobject JNICALL Java_org_monetdb_embedded_result_EmbeddedQueryResult_getColumnWrapper
(JNIEnv *env, jobject object, jlong resultTablePointer, jint columnIndex) {
	(void)object;
	// The result table
	res_table* result = (res_table *)resultTablePointer;
	// Get the column we need
	res_col col = result->cols[columnIndex];
	BAT* b = BATdescriptor(col.b);

	switch (getColumnType(b->T->type)) {
	case TYPE_bit:
		return getBooleanColumn(env, b);
		break;
	case TYPE_bte:
		return getByteColumn(env, b);
		break;
	case TYPE_sht:
		return getShortColumn(env, b);
		break;
	case TYPE_int:
		return getIntegerColumn(env, b);
		break;
	case TYPE_lng:
		return getLongColumn(env, b);
		break;
	case TYPE_flt:
		return getFloatColumn(env, b);
		break;
	case TYPE_dbl:
		return getDoubleColumn(env, b);
		break;
	case TYPE_str:
		return getStringColumn(env, b);
		break;
#ifdef HAVE_HGE
	case TYPE_hge:
		// TODO: support
		return NULL;
		break;
#endif
	default:
		// TODO: support
		return NULL;
	}

}

JNIEXPORT void JNICALL Java_org_monetdb_embedded_result_EmbeddedQueryResult_cleanupResult
(JNIEnv *env, jobject object, jlong resultTablePointer) {
	(void)object;
	(void)env;
	res_table* result = (res_table *)resultTablePointer;

	monetdb_cleanup_result(monetdb_connect(), result);
}
