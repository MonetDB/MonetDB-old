/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * @a M. Raasveldt
 * MonetDB embedded in Python library. Contains functions to initialize MonetDB and to perform SQL queries after it has been initialized.
 */
#include "embeddedpy.h"

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>

#include "monetdb_config.h"
#include "monet_options.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_linker.h"
#include "msabaoth.h"
#include "sql_scenario.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// copy paste from pyapi.c
#include <longintrepr.h>

struct _PyInput{
    void *dataptr;                      //pointer to input data
    BAT *bat;                           //pointer to input BAT
    int bat_type;                       //BAT type as TYPE_<type>
    size_t count;                       //amount of elements in BAT
    bool scalar;                        //True if the input is a scalar (in this case, BAT* is NULL)
};
#define PyInput struct _PyInput

struct _PyReturn{
    PyObject *numpy_array;              //PyArrayObject* with data (can be NULL, as long as array_data is set)
    PyObject *numpy_mask;               //PyArrayObject* with mask (NULL if there is no mask)
    void *array_data;                   //void* pointer to data
    bool *mask_data;                    //bool* pointer to mask data
    size_t count;                       //amount of return elements
    size_t memory_size;                 //memory size of each element
    int result_type;                    //result type as NPY_<TYPE>
    bool multidimensional;              //whether or not the result is multidimensional
};
#define PyReturn struct _PyReturn

PyObject *PyMaskedArray_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message);
PyObject *PyArrayObject_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message);
PyObject *PyNullMask_FromBAT(BAT *b, size_t t_start, size_t t_end);
static char *PyError_CreateException(char *error_text, char *pycall);

PyObject *PyObject_CheckForConversion(PyObject *pResult, int expected_columns, int *actual_columns, char **return_message);
bool PyObject_PreprocessObject(PyObject *pResult, PyReturn *pyreturn_values, int column_count, char **return_message);
BAT *PyObject_ConvertToBAT(PyReturn *ret, int bat_type, int index, int seqbase, char **return_message);
int PyType_ToBat(int);
//! Returns true if a PyObject is a scalar type ('scalars' in this context means numeric or string types)
bool PyType_IsPyScalar(PyObject *object);
//! Returns true if the PyObject is of type numpy.ndarray, and false otherwise
bool PyType_IsNumpyArray(PyObject *object);
//! Returns true if the PyObject is of type numpy.ma.core.MaskedArray, and false otherwise
bool PyType_IsNumpyMaskedArray(PyObject *object);
//! Returns true if the PyObject is of type pandas.core.frame.DataFrame, and false otherwise
bool PyType_IsPandasDataFrame(PyObject *object);
//! Returns true if the PyObject is of type lazyarray, and false otherwise
bool PyType_IsLazyArray(PyObject *object);
char *BatType_Format(int);
//! Formats NPY_#type as a String (so NPY_INT => "INT"), for usage in error reporting and warnings
char *PyType_Format(int);
//! Converts a base-10 utf32-encoded string to a lng value
bool utf32_to_lng(Py_UNICODE *utf32, size_t maxsize, lng *value);
//! Converts a base-10 utf32-encoded string to a dbl value
bool utf32_to_dbl(Py_UNICODE *utf32, size_t maxsize, dbl *value);
//! Converts a base-10 utf32-encoded string to a hge value
bool utf32_to_hge(Py_UNICODE *utf32, size_t maxsize, hge *value);
//! Converts a PyObject to a dbl value
bool py_to_dbl(PyObject *ptr, dbl *value);
//! Converts a PyObject to a lng value
bool py_to_lng(PyObject *ptr, lng *value);
//! Converts a PyObject to a hge value
bool py_to_hge(PyObject *ptr, hge *value);
//! Converts a double to a string and writes it into the string "str"
void dbl_to_string(char* str, dbl value);
//! Copies the string of size up to max_size from the source to the destination, returns FALSE if "source" is not a legal ASCII string (i.e. a character is >= 128)
bool string_copy(char * source, char* dest, size_t max_size);
//! Converts a hge to a string and writes it into the string "str"
int hge_to_string(char *str, hge );
int utf32_strlen(const Py_UNICODE *utf32_str);
bool utf32_to_utf8(size_t offset, size_t size, char *utf8_storage, const Py_UNICODE *utf32);
int utf32_char_to_utf8_char(size_t position, char *utf8_storage, Py_UNICODE utf32_char);
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// copy paste from embedded.c
bool monetdb_isinit(void);
int monetdb_startup(char* dir, char silent);
char* monetdb_query(char* query, void** result);
void monetdb_cleanup_result(void* output);
str monetdb_get_columns(char *table_name, int *column_count, char ***column_names, int **column_types);
str monetdb_create_table(char *table_name, size_t columns, BAT **bats, char **column_names);
str monetdb_insert_into_table(char *table_name, size_t columns, BAT **bats, char **column_names);
/////////////////////////////////////////////////////////////////////////////////////////////////

//LOCALSTATEDIR
PyObject *monetdb_init(PyObject *self, PyObject *args)
{
	(void) self;
	if (!PyString_CheckExact(args)) {
   		PyErr_SetString(PyExc_TypeError, "Expected a directory name as an argument.");
		return NULL;
	}

	{
		char *directory = &(((PyStringObject*)args)->ob_sval[0]);
		printf("Making directory %s\n", directory);
		if (GDKcreatedir(directory) != GDK_SUCCEED) {
   			PyErr_Format(PyExc_Exception, "Failed to create directory %s.", directory);
   			return NULL;
		}
		if (monetdb_startup(directory, 1) < 0) {
	   		PyErr_SetString(PyExc_Exception, "Failed to initialize MonetDB with the specified directory.");
			return NULL;
		}
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_sql(PyObject *self, PyObject *args)
{
	(void) self;
	if (!PyString_CheckExact(args)) {
   		PyErr_SetString(PyExc_TypeError, "Expected a SQL query as a single string argument.");
		return NULL;
	}
	if (!monetdb_isinit()) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}

	{
		PyObject *result;
		res_table* output = NULL;
		char* err;
		// Append ';'' to the SQL query, just in case
		args = PyString_FromFormat("%s;", &(((PyStringObject*)args)->ob_sval[0]));
		// Perform the SQL query
		err = monetdb_query(&(((PyStringObject*)args)->ob_sval[0]), (void**)&output);
		if (err != NULL) { 
	   		PyErr_Format(PyExc_Exception, "SQL Query Failed: %s", (err ? err : "<no error>"));
			return NULL;
		}
		// Construct a dictionary from the output columns (dict[name] = column)
		result = PyDict_New();
		if (output && output->nr_cols > 0) {
			PyInput input;
			PyObject *numpy_array;
			char *msg = NULL;
			int i;
			for (i = 0; i < output->nr_cols; i++) {
				res_col col = output->cols[i];
				BAT* b = BATdescriptor(col.b);

            	input.bat = b;
				input.count = BATcount(b);
            	input.bat_type = ATOMstorage(getColumnType(b->T->type));
            	input.scalar = false;

            	numpy_array = PyMaskedArray_FromBAT(&input, 0, input.count, &msg);
            	if (!numpy_array) {
					monetdb_cleanup_result(output);
			   		PyErr_Format(PyExc_Exception, "SQL Query Failed: %s", (msg ? msg : "<no error>"));
					return NULL;
            	}
            	PyDict_SetItem(result, PyString_FromString(output->cols[i].name), numpy_array);
			}
			monetdb_cleanup_result(output);
			return result;
		} else {
			Py_RETURN_NONE;
		}
	}
}

PyObject *monetdb_create(PyObject *self, PyObject *args)
{
	char *table_name;
	PyObject *dict = NULL, *values = NULL;
	int i;
	(void) self;
	if (!monetdb_isinit()) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "sO|O", &table_name, &dict, &values)) {
		return NULL;
	}
	if (values == NULL) {
		if (!PyDict_CheckExact(dict)) {
	   		PyErr_SetString(PyExc_TypeError, "the second argument to be a dict");
			return NULL;
		}
   		PyErr_Format(PyExc_Exception, "dict is not implemented yet");
		return NULL;
		Py_RETURN_NONE;
	} else {
		if (!PyList_CheckExact(dict)) {
	   		PyErr_SetString(PyExc_TypeError, "the second argument (column names) must be a list");
			return NULL;
		}
		if (PyList_Size(dict) == 0) {
	   		PyErr_SetString(PyExc_TypeError, "the list of column names must be non-zero");
			return NULL;
		}

		for(i = 0; i < PyList_Size(dict); i++) {
			PyObject *column_name = PyList_GetItem(dict, i);
			if (!PyString_CheckExact(column_name)) {
		   		PyErr_Format(PyExc_TypeError, "the entry %d in the column names is not a string", i);
				return NULL;
			}
		}
		//convert the values to BATs and create the table
		{
			char *msg = NULL;
			PyObject *pResult;
			int columns = PyList_Size(dict);
			PyReturn *pyreturn_values = NULL;
			BAT **bats = NULL;
			char **column_names = NULL;

			pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
			if (pResult == NULL) goto cleanup;
			pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
			if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

			bats = GDKzalloc(sizeof(BAT*) * columns);
			column_names = GDKzalloc(sizeof(char*) * columns);
			for(i = 0; i < columns; i++) {
				column_names[i] = PyString_AS_STRING(PyList_GetItem(dict, i));
				bats[i] = PyObject_ConvertToBAT(&pyreturn_values[i], PyType_ToBat(pyreturn_values[i].result_type), i, 0, &msg);
				if (bats[i] == NULL) goto cleanup; 
 			}
 			msg = monetdb_create_table(table_name, columns, bats, column_names);
cleanup:
			if (pyreturn_values) GDKfree(pyreturn_values);
			if (bats) {
				for(i = 0; i < columns; i++) {
					if (bats[i] != NULL) BBPunfix(bats[i]->batCacheid);
				}
				GDKfree(bats);
			}
			if (column_names) GDKfree(column_names);
			if (msg != NULL) {
				PyErr_Format(PyExc_Exception, "%s", msg);
				return NULL;
			}
		}
	}
	Py_RETURN_NONE;
}

PyObject *monetdb_insert(PyObject *self, PyObject *args)
{
	char *table_name;
	PyObject *values = NULL;
	(void) self;
	if (!monetdb_isinit()) {
   		PyErr_SetString(PyExc_Exception, "monetdb has not been initialized yet");
		return NULL;
	}
	if (!PyArg_ParseTuple(args, "sO", &table_name, &values)) {
		return NULL;
	}
	{
		char *msg = NULL;
		PyObject *pResult;
		PyReturn *pyreturn_values = NULL;
		BAT **bats = NULL;
		int i;
		char **column_names = NULL;
		int *column_types = NULL;
		int columns;

		msg = monetdb_get_columns(table_name, &columns, &column_names, &column_types);

		pResult = PyObject_CheckForConversion(values, columns, NULL, &msg);
		if (pResult == NULL) goto cleanup;
		pyreturn_values = GDKzalloc(sizeof(PyReturn) * columns);
		if (!PyObject_PreprocessObject(pResult, pyreturn_values, columns, &msg)) goto cleanup;

		bats = GDKzalloc(sizeof(BAT*) * columns);
		for(i = 0; i < columns; i++) {
			bats[i] = PyObject_ConvertToBAT(&pyreturn_values[i], column_types[i], i, 0, &msg);
			if (bats[i] == NULL) goto cleanup; 
			}
			msg = monetdb_insert_into_table(table_name, columns, bats, column_names);
cleanup:
		if (pyreturn_values) GDKfree(pyreturn_values);
		if (column_names) GDKfree(column_names);
		if (column_types) GDKfree(column_types);
		if (bats) {
			for(i = 0; i < columns; i++) {
				if (bats[i] != NULL) BBPunfix(bats[i]->batCacheid);
			}
			GDKfree(bats);
		}
		if (msg != NULL) {
			PyErr_Format(PyExc_Exception, "%s", msg);
			return NULL;
		}
	}
	Py_RETURN_NONE;
}

void init_embedded_py(void)
{
    //import numpy stuff
    import_array();
}

/////////////////////////////////////////////////////////
// This code is copy pasted from embedded.c, should be changed to a shared library
/////////////////////////////////////////////////////////
typedef str (*SQLstatementIntern_ptr_tpe)(Client, str*, str, bit, bit, res_table**);
SQLstatementIntern_ptr_tpe SQLstatementIntern_ptr = NULL;
typedef void (*res_table_destroy_ptr_tpe)(res_table *t);
res_table_destroy_ptr_tpe res_table_destroy_ptr = NULL;

static bit monetdb_embedded_initialized = 0;
static MT_Lock monetdb_embedded_lock;

bool monetdb_isinit(void) {
	return monetdb_embedded_initialized;
}

static void* lookup_function(char* lib, char* func) {
	void *dl, *fun;
	dl = mdlopen(lib, RTLD_NOW | RTLD_GLOBAL);
	if (dl == NULL) {
		return NULL;
	}
	fun = dlsym(dl, func);
	dlclose(dl);
	return fun;
}

int monetdb_startup(char* dir, char silent) {
	opt *set = NULL;
	int setlen = 0;
	int retval = -1;
	void* res = NULL;
	char mod_path[1000];

	MT_lock_init(&monetdb_embedded_lock, "monetdb_embedded_lock");
	MT_lock_set(&monetdb_embedded_lock, "monetdb.startup");
	if (monetdb_embedded_initialized) goto cleanup;

	setlen = mo_builtin_settings(&set);
	setlen = mo_add_option(&set, setlen, opt_cmdline, "gdk_dbpath", dir);
	if (GDKinit(set, setlen) == 0)  goto cleanup;

	snprintf(mod_path, 1000, "%s/../lib/monetdb5", BINDIR);
	GDKsetenv("monet_mod_path", mod_path);
	GDKsetenv("mapi_disable", "true");
	GDKsetenv("max_clients", "0");

	if (silent) THRdata[0] = stream_blackhole_create();
	msab_dbpathinit(GDKgetenv("gdk_dbpath"));
	if (mal_init() != 0) goto cleanup;
	if (silent) mal_clients[0].fdout = THRdata[0];

	// This dynamically looks up functions, because the library containing them is loaded at runtime.
	SQLstatementIntern_ptr = (SQLstatementIntern_ptr_tpe) lookup_function("lib_sql",  "SQLstatementIntern");
	res_table_destroy_ptr  = (res_table_destroy_ptr_tpe)  lookup_function("libstore", "res_table_destroy");
	if (SQLstatementIntern_ptr == NULL || res_table_destroy_ptr == NULL) goto cleanup;

	monetdb_embedded_initialized = true;
	// sanity check, run a SQL query
	if (monetdb_query("SELECT * FROM tables;", res) != NULL) {
		monetdb_embedded_initialized = false;
		goto cleanup;
	}
	if (res != NULL) monetdb_cleanup_result(res);
	retval = 0;

cleanup:
	mo_free_options(set, setlen);
	MT_lock_unset(&monetdb_embedded_lock, "monetdb.startup");
	return retval;
}

char* monetdb_query(char* query, void** result) {
	str res;
	Client c = &mal_clients[0];
	if (!monetdb_embedded_initialized) {
		fprintf(stderr, "Embedded MonetDB is not started.\n");
		return NULL;
	}
	res = (*SQLstatementIntern_ptr)(c, &query, "name", 1, 0, (res_table **) result);
	SQLautocommit(c, ((backend *) c->sqlcontext)->mvc);
	return res;
}

void monetdb_cleanup_result(void* output) {
	(*res_table_destroy_ptr)((res_table*) output);
}

static char *BatType_ToSQLType(int type)
{
    switch (type)
    {
        case TYPE_bit:
        case TYPE_bte: return "TINYINT";
        case TYPE_sht: return "SMALLINT";
        case TYPE_int: return "INTEGER";
        case TYPE_lng: return "BIGINT";
        case TYPE_flt: return "FLOAT";
        case TYPE_dbl: return "DOUBLE";
        case TYPE_str: return "STRING";
        case TYPE_hge: return "HUGEINT";
        case TYPE_oid: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

static char *BatType_DefaultValue(int type)
{
    switch (type)
    {
        case TYPE_bit:
        case TYPE_bte: 
        case TYPE_sht: 
        case TYPE_int:
        case TYPE_lng:
        case TYPE_flt: 
        case TYPE_hge:
        case TYPE_dbl: return "0";
        case TYPE_str: return "''";
        default: return "UNKNOWN";
    }
}

str monetdb_insert_into_table(char *table_name, size_t columns, BAT **bats, char **column_names)
{
	size_t i;
	const int max_length = 10000;
	char query[max_length], copy[max_length];
	Client c = &mal_clients[0];
	char *msg = MAL_SUCCEED;
	void *res;

	snprintf(query, max_length, "insert into %s values ", table_name);
	for(i = 0; i < columns; i++) {
		snprintf(copy, max_length, "%s%s%s%s", query, i == 0 ? "(" : "", BatType_DefaultValue(bats[i]->T->type), i < columns - 1 ? "," : ");");
		strcpy(query, copy);
	}

	c->_append_columns = columns;
	c->_append_bats = (void**)bats;
	c->_append_column_names = column_names;
	msg = monetdb_query(query, &res);
	c->_append_columns = 0;
	c->_append_bats = NULL;
	c->_append_column_names = NULL;
	if (msg != MAL_SUCCEED) {
		return msg;
	}
	return msg;
}

str monetdb_create_table(char *table_name, size_t columns, BAT **bats, char **column_names)
{
	const int max_length = 10000;
	char query[max_length], copy[max_length];
	size_t i;
	void *res;
	char *msg = MAL_SUCCEED;

	if (!monetdb_embedded_initialized) {
		fprintf(stderr, "Embedded MonetDB is not started.\n");
		return NULL;
	}
	//format the CREATE TABLE query
	snprintf(query, max_length, "create table %s(", table_name);
	for(i = 0; i < columns; i++) {
		snprintf(copy, max_length, "%s %s %s%s", query, column_names[i], BatType_ToSQLType(bats[i]->T->type), i < columns - 1 ? "," : ");");
		strcpy(query, copy);
	}

	msg = monetdb_query(query, &res);
	if (msg != MAL_SUCCEED) {
		return msg;
	}

	return monetdb_insert_into_table(table_name, columns, bats, column_names);
}


str monetdb_get_columns(char *table_name, int *column_count, char ***column_names, int **column_types)
{
	Client c = &mal_clients[0];
	mvc *m;
	sql_schema *s;
	sql_table *t;
	char *msg = MAL_SUCCEED;

	if ((msg = getSQLContext(c, NULL, &m, NULL)) != NULL)
		return msg;

	s = mvc_bind_schema(m, "sys");
	if (s == NULL)
		msg = createException(MAL, "embedded", "Missing schema!");
	t = mvc_bind_table(m, s, table_name);
	if (t == NULL)
		msg = createException(MAL, "embedded", "Could not find table %s", table_name);

	{
		const int columns = t->columns.set->cnt;
		if (column_count != NULL) {
			*column_count = columns;
		}
		if (column_names != NULL) {
			int i;
			*column_names = GDKzalloc(sizeof(char*) * columns);
			for(i = 0; i < columns; i++) {
				*column_names[i] = ((sql_base*)t->columns.set->h->data)[i].name;
			}
		}
		if (column_types != NULL) {
			int i;
			*column_types = GDKzalloc(sizeof(int) * columns);
			for(i = 0; i < columns; i++) {
				*column_types[i] = ((sql_column*)t->columns.set->h->data)[i].type.type->localtype;
			}
		}
	}

	return msg;
}

/////////////////////////////////////////////////////////
// This code is copy pasted from pyapi.c, should be changed to a shared library but PyAPI is not in this branch
/////////////////////////////////////////////////////////


int utf8_length(unsigned char utf8_char);
int utf8_strlen(const char *utf8_str, bool *ascii);
PyObject *PyLong_FromHge(hge h);
void lng_to_string(char* str, lng value);
#ifdef HAVE_HGE
bool s_to_hge(char *ptr, size_t size, hge *value);
#endif
bool s_to_dbl(char *ptr, size_t size, dbl *value);
bool s_to_lng(char *ptr, size_t size, lng *value);
//using macros, create a number of str_to_<type>, unicode_to_<type> and pyobject_to_<type> functions (we are Java now)
#define CONVERSION_FUNCTION_HEADER_FACTORY(tpe)          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value);          \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value);                  \
    bool pyobject_to_##tpe(void *ptr, size_t size, tpe *value);                  \

CONVERSION_FUNCTION_HEADER_FACTORY(bit)
CONVERSION_FUNCTION_HEADER_FACTORY(sht)
CONVERSION_FUNCTION_HEADER_FACTORY(int)
CONVERSION_FUNCTION_HEADER_FACTORY(lng)
CONVERSION_FUNCTION_HEADER_FACTORY(flt)
CONVERSION_FUNCTION_HEADER_FACTORY(dbl)
#ifdef HAVE_HGE
CONVERSION_FUNCTION_HEADER_FACTORY(hge)
#endif



#define BAT_TO_NP(bat, mtpe, nptpe)                                                                                                 \
        vararray = PyArray_New(&PyArray_Type, 1, (npy_intp[1]) {(t_end-t_start)},                                                   \
            nptpe, NULL, &((mtpe*) Tloc(bat, BUNfirst(bat)))[t_start], 0,                                                           \
            NPY_ARRAY_CARRAY || !NPY_ARRAY_WRITEABLE, NULL);                                                                        


// This #define creates a new BAT with the internal data and mask from a Numpy array, without copying the data
// 'bat' is a BAT* pointer, which will contain the new BAT. TYPE_'mtpe' is the BAT type, and 'batstore' is the heap storage type of the BAT (this should be STORE_CMEM or STORE_SHARED)
#define CREATE_BAT_ZEROCOPY(bat, mtpe, batstore) {                                                                      \
        bat = BATnew(TYPE_void, TYPE_##mtpe, 0, TRANSIENT);                                                             \
        BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                   \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                                           \
        /*Change nil values to the proper values, if they exist*/                                                       \
        if (mask != NULL)                                                                                               \
        {                                                                                                               \
            for (iu = 0; iu < ret->count; iu++)                                                                         \
            {                                                                                                           \
                if (mask[index_offset * ret->count + iu] == TRUE)                                                       \
                {                                                                                                       \
                    (*(mtpe*)(&data[(index_offset * ret->count + iu) * ret->memory_size])) = mtpe##_nil;                \
                    bat->T->nil = 1;                                                                                    \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        bat->T->nonil = 1 - bat->T->nil;                                                                                \
        /*When we create a BAT a small part of memory is allocated, free it*/                                           \
        GDKfree(bat->T->heap.base);                                                                                     \
                                                                                                                        \
        bat->T->heap.base = &data[(index_offset * ret->count) * ret->memory_size];                                      \
        bat->T->heap.size = ret->count * ret->memory_size;                                                              \
        bat->T->heap.free = bat->T->heap.size;  /*There are no free places in the array*/                               \
        /*If index_offset > 0, we are mapping part of a multidimensional array.*/                                       \
        /*The entire array will be cleared when the part with index_offset=0 is freed*/                                 \
        /*So we set this part of the mapping to 'NOWN'*/                                                                \
        if (index_offset > 0) bat->T->heap.storage = STORE_NOWN;                                                        \
        else bat->T->heap.storage = batstore;                                                                           \
        bat->T->heap.newstorage = STORE_MEM;                                                                            \
        bat->S->count = ret->count;                                                                                     \
        bat->S->capacity = ret->count;                                                                                  \
        bat->S->copiedtodisk = false;                                                                                   \
                                                                                                                        \
        /*Take over the data from the numpy array*/                                                                     \
        if (ret->numpy_array != NULL) PyArray_CLEARFLAGS((PyArrayObject*)ret->numpy_array, NPY_ARRAY_OWNDATA);          \
    }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It assumes the BAT 'bat' is already created with the proper size.
// This should only be used with integer data that can be cast. It assumes the Numpy Array has an internal array of type 'mtpe_from', and the BAT has an internal array of type 'mtpe_to'.
// it then does the cast by simply doing BAT[i] = (mtpe_to) ((mtpe_from*)NUMPY_ARRAY[i]), which only works if both mtpe_to and mtpe_from are integers
#define NP_COL_BAT_LOOP(bat, mtpe_to, mtpe_from) {                                                                                               \
    if (mask == NULL)                                                                                                                            \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));    \
        }                                                                                                                                        \
    }                                                                                                                                            \
    else                                                                                                                                         \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                    \
            {                                                                                                                                    \
                bat->T->nil = 1;                                                                                                                 \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                       \
            }                                                                                                                                    \
            else                                                                                                                                 \
            {                                                                                                                                    \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));\
            }                                                                                                                                    \
        }                                                                                                                                        \
    } }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It converts the data from the Numpy Array to the BAT using a function
// This function has to have the prototype 'bool function(void *data, size_t memory_size, mtpe_to *resulting_value)', and either return False (if conversion fails) 
//  or write the value into the 'resulting_value' pointer. This is used convertring strings/unicodes/python objects to numeric values.
#define NP_COL_BAT_LOOP_FUNC(bat, mtpe_to, func) {                                                                                                    \
    mtpe_to value;                                                                                                                                    \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                          \
            {                                                                                                                                         \
                msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to);     \
                goto wrapup;                                                                                                                          \
            }                                                                                                                                         \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                        \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                            \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                      \
                {                                                                                                                                     \
                    msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to); \
                    goto wrapup;                                                                                                                      \
                }                                                                                                                                     \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                    \
            }                                                                                                                                         \
        }                                                                                                                                             \
    } }
    

// This #define is for converting a numeric numpy array into a string BAT. 'conv' is a function that turns a numeric value of type 'mtpe' to a char* array.
#define NP_COL_BAT_STR_LOOP(bat, mtpe, conv)                                                                                                          \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                                  \
            BUNappend(bat, utf8_string, FALSE);                                                                                                       \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                BUNappend(b, str_nil, FALSE);                                                                                                         \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                              \
                BUNappend(bat, utf8_string, FALSE);                                                                                                   \
            }                                                                                                                                         \
        }                                                                                                                                             \
    }


PyObject *PyMaskedArray_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message)
{
    BAT *b = inp->bat;
    char *msg;
    PyObject *vararray = PyArrayObject_FromBAT(inp, t_start, t_end, return_message);
    if (vararray == NULL) {
        return NULL;
    }
    // To deal with null values, we use the numpy masked array structure
    // The masked array structure is an object with two arrays of equal size, a data array and a mask array
    // The mask array is a boolean array that has the value 'True' when the element is NULL, and 'False' otherwise
    // If the BAT has Null values, we construct this masked array
    if (!(b->T->nil == 0 && b->T->nonil == 1))
    {
        PyObject *mask;
        PyObject *mafunc = PyObject_GetAttrString(PyImport_Import(PyString_FromString("numpy.ma")), "masked_array");
        PyObject *maargs;
        PyObject *nullmask = PyNullMask_FromBAT(b, t_start, t_end);

        if (nullmask == Py_None) {
            maargs = PyTuple_New(1);
            PyTuple_SetItem(maargs, 0, vararray);
        } else {
            maargs = PyTuple_New(2);
            PyTuple_SetItem(maargs, 0, vararray);
            PyTuple_SetItem(maargs, 1, (PyObject*) nullmask);
        }
       
        // Now we will actually construct the mask by calling the masked array constructor
        mask = PyObject_CallObject(mafunc, maargs);
        if (!mask) {
            msg = PyError_CreateException("Failed to create mask", NULL);
            goto wrapup;
        }
        Py_DECREF(maargs);
        Py_DECREF(mafunc);

        vararray = mask;
    }
    return vararray;
wrapup:
    *return_message = msg;
    return NULL;
}

PyObject *PyArrayObject_FromBAT(PyInput *inp, size_t t_start, size_t t_end, char **return_message)
{
    // This variable will hold the converted Python object
    PyObject *vararray = NULL; 
    char *msg;
    size_t j = 0;
    BUN p = 0, q = 0;
    BATiter li;
    BAT *b = inp->bat;

    assert(!inp->scalar); //input has to be a BAT

    if (b == NULL) 
    {
        // No BAT was found, we can't do anything in this case
        msg = createException(MAL, "pyapi.eval", MAL_MALLOC_FAIL);
        goto wrapup;
    }

    switch (inp->bat_type) {
    case TYPE_bte:
        BAT_TO_NP(b, bte, NPY_INT8);
        break;
    case TYPE_sht:
        BAT_TO_NP(b, sht, NPY_INT16);
        break;
    case TYPE_int:
        BAT_TO_NP(b, int, NPY_INT32);
        break;
    case TYPE_lng:
        BAT_TO_NP(b, lng, NPY_INT64);
        break;
    case TYPE_flt:
        BAT_TO_NP(b, flt, NPY_FLOAT32);
        break;
    case TYPE_dbl:
        BAT_TO_NP(b, dbl, NPY_FLOAT64);
        break;
    case TYPE_str:
        {
            bool unicode = false;
            li = bat_iterator(b);
            //create a NPY_OBJECT array object
            vararray = PyArray_New(
                &PyArray_Type, 
                1, 
                (npy_intp[1]) {t_end - t_start},  
                NPY_OBJECT,
                NULL, 
                NULL, 
                0,         
                0, 
                NULL);

            {
                j = 0;
                BATloop(b, p, q) {
                    if (j >= t_start) {
                        bool ascii;
                        const char *t = (const char *) BUNtail(li, p);
                        if (strcmp(t, str_nil) == 0) continue;
                        utf8_strlen(t, &ascii); 
                        unicode = !ascii || unicode; 
                        if (unicode) break;
                    }
                    if (j == t_end) break;
                    j++;
                }
            }

            j = 0;
            BATloop(b, p, q)
            {
                if (j >= t_start) {
                    char *t = (char *) BUNtail(li, p);
                    PyObject *obj;
                    if (unicode)
                    {
                        if (strcmp(t, str_nil) == 0) {
                             //str_nil isn't a valid UTF-8 character (it's 0x80), so we can't decode it as UTF-8 (it will throw an error)
                            obj = PyUnicode_FromString("-");
                        }
                        else {
                            //otherwise we can just decode the string as UTF-8
                            obj = PyUnicode_FromString(t);
                        }
                    } else {
                        {
                            obj = PyString_FromString(t);
                        }
                    }

                    if (obj == NULL)
                    {
                        msg = createException(MAL, "pyapi.eval", "Failed to create string.");
                        goto wrapup;
                    }
                    ((PyObject**)PyArray_DATA((PyArrayObject*)vararray))[j - t_start] = obj;
                }
                if (j == t_end) break;
                j++;
            }
        }
        break;
#ifdef HAVE_HGE
    case TYPE_hge:
    {
        li = bat_iterator(b);

        //create a NPY_OBJECT array to hold the huge type
        vararray = PyArray_New(
            &PyArray_Type, 
            1, 
            (npy_intp[1]) { t_end - t_start },  
            NPY_OBJECT, 
            NULL, 
            NULL, 
            0,
            0,
            NULL);

        j = 0;
        BATloop(b, p, q) {
            if (j >= t_start) {
                PyObject *obj;
                const hge *t = (const hge *) BUNtail(li, p);
                obj = PyLong_FromHge(*t);
                ((PyObject**)PyArray_DATA((PyArrayObject*)vararray))[j - t_start] = obj;
            }
            if (j == t_end) break;
            j++;
        }
        break;
    }
#endif
    default:
        msg = createException(MAL, "pyapi.eval", "unknown argument type ");
        goto wrapup;
    }
    if (vararray == NULL) {
        msg = PyError_CreateException("Failed to convert BAT to Numpy array.", NULL);
        goto wrapup;
    }
    return vararray;
wrapup:
    *return_message = msg;
    return NULL;
}

PyObject *PyNullMask_FromBAT(BAT *b, size_t t_start, size_t t_end)
{
    // We will now construct the Masked array, we start by setting everything to False
    PyArrayObject* nullmask = (PyArrayObject*) PyArray_ZEROS(1, (npy_intp[1]) {(t_end - t_start)}, NPY_BOOL, 0);
    const void *nil = ATOMnilptr(b->ttype);
    int (*atomcmp)(const void *, const void *) = ATOMcompare(b->ttype);
    size_t j;
    bool found_nil = false;
    BATiter bi = bat_iterator(b);

    for (j = 0; j < t_end - t_start; j++) {
        if ((*atomcmp)(BUNtail(bi, BUNfirst(b) + t_start + j), nil) == 0) {
            ((bool*)PyArray_DATA(nullmask))[j] = true;
            found_nil = true;
        }
    }
    if (!found_nil) {
        Py_DECREF(nullmask);
        Py_RETURN_NONE;
    }

    return (PyObject*)nullmask;
}


static char *PyError_CreateException(char *error_text, char *pycall)
{
    PyObject *py_error_type = NULL, *py_error_value = NULL, *py_error_traceback = NULL;
    char *py_error_string = NULL;
    lng line_number;

    PyErr_Fetch(&py_error_type, &py_error_value, &py_error_traceback);
    if (py_error_value) {
        PyObject *error;
        PyErr_NormalizeException(&py_error_type, &py_error_value, &py_error_traceback);
        error = PyObject_Str(py_error_value);

        py_error_string = PyString_AS_STRING(error);
        Py_XDECREF(error);
        if (pycall != NULL && strlen(pycall) > 0) {
            // If pycall is given, we try to parse the line number from the error string and format pycall so it only displays the lines around the line number
            // (This code is pretty ugly, sorry)
            char line[] = "line ";
            char linenr[32]; //we only support functions with at most 10^32 lines, mostly out of philosophical reasons
            size_t i = 0, j = 0, pos = 0, nrpos = 0;

            // First parse the line number from py_error_string
            for(i = 0; i < strlen(py_error_string); i++) {
                if (pos < strlen(line)) {
                    if (py_error_string[i] == line[pos]) {
                        pos++;
                    }
                } else {
                    if (py_error_string[i] == '0' || py_error_string[i] == '1' || py_error_string[i] == '2' || 
                        py_error_string[i] == '3' || py_error_string[i] == '4' || py_error_string[i] == '5' || 
                        py_error_string[i] == '6' || py_error_string[i] == '7' || py_error_string[i] == '8' || py_error_string[i] == '9') {
                        linenr[nrpos++] = py_error_string[i];
                    }
                }
            }
            linenr[nrpos] = '\0';
            if (!str_to_lng(linenr, nrpos, &line_number)) {
                // No line number in the error, so just display a normal error
                goto finally;
            }

            // Now only display the line numbers around the error message, we display 5 lines around the error message
            {
                char lineinformation[5000]; //we only support 5000 characters for 5 lines of the program, should be enough
                nrpos = 0; // Current line number 
                pos = 0; //Current position in the lineinformation result array
                for(i = 0; i < strlen(pycall); i++) {
                    if (pycall[i] == '\n' || i == 0) { 
                        // Check if we have arrived at a new line, if we have increment the line count
                        nrpos++;  
                        // Now check if we should display this line 
                        if (nrpos >= ((size_t)line_number - 2) && nrpos <= ((size_t)line_number + 2) && pos < 4997) { 
                            // We shouldn't put a newline on the first line we encounter, only on subsequent lines
                            if (nrpos > ((size_t)line_number - 2)) lineinformation[pos++] = '\n';
                            if ((size_t)line_number == nrpos) {
                                // If this line is the 'error' line, add an arrow before it, otherwise just add spaces
                                lineinformation[pos++] = '>';
                                lineinformation[pos++] = ' ';
                            } else {
                                lineinformation[pos++] = ' ';
                                lineinformation[pos++] = ' ';
                            }
                            lng_to_string(linenr, nrpos); // Convert the current line number to string and add it to lineinformation
                            for(j = 0; j < strlen(linenr); j++) {
                                lineinformation[pos++] = linenr[j];
                            }
                            lineinformation[pos++] = '.';
                            lineinformation[pos++] = ' ';
                        }
                    }
                    if (pycall[i] != '\n' && nrpos >= (size_t)line_number - 2 && nrpos <= (size_t)line_number + 2 && pos < 4999) { 
                        // If we are on a line number that we have to display, copy the text from this line for display
                        lineinformation[pos++] = pycall[i];
                    }
                }
                lineinformation[pos] = '\0';
                return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, lineinformation, py_error_string);
            }
        }
    }
    else {
        py_error_string = "";
    }
finally:
    if (pycall == NULL) return createException(MAL, "pyapi.eval", "%s\n%s", error_text, py_error_string);
    return createException(MAL, "pyapi.eval", "%s\n%s\n%s", error_text, pycall, py_error_string);
}



int utf8_strlen(const char *utf8_str, bool *ascii)
{
    int utf8_char_count = 0;
    int i = 0;
    //we traverse the string and simply count the amount of utf8 characters in the string
    while (true)
    {
        int offset;
        if (utf8_str[i] == '\0') break;
        offset = utf8_length(utf8_str[i]);
        if (offset < 0) return -1; //invalid utf8 character
        i += offset;
        utf8_char_count++;
    }
    if (ascii != NULL) *ascii = i == utf8_char_count;
    return utf8_char_count;
}

PyObject *PyLong_FromHge(hge h)
{
    PyLongObject *z;
    size_t size = 0;
    hge shift = h >= 0 ? h : -h;
    hge prev = shift;
    int i;
    while(shift > 0) {
        size++;
        shift = shift >> PyLong_SHIFT;
    }
    z = _PyLong_New(size);
    for(i = size - 1; i >= 0; i--) {
        digit result = (digit)(prev >> (PyLong_SHIFT * i));
        prev = prev - ((prev >> (PyLong_SHIFT * i)) << (PyLong_SHIFT * i));
        z->ob_digit[i] = result;
    }
    if (h < 0) Py_SIZE(z) = -(Py_SIZE(z));
    return (PyObject*) z;
}


#define CONVERSION_FUNCTION_FACTORY(tpe, strval)          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value)           \
    {                                                              \
        strval val;                                                \
        if (!s_to_##strval((char*)ptr, size, &val)) return false;   \
        *value = (tpe)val;                                         \
        return true;                                               \
    }                                                              \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value)                   \
    {                                                              \
        strval val;                                                \
        if (!utf32_to_##strval((Py_UNICODE*)ptr, size / 4, &val)) return false;         \
        *value = (tpe)val;                                         \
        return true;                                               \
    }                                                              \
    bool pyobject_to_##tpe(void *ptr, size_t size, tpe *value)                   \
    {                                                              \
        strval val;                                                \
        (void) size;                                               \
        if (!py_to_##strval(*((PyObject**)ptr), &val)) return false;         \
        *value = (tpe)val;                                         \
        return true;                                               \
    }                       
    
CONVERSION_FUNCTION_FACTORY(bit, lng)
CONVERSION_FUNCTION_FACTORY(sht, lng)
CONVERSION_FUNCTION_FACTORY(int, lng)
CONVERSION_FUNCTION_FACTORY(lng, lng)
CONVERSION_FUNCTION_FACTORY(flt, dbl)
CONVERSION_FUNCTION_FACTORY(dbl, dbl)
#ifdef HAVE_HGE
CONVERSION_FUNCTION_FACTORY(hge, hge)
#endif


void lng_to_string(char* str, lng value)
{
    lng k = 0;
    int ind = 0;
    int base = 0;
    int ch = 0;
    char c;
    //sign
    if (value < 0) { base = 1; str[ind++] = '-'; value *= -1; }

    if (value == 0) str[ind++] = '0';
    while(value > 0)
    {
        k = value / 10;
        ch = value - k * 10;
        value = k;

        switch(ch)
        {
            case 0: str[ind++] = '0'; break;
            case 1: str[ind++] = '1'; break;
            case 2: str[ind++] = '2'; break;
            case 3: str[ind++] = '3'; break;
            case 4: str[ind++] = '4'; break;
            case 5: str[ind++] = '5'; break;
            case 6: str[ind++] = '6'; break;
            case 7: str[ind++] = '7'; break;
            case 8: str[ind++] = '8'; break;
            case 9: str[ind++] = '9'; break;
        }
    }
    str[ind] = '\0';
    for (ind--; ind > base; ind--)
    {
        c = str[ind];
        str[ind] = str[base];
        str[base++] = c;
    }
}

int utf8_length(unsigned char utf8_char)
{
    //the first byte tells us how many bytes the utf8 character uses
    if      (utf8_char < 0b10000000) return 1;
    else if (utf8_char < 0b11100000) return 2;
    else if (utf8_char < 0b11110000) return 3;
    else if (utf8_char < 0b11111000) return 4;
    else return -1; //invalid utf8 character, the maximum value of the first byte is 0b11110111
}

bool s_to_lng(char *ptr, size_t size, lng *value)
{
    size_t length = size - 1;
    int i = length;
    lng factor = 1;
    *value = 0;
    for( ; i >= 0; i--)
    {
        switch(ptr[i])
        {
            case '0': break;
            case '1': *value += factor; break;
            case '2': *value += 2 * factor; break;
            case '3': *value += 3 * factor; break;
            case '4': *value += 4 * factor; break;
            case '5': *value += 5 * factor; break;
            case '6': *value += 6 * factor; break;
            case '7': *value += 7 * factor; break;
            case '8': *value += 8 * factor; break;
            case '9': *value += 9 * factor; break;
            case '-': *value *= -1; break;
            case '.':
            case ',': *value = 0; factor = 1; continue;
            case '\0': continue;
            default: 
            {
                return false;
            }
        }
        factor *= 10;
    }
    return true;
}

#ifdef HAVE_HGE
bool s_to_hge(char *ptr, size_t size, hge *value)
{
    size_t length = size - 1;
    int i = length;
    hge factor = 1;
    *value = 0;
    for( ; i >= 0; i--)
    {
        switch(ptr[i])
        {
            case '0': break;
            case '1': *value += factor; break;
            case '2': *value += 2 * factor; break;
            case '3': *value += 3 * factor; break;
            case '4': *value += 4 * factor; break;
            case '5': *value += 5 * factor; break;
            case '6': *value += 6 * factor; break;
            case '7': *value += 7 * factor; break;
            case '8': *value += 8 * factor; break;
            case '9': *value += 9 * factor; break;
            case '-': *value *= -1; break;
            case '.':
            case ',': *value = 0; factor = 1; continue;
            case '\0': continue;
            default: 
            {
                return false;
            }
        }
        factor *= 10;
    }
    return true;
}
#endif

bool s_to_dbl(char *ptr, size_t size, dbl *value)
{
    size_t length = size - 1;
    int i = length;
    dbl factor = 1;
    *value = 0;
    for( ; i >= 0; i--)
    {
        switch(ptr[i])
        {
            case '0': break;
            case '1': *value += factor; break;
            case '2': *value += 2 * factor; break;
            case '3': *value += 3 * factor; break;
            case '4': *value += 4 * factor; break;
            case '5': *value += 5 * factor; break;
            case '6': *value += 6 * factor; break;
            case '7': *value += 7 * factor; break;
            case '8': *value += 8* factor; break;
            case '9': *value += 9 * factor; break;
            case '-': *value *= -1; break;
            case '.':
            case ',': *value /= factor; factor = 1; continue;
            case '\0': continue;
            default: return false;
        }
        factor *= 10;
    }
    return true;
}


// This #define creates a new BAT with the internal data and mask from a Numpy array, without copying the data
// 'bat' is a BAT* pointer, which will contain the new BAT. TYPE_'mtpe' is the BAT type, and 'batstore' is the heap storage type of the BAT (this should be STORE_CMEM or STORE_SHARED)
#define CREATE_BAT_ZEROCOPY(bat, mtpe, batstore) {                                                                      \
        bat = BATnew(TYPE_void, TYPE_##mtpe, 0, TRANSIENT);                                                             \
        BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                   \
        bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                                           \
        /*Change nil values to the proper values, if they exist*/                                                       \
        if (mask != NULL)                                                                                               \
        {                                                                                                               \
            for (iu = 0; iu < ret->count; iu++)                                                                         \
            {                                                                                                           \
                if (mask[index_offset * ret->count + iu] == TRUE)                                                       \
                {                                                                                                       \
                    (*(mtpe*)(&data[(index_offset * ret->count + iu) * ret->memory_size])) = mtpe##_nil;                \
                    bat->T->nil = 1;                                                                                    \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        bat->T->nonil = 1 - bat->T->nil;                                                                                \
        /*When we create a BAT a small part of memory is allocated, free it*/                                           \
        GDKfree(bat->T->heap.base);                                                                                     \
                                                                                                                        \
        bat->T->heap.base = &data[(index_offset * ret->count) * ret->memory_size];                                      \
        bat->T->heap.size = ret->count * ret->memory_size;                                                              \
        bat->T->heap.free = bat->T->heap.size;  /*There are no free places in the array*/                               \
        /*If index_offset > 0, we are mapping part of a multidimensional array.*/                                       \
        /*The entire array will be cleared when the part with index_offset=0 is freed*/                                 \
        /*So we set this part of the mapping to 'NOWN'*/                                                                \
        if (index_offset > 0) bat->T->heap.storage = STORE_NOWN;                                                        \
        else bat->T->heap.storage = batstore;                                                                           \
        bat->T->heap.newstorage = STORE_MEM;                                                                            \
        bat->S->count = ret->count;                                                                                     \
        bat->S->capacity = ret->count;                                                                                  \
        bat->S->copiedtodisk = false;                                                                                   \
                                                                                                                        \
        /*Take over the data from the numpy array*/                                                                     \
        if (ret->numpy_array != NULL) PyArray_CLEARFLAGS((PyArrayObject*)ret->numpy_array, NPY_ARRAY_OWNDATA);          \
    }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It assumes the BAT 'bat' is already created with the proper size.
// This should only be used with integer data that can be cast. It assumes the Numpy Array has an internal array of type 'mtpe_from', and the BAT has an internal array of type 'mtpe_to'.
// it then does the cast by simply doing BAT[i] = (mtpe_to) ((mtpe_from*)NUMPY_ARRAY[i]), which only works if both mtpe_to and mtpe_from are integers
#define NP_COL_BAT_LOOP(bat, mtpe_to, mtpe_from) {                                                                                               \
    if (mask == NULL)                                                                                                                            \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));    \
        }                                                                                                                                        \
    }                                                                                                                                            \
    else                                                                                                                                         \
    {                                                                                                                                            \
        for (iu = 0; iu < ret->count; iu++)                                                                                                      \
        {                                                                                                                                        \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                    \
            {                                                                                                                                    \
                bat->T->nil = 1;                                                                                                                 \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                       \
            }                                                                                                                                    \
            else                                                                                                                                 \
            {                                                                                                                                    \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = (mtpe_to)(*(mtpe_from*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));\
            }                                                                                                                                    \
        }                                                                                                                                        \
    } }

// This #define converts a Numpy Array to a BAT by copying the internal data to the BAT. It converts the data from the Numpy Array to the BAT using a function
// This function has to have the prototype 'bool function(void *data, size_t memory_size, mtpe_to *resulting_value)', and either return False (if conversion fails) 
//  or write the value into the 'resulting_value' pointer. This is used convertring strings/unicodes/python objects to numeric values.
#define NP_COL_BAT_LOOP_FUNC(bat, mtpe_to, func) {                                                                                                    \
    mtpe_to value;                                                                                                                                    \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                          \
            {                                                                                                                                         \
                msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to);     \
                goto wrapup;                                                                                                                          \
            }                                                                                                                                         \
            ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                        \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = mtpe_to##_nil;                                                                            \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                if (!func(&data[(index_offset * ret->count + iu) * ret->memory_size], ret->memory_size, &value))                                      \
                {                                                                                                                                     \
                    msg = createException(MAL, "pyapi.eval", "Could not convert from type %s to type %s", PyType_Format(ret->result_type), #mtpe_to); \
                    goto wrapup;                                                                                                                      \
                }                                                                                                                                     \
                ((mtpe_to*) Tloc(bat, BUNfirst(bat)))[iu] = value;                                                                                    \
            }                                                                                                                                         \
        }                                                                                                                                             \
    } }
    

// This #define is for converting a numeric numpy array into a string BAT. 'conv' is a function that turns a numeric value of type 'mtpe' to a char* array.
#define NP_COL_BAT_STR_LOOP(bat, mtpe, conv)                                                                                                          \
    if (mask == NULL)                                                                                                                                 \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                                  \
            BUNappend(bat, utf8_string, FALSE);                                                                                                       \
        }                                                                                                                                             \
    }                                                                                                                                                 \
    else                                                                                                                                              \
    {                                                                                                                                                 \
        for (iu = 0; iu < ret->count; iu++)                                                                                                           \
        {                                                                                                                                             \
            if (mask[index_offset * ret->count + iu] == TRUE)                                                                                         \
            {                                                                                                                                         \
                bat->T->nil = 1;                                                                                                                      \
                BUNappend(b, str_nil, FALSE);                                                                                                         \
            }                                                                                                                                         \
            else                                                                                                                                      \
            {                                                                                                                                         \
                conv(utf8_string, *((mtpe*)&data[(index_offset * ret->count + iu) * ret->memory_size]));                                              \
                BUNappend(bat, utf8_string, FALSE);                                                                                                   \
            }                                                                                                                                         \
        }                                                                                                                                             \
    }

// This very big #define combines all the previous #defines for one big #define that is responsible for converting a Numpy array (described in the PyReturn object 'ret')
// to a BAT of type 'mtpe'. This should only be used for numeric BATs (but can be used for any Numpy Array). The resulting BAT will be stored in 'bat'.
#define NP_CREATE_BAT(bat, mtpe) {                                                                                                                             \
        bool *mask = NULL;                                                                                                                                     \
        char *data = NULL;                                                                                                                                     \
        if (ret->mask_data != NULL)                                                                                                                            \
        {                                                                                                                                                      \
            mask = (bool*) ret->mask_data;                                                                                                                     \
        }                                                                                                                                                      \
        if (ret->array_data == NULL)                                                                                                                           \
        {                                                                                                                                                      \
            msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.\n");                                                            \
            goto wrapup;                                                                                                                                       \
        }                                                                                                                                                      \
        data = (char*) ret->array_data;                                                                                                                        \
        if (ret->count > 0 && TYPE_##mtpe == PyType_ToBat(ret->result_type) && (ret->count * ret->memory_size < BUN_MAX) &&                    \
            (ret->numpy_array == NULL || PyArray_FLAGS((PyArrayObject*)ret->numpy_array) & NPY_ARRAY_OWNDATA))                                                 \
        {                                                                                                                                                      \
            /*We can only create a direct map if the numpy array type and target BAT type*/                                                                    \
            /*are identical, otherwise we have to do a conversion.*/                                                                                           \
            assert(ret->array_data != NULL);                                                                                                                   \
            if (ret->numpy_array == NULL)                                                                                                                      \
            {                                                                                                                                                  \
                /*shared memory return*/                                                                                                                       \
                CREATE_BAT_ZEROCOPY(bat, mtpe, STORE_SHARED);                                                                                                  \
                ret->array_data = NULL;                                                                                                                        \
            }                                                                                                                                                  \
            else                                                                                                                                               \
            {                                                                                                                                                  \
                CREATE_BAT_ZEROCOPY(bat, mtpe, STORE_CMEM);                                                                                                    \
            }                                                                                                                                                  \
        }                                                                                                                                                      \
        else                                                                                                                                                   \
        {                                                                                                                                                      \
            bat = BATnew(TYPE_void, TYPE_##mtpe, ret->count, TRANSIENT);                                                                                       \
            BATseqbase(bat, seqbase); bat->T->nil = 0; bat->T->nonil = 1;                                                                                      \
            bat->tkey = 0; bat->tsorted = 0; bat->trevsorted = 0;                                                                                              \
            switch(ret->result_type)                                                                                                                           \
            {                                                                                                                                                  \
                case NPY_BOOL:       NP_COL_BAT_LOOP(bat, mtpe, bit); break;                                                                                   \
                case NPY_BYTE:       NP_COL_BAT_LOOP(bat, mtpe, bte); break;                                                                                   \
                case NPY_SHORT:      NP_COL_BAT_LOOP(bat, mtpe, sht); break;                                                                                   \
                case NPY_INT:        NP_COL_BAT_LOOP(bat, mtpe, int); break;                                                                                   \
                case NPY_LONG:                                                                                                                                 \
                case NPY_LONGLONG:   NP_COL_BAT_LOOP(bat, mtpe, lng); break;                                                                                   \
                case NPY_UBYTE:      NP_COL_BAT_LOOP(bat, mtpe, unsigned char); break;                                                                         \
                case NPY_USHORT:     NP_COL_BAT_LOOP(bat, mtpe, unsigned short); break;                                                                        \
                case NPY_UINT:       NP_COL_BAT_LOOP(bat, mtpe, unsigned int); break;                                                                          \
                case NPY_ULONG:                                                                                                                                \
                case NPY_ULONGLONG:  NP_COL_BAT_LOOP(bat, mtpe, unsigned long); break;                                                                         \
                case NPY_FLOAT16:                                                                                                                              \
                case NPY_FLOAT:      NP_COL_BAT_LOOP(bat, mtpe, flt); break;                                                                                   \
                case NPY_DOUBLE:                                                                                                                               \
                case NPY_LONGDOUBLE: NP_COL_BAT_LOOP(bat, mtpe, dbl); break;                                                                                   \
                case NPY_STRING:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, str_to_##mtpe); break;                                                                    \
                case NPY_UNICODE:    NP_COL_BAT_LOOP_FUNC(bat, mtpe, unicode_to_##mtpe); break;                                                                \
                case NPY_OBJECT:     NP_COL_BAT_LOOP_FUNC(bat, mtpe, pyobject_to_##mtpe); break;                                                               \
                default:                                                                                                                                       \
                    msg = createException(MAL, "pyapi.eval", "Unrecognized type. Could not convert to %s.\n", BatType_Format(TYPE_##mtpe));                    \
                    goto wrapup;                                                                                                                               \
            }                                                                                                                                                  \
            bat->T->nonil = 1 - bat->T->nil;                                                                                                                   \
            BATsetcount(bat, ret->count);                                                                                                                      \
            BATsettrivprop(bat);                                                                                                                               \
        }                                                                                                                                                      \
    }


PyObject *PyObject_CheckForConversion(PyObject *pResult, int expected_columns, int *actual_columns, char **return_message)
{
    char *msg;
    int columns = 0;
    if (pResult) {
        PyObject * pColO = NULL;
        if (PyType_IsPandasDataFrame(pResult)) {
            //the result object is a Pandas data frame
            //we can convert the pandas data frame to a numpy array by simply accessing the "values" field (as pandas dataframes are numpy arrays internally)
            pResult = PyObject_GetAttrString(pResult, "values"); 
            if (pResult == NULL) {
                msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                goto wrapup; 
            }
            //we transpose the values field so it's aligned correctly for our purposes
            pResult = PyObject_GetAttrString(pResult, "T");
            if (pResult == NULL) {
                msg = createException(MAL, "pyapi.eval", "Invalid Pandas data frame.");
                goto wrapup; 
            }
        }

        if (PyType_IsPyScalar(pResult)) { //check if the return object is a scalar 
            if (expected_columns == 1 || expected_columns <= 0)  {
                //if we only expect a single return value, we can accept scalars by converting it into an array holding an array holding the element (i.e. [[pResult]])
                PyObject *list = PyList_New(1);
                PyList_SetItem(list, 0, pResult);
                pResult = list;

                list = PyList_New(1);
                PyList_SetItem(list, 0, pResult);
                pResult = list;

                columns = 1;
            }
            else {
                //the result object is a scalar, yet we expect more than one return value. We can only convert the result into a list with a single element, so the output is necessarily wrong.
                msg = createException(MAL, "pyapi.eval", "A single scalar was returned, yet we expect a list of %d columns. We can only convert a single scalar into a single column, thus the result is invalid.", expected_columns);
                goto wrapup;
            }
        }
        else {
            //if it is not a scalar, we check if it is a single array
            bool IsSingleArray = TRUE;
            PyObject *data = pResult;
            if (PyType_IsNumpyMaskedArray(data)) {
                data = PyObject_GetAttrString(pResult, "data");   
                if (data == NULL) {
                    msg = createException(MAL, "pyapi.eval", "Invalid masked array.");
                    goto wrapup;
                }           
            }
            if (PyType_IsNumpyArray(data)) {
                if (PyArray_NDIM((PyArrayObject*)data) != 1) {
                    IsSingleArray = FALSE;
                }
                else {
                    pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, 0));
                    IsSingleArray = PyType_IsPyScalar(pColO);
                }
            }
            else if (PyList_Check(data)) {
                pColO = PyList_GetItem(data, 0);
                IsSingleArray = PyType_IsPyScalar(pColO);
            }
            else if (PyType_IsLazyArray(data)) {
                pColO = data;
                IsSingleArray = TRUE;
            } else if (!PyType_IsNumpyMaskedArray(data)) {
                //it is neither a python array, numpy array or numpy masked array, thus the result is unsupported! Throw an exception!
                msg = createException(MAL, "pyapi.eval", "Unsupported result object. Expected either an array, a numpy array, a numpy masked array or a pandas data frame, but received an object of type \"%s\"", PyString_AsString(PyObject_Str(PyObject_Type(data))));
                goto wrapup;
            }

            if (IsSingleArray) {
                if (expected_columns == 1 || expected_columns <= 0) {
                    //if we only expect a single return value, we can accept a single array by converting it into an array holding an array holding the element (i.e. [pResult])
                    PyObject *list = PyList_New(1);
                    PyList_SetItem(list, 0, pResult);
                    pResult = list;

                    columns = 1;
                }
                else {
                    //the result object is a single array, yet we expect more than one return value. We can only convert the result into a list with a single array, so the output is necessarily wrong.
                    msg = createException(MAL, "pyapi.eval", "A single array was returned, yet we expect a list of %d columns. The result is invalid.", expected_columns);
                    goto wrapup;
                }
            }
            else {
                //the return value is an array of arrays, all we need to do is check if it is the correct size
                int results = 0;
                if (PyList_Check(data)) results = PyList_Size(data);
                else results = PyArray_DIMS((PyArrayObject*)data)[0];
                columns = results;
                if (results != expected_columns && expected_columns > 0) {
                    //wrong return size, we expect pci->retc arrays
                    msg = createException(MAL, "pyapi.eval", "An array of size %d was returned, yet we expect a list of %d columns. The result is invalid.", results, expected_columns);
                    goto wrapup;
                }
            }
        }
    } else {
        msg = createException(MAL, "pyapi.eval", "Invalid result object. No result object could be generated.");
        goto wrapup;
    }

    if (actual_columns != NULL) *actual_columns = columns;
    return pResult;
wrapup:
    if (actual_columns != NULL) *actual_columns = columns;
    *return_message = msg;
    return NULL;
}


bool PyObject_PreprocessObject(PyObject *pResult, PyReturn *pyreturn_values, int column_count, char **return_message)
{
    int i;
    char *msg;
    for (i = 0; i < column_count; i++) {
        // Refers to the current Numpy mask (if it exists)
        PyObject *pMask = NULL;
        // Refers to the current Numpy array
        PyObject * pColO = NULL;
        // This is the PyReturn header information for the current return value, we will fill this now
        PyReturn *ret = &pyreturn_values[i];

        ret->multidimensional = FALSE;
        // There are three possibilities (we have ensured this right after executing the Python call by calling PyObject_CheckForConversion)
        // 1: The top level result object is a PyList or Numpy Array containing pci->retc Numpy Arrays
        // 2: The top level result object is a (pci->retc x N) dimensional Numpy Array [Multidimensional]
        // 3: The top level result object is a (pci->retc x N) dimensional Numpy Masked Array [Multidimensional]
        if (PyList_Check(pResult)) {
            // If it is a PyList, we simply get the i'th Numpy array from the PyList
            pColO = PyList_GetItem(pResult, i);
        }
        else {
            // If it isn't, the result object is either a Nump Masked Array or a Numpy Array
            PyObject *data = pResult;
            if (PyType_IsNumpyMaskedArray(data)) {
                data = PyObject_GetAttrString(pResult, "data"); // If it is a Masked array, the data is stored in the masked_array.data attribute
                pMask = PyObject_GetAttrString(pResult, "mask");    
            }

            // We can either have a multidimensional numpy array, or a single dimensional numpy array 
            if (PyArray_NDIM((PyArrayObject*)data) != 1) {
                // If it is a multidimensional numpy array, we have to convert the i'th dimension to a NUMPY array object
                ret->multidimensional = TRUE;
                ret->result_type = PyArray_DESCR((PyArrayObject*)data)->type_num;
            }
            else {
                // If it is a single dimensional Numpy array, we get the i'th Numpy array from the Numpy Array
                pColO = PyArray_GETITEM((PyArrayObject*)data, PyArray_GETPTR1((PyArrayObject*)data, i));
            }
        }

        // Now we have to do some preprocessing on the data
        if (ret->multidimensional) {
            // If it is a multidimensional Numpy array, we don't need to do any conversion, we can just do some pointers
            ret->count = PyArray_DIMS((PyArrayObject*)pResult)[1];        
            ret->numpy_array = pResult;                   
            ret->numpy_mask = pMask;   
            ret->array_data = PyArray_DATA((PyArrayObject*)ret->numpy_array);
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA((PyArrayObject*)ret->numpy_mask);                 
            ret->memory_size = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->elsize;   
        }
        else {
            if (PyType_IsLazyArray(pColO)) {
                // To handle returning of lazy arrays, we just convert them to a Numpy array. This is slow and could be done much faster, but since this can only happen if we directly return one of the input arguments this should be a rare situation anyway.
                //pColO = PyLazyArray_AsNumpyArray(pColO);
                //if (pColO == NULL) {
                    msg = PyError_CreateException("Failed to convert lazy array to numpy array.\n", NULL);
                    goto wrapup;
                //}
            }
            // If it isn't we need to convert pColO to the expected Numpy Array type
            ret->numpy_array = PyArray_FromAny(pColO, NULL, 1, 1, NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, NULL);
            if (ret->numpy_array == NULL) {
                msg = createException(MAL, "pyapi.eval", "Could not create a Numpy array from the return type.\n");
                goto wrapup;
            }
            
            ret->result_type = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->type_num; // We read the result type from the resulting array
            ret->memory_size = PyArray_DESCR((PyArrayObject*)ret->numpy_array)->elsize;
            ret->count = PyArray_DIMS((PyArrayObject*)ret->numpy_array)[0];
            ret->array_data = PyArray_DATA((PyArrayObject*)ret->numpy_array);
            // If pColO is a Masked array, we convert the mask to a NPY_BOOL numpy array     
            if (PyObject_HasAttrString(pColO, "mask")) {
                pMask = PyObject_GetAttrString(pColO, "mask");
                if (pMask != NULL) {
                    ret->numpy_mask = PyArray_FromAny(pMask, PyArray_DescrFromType(NPY_BOOL), 1, 1,  NPY_ARRAY_CARRAY, NULL);
                    if (ret->numpy_mask == NULL || PyArray_DIMS((PyArrayObject*)ret->numpy_mask)[0] != (int)ret->count)
                    {
                        PyErr_Clear();
                        pMask = NULL;
                        ret->numpy_mask = NULL;                            
                    }
                }
            }
            if (ret->numpy_mask != NULL) ret->mask_data = PyArray_DATA((PyArrayObject*)ret->numpy_mask); 
        }
    }
    return TRUE;
wrapup:
    *return_message = msg;
    return FALSE;
}

BAT *PyObject_ConvertToBAT(PyReturn *ret, int bat_type, int i, int seqbase, char **return_message)
{
    BAT *b = NULL;
    size_t index_offset = 0;
    char *msg;
    size_t iu;

    if (ret->multidimensional) index_offset = i;
    switch (bat_type) 
    {
    case TYPE_bte:
        NP_CREATE_BAT(b, bit);
        break;
    case TYPE_sht:
        NP_CREATE_BAT(b, sht);
        break;
    case TYPE_int:
        NP_CREATE_BAT(b, int);
        break;
    case TYPE_lng:
        NP_CREATE_BAT(b, lng);
        break;
    case TYPE_flt:
        NP_CREATE_BAT(b, flt);
        break;
    case TYPE_dbl:
        NP_CREATE_BAT(b, dbl);
        break;
#ifdef HAVE_HGE
    case TYPE_hge:
        NP_CREATE_BAT(b, hge);
        break;
#endif
    case TYPE_str:
        {
            bool *mask = NULL;   
            char *data = NULL;  
            char *utf8_string = NULL;
            if (ret->mask_data != NULL)   
            {   
                mask = (bool*)ret->mask_data;   
            }   
            if (ret->array_data == NULL)   
            {   
                msg = createException(MAL, "pyapi.eval", "No return value stored in the structure.  n");         
                goto wrapup;      
            }          
            data = (char*) ret->array_data;   

            if (ret->result_type != NPY_OBJECT) {
                utf8_string = GDKzalloc(256 + ret->memory_size + 1); 
                utf8_string[256 + ret->memory_size] = '\0';       
            }

            b = BATnew(TYPE_void, TYPE_str, ret->count, TRANSIENT);    
            BATseqbase(b, seqbase); b->T->nil = 0; b->T->nonil = 1;         
            b->tkey = 0; b->tsorted = 0; b->trevsorted = 0;
            switch(ret->result_type)                                                          
            {                                                                                 
                case NPY_BOOL:      NP_COL_BAT_STR_LOOP(b, bit, lng_to_string); break;
                case NPY_BYTE:      NP_COL_BAT_STR_LOOP(b, bte, lng_to_string); break;
                case NPY_SHORT:     NP_COL_BAT_STR_LOOP(b, sht, lng_to_string); break;
                case NPY_INT:       NP_COL_BAT_STR_LOOP(b, int, lng_to_string); break;
                case NPY_LONG:      
                case NPY_LONGLONG:  NP_COL_BAT_STR_LOOP(b, lng, lng_to_string); break;
                case NPY_UBYTE:     NP_COL_BAT_STR_LOOP(b, unsigned char, lng_to_string); break;
                case NPY_USHORT:    NP_COL_BAT_STR_LOOP(b, unsigned short, lng_to_string); break;
                case NPY_UINT:      NP_COL_BAT_STR_LOOP(b, unsigned int, lng_to_string); break;
                case NPY_ULONG:     NP_COL_BAT_STR_LOOP(b, unsigned long, lng_to_string); break;  
                case NPY_ULONGLONG: NP_COL_BAT_STR_LOOP(b, unsigned long long, lng_to_string); break;  
                case NPY_FLOAT16:                                                             
                case NPY_FLOAT:     NP_COL_BAT_STR_LOOP(b, flt, dbl_to_string); break;             
                case NPY_DOUBLE:                                                              
                case NPY_LONGDOUBLE: NP_COL_BAT_STR_LOOP(b, dbl, dbl_to_string); break;                  
                case NPY_STRING:    
                    for (iu = 0; iu < ret->count; iu++) {              
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {                                                           
                            b->T->nil = 1;    
                            BUNappend(b, str_nil, FALSE);                                                            
                        }  else {
                            if (!string_copy(&data[(index_offset * ret->count + iu) * ret->memory_size], utf8_string, ret->memory_size)) {
                                msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");       
                                goto wrapup;
                            }
                            BUNappend(b, utf8_string, FALSE); 
                        }                                                       
                    }    
                    break;
                case NPY_UNICODE:    
                    for (iu = 0; iu < ret->count; iu++) {              
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {                                                           
                            b->T->nil = 1;    
                            BUNappend(b, str_nil, FALSE);
                        }  else {
                            utf32_to_utf8(0, ret->memory_size / 4, utf8_string, (const Py_UNICODE*)(&data[(index_offset * ret->count + iu) * ret->memory_size]));
                            BUNappend(b, utf8_string, FALSE);
                        }                                                       
                    }    
                    break;
                case NPY_OBJECT:
                {
                    //The resulting array is an array of pointers to various python objects
                    //Because the python objects can be of any size, we need to allocate a different size utf8_string for every object
                    //we will first loop over all the objects to get the maximum size needed, so we only need to do one allocation
                    size_t utf8_size = 256;
                    for (iu = 0; iu < ret->count; iu++) {
                        size_t size = 256;
                        PyObject *obj;
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) continue;
                        obj = *((PyObject**) &data[(index_offset * ret->count + iu) * ret->memory_size]);
                        if (PyString_CheckExact(obj) || PyByteArray_CheckExact(obj)) {
                            size = Py_SIZE(obj);
                        } else if (PyUnicode_CheckExact(obj)) {
                            size = Py_SIZE(obj) * 4;
                        }
                        if (size > utf8_size) utf8_size = size;
                    }
                    utf8_string = GDKzalloc(utf8_size);
                    for (iu = 0; iu < ret->count; iu++) {          
                        if (mask != NULL && (mask[index_offset * ret->count + iu]) == TRUE) {                
                            b->T->nil = 1;    
                            BUNappend(b, str_nil, FALSE);
                        } else {
                            //we try to handle as many types as possible
                            PyObject *obj = *((PyObject**) &data[(index_offset * ret->count + iu) * ret->memory_size]);
                            if (PyString_CheckExact(obj)) {
                                char *str = ((PyStringObject*)obj)->ob_sval;
                                if (!string_copy(str, utf8_string, strlen(str) + 1)) {
                                    msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");       
                                    goto wrapup;    
                                }
                            } else if (PyByteArray_CheckExact(obj)) {
                                char *str = ((PyByteArrayObject*)obj)->ob_bytes;
                                if (!string_copy(str, utf8_string, strlen(str) + 1)) {
                                    msg = createException(MAL, "pyapi.eval", "Invalid string encoding used. Please return a regular ASCII string, or a Numpy_Unicode object.\n");       
                                    goto wrapup;    
                                }
                            } else if (PyUnicode_CheckExact(obj)) {
                                Py_UNICODE *str = (Py_UNICODE*)((PyUnicodeObject*)obj)->str;
                                utf32_to_utf8(0, ((PyUnicodeObject*)obj)->length, utf8_string, str);
                            } else if (PyBool_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj) || PyFloat_Check(obj)) { 
#ifdef HAVE_HGE
                                hge h;
                                py_to_hge(obj, &h);
                                hge_to_string(utf8_string, h);
#else
                                lng h;
                                py_to_lng(obj, &h);
                                lng_to_string(utf8_string, h);
#endif
                            } else {
                                msg = createException(MAL, "pyapi.eval", "Unrecognized Python object. Could not convert to NPY_UNICODE.\n");       
                                goto wrapup; 
                            }
                            BUNappend(b, utf8_string, FALSE); 
                        }                                                       
                    }
                    break;
                }
                default:
                    msg = createException(MAL, "pyapi.eval", "Unrecognized type. Could not convert to NPY_UNICODE.\n");       
                    goto wrapup;    
            }                           
            GDKfree(utf8_string);   
                                                
            b->T->nonil = 1 - b->T->nil;                                                  
            BATsetcount(b, ret->count);                                                     
            BATsettrivprop(b); 
            break;
        }
    default:
        msg = createException(MAL, "pyapi.eval", "Unrecognized BAT type %s.\n", BatType_Format(bat_type));       
        goto wrapup; 
    }
    return b;
wrapup:
    *return_message = msg;
    return NULL;
}


int PyType_ToBat(int type)
{
    switch (type)
    {
        case NPY_BOOL: return TYPE_bit;
        case NPY_BYTE: return TYPE_bte;
        case NPY_SHORT: return TYPE_sht;
        case NPY_INT: return TYPE_int;
        case NPY_LONG: 
        case NPY_LONGLONG: return TYPE_lng;
        case NPY_UBYTE:
        case NPY_USHORT: 
        case NPY_UINT:
        case NPY_ULONG: 
        case NPY_ULONGLONG: return TYPE_void;
        case NPY_FLOAT16: 
        case NPY_FLOAT: return TYPE_flt;
        case NPY_DOUBLE: 
        case NPY_LONGDOUBLE: return TYPE_dbl;
        case NPY_STRING: return TYPE_str;
        case NPY_UNICODE: return TYPE_str;
        default: return TYPE_void;
    }
}



bool PyType_IsPandasDataFrame(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<class 'pandas.core.frame.DataFrame'>") == 0;
    Py_DECREF(str);
    return ret;
}

bool PyType_IsNumpyArray(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<type 'numpy.ndarray'>") == 0;
    Py_DECREF(str);
    return ret;
}

bool PyType_IsNumpyMaskedArray(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<class 'numpy.ma.core.MaskedArray'>") == 0;
    Py_DECREF(str);
    return ret;
}

bool PyType_IsLazyArray(PyObject *object)
{
    PyObject *str = PyObject_Str(PyObject_Type(object));
    bool ret = strcmp(PyString_AsString(str), "<class 'lazyarray'>") == 0;
    Py_DECREF(str);
    return ret;
}


char *BatType_Format(int type)
{
    switch (type)
    {
        case TYPE_bit: return "BIT";
        case TYPE_bte: return "BYTE";
        case TYPE_sht: return "SHORT";
        case TYPE_int: return "INT";
        case TYPE_lng: return "LONG";
        case TYPE_flt: return "FLOAT";
        case TYPE_dbl: return "DOUBLE";
        case TYPE_str: return "STRING";
        case TYPE_hge: return "HUGE";
        case TYPE_oid: return "OID";
        default: return "UNKNOWN";
    }
}


char *PyType_Format(int type)
{
    switch (type)
    {
        case NPY_BOOL: return "BOOL";
        case NPY_BYTE: return "BYTE";
        case NPY_SHORT: return "SHORT";
        case NPY_INT: return "INT";
        case NPY_LONG: return "LONG";
        case NPY_LONGLONG: return "LONG LONG";
        case NPY_UBYTE: return "UNSIGNED BYTE";
        case NPY_USHORT: return "UNSIGNED SHORT";
        case NPY_UINT: return "UNSIGNED INT";
        case NPY_ULONG: return "UNSIGNED LONG";
        case NPY_ULONGLONG: return "UNSIGNED LONG LONG";
        case NPY_FLOAT16: return "HALF-FLOAT (FLOAT16)";
        case NPY_FLOAT: return "FLOAT";
        case NPY_DOUBLE: return "DOUBLE";
        case NPY_LONGDOUBLE: return "LONG DOUBLE";
        case NPY_COMPLEX64: return "COMPLEX FLOAT";
        case NPY_COMPLEX128: return "COMPLEX DOUBLE";
        case NPY_CLONGDOUBLE: return "COMPLEX LONG DOUBLE";
        case NPY_DATETIME: return "DATETIME";
        case NPY_TIMEDELTA: return "TIMEDELTA";
        case NPY_STRING: return "STRING";
        case NPY_UNICODE: return "UNICODE STRING";
        case NPY_OBJECT: return "PYTHON OBJECT";
        case NPY_VOID: return "VOID";
        default: return "UNKNOWN";
    }
}

bool utf32_to_lng(Py_UNICODE *utf32, size_t maxsize, lng *value)
{
    size_t length = utf32_strlen(utf32);
    int i;
    size_t factor = 1;
    if (length > maxsize) length = maxsize;
    i = length;
    *value = 0;
    for( ; i >= 0; i--)
    {
        switch(utf32[i])
        {
            case '0': break;
            case '1': *value += factor; break;
            case '2': *value += 2 * factor; break;
            case '3': *value += 3 * factor; break;
            case '4': *value += 4 * factor; break;
            case '5': *value += 5 * factor; break;
            case '6': *value += 6 * factor; break;
            case '7': *value += 7 * factor; break;
            case '8': *value += 8 * factor; break;
            case '9': *value += 9 * factor; break;
            case '-': *value *= -1; break;
            case '.':
            case ',': *value = 0; factor = 1; continue;
            case '\0': continue;
            default: return false;
        }
        factor *= 10;
    }
    return true;
}

bool utf32_to_dbl(Py_UNICODE *utf32, size_t maxsize, dbl *value)
{
    size_t length = utf32_strlen(utf32);
    int i;
    size_t factor = 1;
    if (length > maxsize) length = maxsize;
    i = length;
    *value = 0;
    for( ; i >= 0; i--)
    {
        switch(utf32[i])
        {
            case '0': break;
            case '1': *value += factor; break;
            case '2': *value += 2 * factor; break;
            case '3': *value += 3 * factor; break;
            case '4': *value += 4 * factor; break;
            case '5': *value += 5 * factor; break;
            case '6': *value += 6 * factor; break;
            case '7': *value += 7 * factor; break;
            case '8': *value += 8 * factor; break;
            case '9': *value += 9 * factor; break;
            case '-': *value *= -1; break;
            case '.':
            case ',': *value /= factor; factor = 1; continue;
            case '\0': continue;
            default: return false;
        }
        factor *= 10;
    }
    return true;
}

#ifdef HAVE_HGE
bool utf32_to_hge(Py_UNICODE *utf32, size_t maxsize, hge *value)
{
    size_t length = utf32_strlen(utf32) + 1;    
    char utf8[200];
    if (length > maxsize) length = maxsize;
    utf32_to_utf8(0, maxsize, utf8, utf32);
    return s_to_hge(utf8, length, value);
}
#endif


//py_to_hge and py_to_lng are almost identical, so use a generic #define to make them
#define PY_TO_(type)                                                                                             \
bool py_to_##type(PyObject *ptr, type *value)                                                                    \
{                                                                                                                \
    if (PyLong_CheckExact(ptr)) {                                                                                     \
        PyLongObject *p = (PyLongObject*) ptr;                                                                   \
        type h = 0;                                                                                              \
        type prev = 0;                                                                                           \
        int i = Py_SIZE(p);                                                                                      \
        int sign = i < 0 ? -1 : 1;                                                                               \
        i *= sign;                                                                                               \
        while (--i >= 0) {                                                                                       \
            prev = h; (void)prev;                                                                                \
            h = (h << PyLong_SHIFT) + p->ob_digit[i];                                                            \
            if ((h >> PyLong_SHIFT) != prev) {                                                                   \
                printf("Overflow!\n");                                                                           \
                return false;                                                                                    \
            }                                                                                                    \
        }                                                                                                        \
        *value = h * sign;                                                                                       \
        return true;                                                                                             \
    } else if (PyInt_CheckExact(ptr) || PyBool_Check(ptr)) {                                                          \
        *value = (type)((PyIntObject*)ptr)->ob_ival;                                                             \
        return true;                                                                                             \
    } else if (PyFloat_CheckExact(ptr)) {                                                                             \
        *value = (type) ((PyFloatObject*)ptr)->ob_fval;                                                          \
        return true;                                                                                             \
    } else if (PyString_CheckExact(ptr)) {                                                                            \
        return s_to_##type(((PyStringObject*)ptr)->ob_sval, strlen(((PyStringObject*)ptr)->ob_sval), value);     \
    }  else if (PyByteArray_CheckExact(ptr)) {                                                                        \
        return s_to_##type(((PyByteArrayObject*)ptr)->ob_bytes, strlen(((PyByteArrayObject*)ptr)->ob_bytes), value);\
    } else if (PyUnicode_CheckExact(ptr)) {                                                                           \
        return utf32_to_##type(((PyUnicodeObject*)ptr)->str, 64, value);                                             \
    }                                                                                                            \
    return false;                                                                                                \
}

PY_TO_(lng);
#ifdef HAVE_HGE
PY_TO_(hge);
#endif


void dbl_to_string(char* str, dbl value)
{
    snprintf(str, 256, "%lf", value);
}

bool string_copy(char * source, char* dest, size_t max_size)
{
    size_t i;
    for(i = 0; i < max_size; i++)
    {
        dest[i] = source[i];
        if (dest[i] == 0) return TRUE;
        if ((*(unsigned char*)&source[i]) >= 128) return FALSE;
    }
    dest[max_size] = '\0';
    return TRUE;
}


#ifdef HAVE_HGE
int hge_to_string(char * str, hge x)
{
    int i = 0;
    size_t size = 1;
    hge cpy = x > 0 ? x : -x;
    while(cpy > 0) {
        cpy /= 10;
        size++;
    }
    if (x < 0) size++;
    if (x < 0) 
    {
        x *= -1;
        str[0] = '-';
    }
    str[size - 1] = '\0';
    i = size - 1;
    while(x > 0)
    {
        int v = x % 10;
        i--;
        if (i < 0) return FALSE;
        if (v == 0)       str[i] = '0';
        else if (v == 1)  str[i] = '1';
        else if (v == 2)  str[i] = '2';
        else if (v == 3)  str[i] = '3';
        else if (v == 4)  str[i] = '4';
        else if (v == 5)  str[i] = '5';
        else if (v == 6)  str[i] = '6';
        else if (v == 7)  str[i] = '7';
        else if (v == 8)  str[i] = '8';
        else if (v == 9)  str[i] = '9';
        x = x / 10;
    }

    return TRUE;
}
#endif

int utf32_strlen(const Py_UNICODE *utf32_str)
{
	size_t i = 0;
	while(utf32_str[i] != 0)
		i++;
	return (i - 1);
}


int utf32_char_to_utf8_char(size_t position, char *utf8_storage, Py_UNICODE utf32_char)
{
    int utf8_size = 4;
    if      (utf32_char < 0x80)        utf8_size = 1;
    else if (utf32_char < 0x800)       utf8_size = 2;
#if Py_UNICODE_SIZE >= 4
    else if (utf32_char < 0x10000)     utf8_size = 3;
    else if (utf32_char > 0x0010FFFF)  return -1; //utf32 character is out of legal range
#endif
    
    switch(utf8_size)
    {
        case 4:
            utf8_storage[position + 3] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 2] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11110000);
            return utf8_size;
        case 3:
            utf8_storage[position + 2] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11100000);
            return utf8_size;
        case 2:
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11000000);
            return utf8_size;
        default:
            utf8_storage[position]     = utf32_char;
            return utf8_size;
    }
}

bool utf32_to_utf8(size_t offset, size_t size, char *utf8_storage, const Py_UNICODE *utf32)
{
    size_t i = 0;
    int position = 0;
    int shift;
    for(i = 0; i < size; i++)
    {
        if (utf32[offset + i] == 0) 
        {
            utf8_storage[position] = '\0';
            return true;
        }

        shift = utf32_char_to_utf8_char(position, utf8_storage, utf32[offset + i]);
        if (shift < 0) return false;
        position += shift;
    }
    utf8_storage[position] = '\0';
    return true;
}

bool py_to_dbl(PyObject *ptr, dbl *value)
{
    if (PyFloat_Check(ptr)) {
        *value = ((PyFloatObject*)ptr)->ob_fval;
    } else {
#ifdef HAVE_HGE
        hge h;
        if (!py_to_hge(ptr, &h)) {
            return false;
        }
        *value = (dbl) h;
#else
        lng l;
        if (!py_to_lng(ptr, &l)) {
            return false;
        }
        *value = (dbl) l;
#endif
        return true;
    }
    return false;
}

//Returns true if the type of [object] is a scalar (i.e. numeric scalar or string, basically "not an array but a single value")
bool PyType_IsPyScalar(PyObject *object)
{
    PyArray_Descr *descr;

    if (object == NULL) return false;
    if (PyList_Check(object)) return false;
    if (PyObject_HasAttrString(object, "mask")) return false;

    descr = PyArray_DescrFromScalar(object);
    if (descr == NULL) return false;
    if (descr->type_num != NPY_OBJECT) return true; //check if the object is a numpy scalar
    if (PyInt_Check(object) || PyFloat_Check(object) || PyLong_Check(object) || PyString_Check(object) || PyBool_Check(object) || PyUnicode_Check(object) || PyByteArray_Check(object)) return true;

    return false;
}
