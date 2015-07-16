
#include "lazyarray.h"
#include "pytypes.h"
#include "type_conversion.h"

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code, so hopefully this works
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>


void lazyarray_init(void)
{
    import_array();
}

#define LAZYARRAY_BAT(op) ((BAT*)((PyLazyArrayObject*)op)->bat)
#define LAZYARRAY_GETNUMERICITEM(op, tpe, index) ((tpe*)LAZYARRAY_BAT(op)->T->heap.base)[index]
#define LAZYARRAY_GETITEMPTR(op, index) ((LAZYARRAY_BAT(op)->T->heap.base)[index * LAZYARRAY_BAT(op)->T->width])

char* PyLazyArray_GetString(PyLazyArrayObject *a, Py_ssize_t index)
{
    return LAZYARRAY_BAT(a)->T->vheap->base + VarHeapVal(LAZYARRAY_BAT(a)->T->heap.base, index, LAZYARRAY_BAT(a)->T->width);
}

hge PyLazyArray_GetHuge(PyLazyArrayObject *a, Py_ssize_t index)
{
    return ((hge*)LAZYARRAY_BAT(a)->T->heap.base)[index];
}

bool PyLazyArray_IsNil(PyLazyArrayObject *a, Py_ssize_t index)
{
    if (LAZYARRAY_BAT(a)->T->nonil) return false;

    if (PyLazyArray_GET_TYPE(a) == TYPE_str) {
        return strcmp(PyLazyArray_GetString(a, index), str_nil) == 0;
    }
    else {
        const void *nil = ATOMnilptr(LAZYARRAY_BAT(a)->ttype);
        int (*atomcmp)(const void *, const void *) = ATOMcompare(LAZYARRAY_BAT(a)->ttype);
        return (*atomcmp)(&LAZYARRAY_GETITEMPTR(a, index), nil) == 0;
    }
}

PyObject* PyLazyArray_GetItem(PyLazyArrayObject *a, Py_ssize_t index)
{
    if (a->array != NULL && a->array[index] != NULL) {
        Py_INCREF(a->array[index]);
        return a->array[index];
    }
    if (a->array == NULL) {
        a->array = GDKzalloc(sizeof(PyObject*) * Py_SIZE(a));
    }

    if (PyLazyArray_IsNil(a, index)) {
        a->array[index] = Py_None;
        Py_RETURN_NONE;
    }

    if (PyLazyArray_GET_TYPE(a) == TYPE_str) {
        PyObject *obj = PyUnicode_FromString(PyLazyArray_GetString(a, index));
        a->array[index] = obj;
        Py_INCREF(obj);
        return obj;
    } 
    {
        PyObject *obj = NULL;
        switch (PyLazyArray_GET_TYPE(a))
        {
            case TYPE_bte: obj = SCALAR_TO_PYSCALAR(bte, LAZYARRAY_GETNUMERICITEM(a, bte, index)); break; 
            case TYPE_bit: obj = SCALAR_TO_PYSCALAR(bit, LAZYARRAY_GETNUMERICITEM(a, bit, index)); break;
            case TYPE_sht: obj = SCALAR_TO_PYSCALAR(sht, LAZYARRAY_GETNUMERICITEM(a, sht, index)); break;
            case TYPE_int: obj = SCALAR_TO_PYSCALAR(int, LAZYARRAY_GETNUMERICITEM(a, int, index)); break;
            case TYPE_lng: obj = SCALAR_TO_PYSCALAR(lng, LAZYARRAY_GETNUMERICITEM(a, lng, index)); break;
            case TYPE_flt: obj = SCALAR_TO_PYSCALAR(flt, LAZYARRAY_GETNUMERICITEM(a, flt, index)); break;
            case TYPE_dbl: obj = SCALAR_TO_PYSCALAR(dbl, LAZYARRAY_GETNUMERICITEM(a, dbl, index)); break;
            case TYPE_hge: obj = PyLong_FromHge(PyLazyArray_GetHuge(a, index)); break;
        }
        a->array[index] = obj;
        Py_INCREF(obj);
        return obj;
    }
}


static void
lazyarray_dealloc(PyLazyArrayObject *op)
{
    Py_ssize_t i;
    if (op->array != NULL) 
    {
        i = Py_SIZE(op);
        while (--i >= 0) 
        {
            Py_XDECREF(op->array[i]);
        }
        GDKfree(op->array);
    }
    Py_XDECREF(op->numpy_array);
    Py_TYPE(op)->tp_free((PyObject *)op);
}


static int
lazyarray_print(PyLazyArrayObject *op, FILE *fp, int flags)
{
    (void) op; (void) flags;
    fprintf(fp, "[");
    //todo
    fprintf(fp, "]");
    return 0;
}


static PyObject *
lazyarray_repr(PyLazyArrayObject *v)
{
    PyObject *s, *temp;
    (void) v;
    s = PyString_FromString("[");
    //todo
    temp = PyString_FromString("]");
    PyString_ConcatAndDel(&s, temp);
    return s;
}

static Py_ssize_t
lazyarray_length(PyLazyArrayObject *a)
{
    return Py_SIZE(a);
}

static PyObject *
lazyarray_slice(PyLazyArrayObject *a, Py_ssize_t ilow, Py_ssize_t ihigh)
{
    PyListObject *np;
    PyObject **dest;
    Py_ssize_t i, len;
    if (ilow < 0)
        ilow = 0;
    else if (ilow > Py_SIZE(a))
        ilow = Py_SIZE(a);
    if (ihigh < ilow)
        ihigh = ilow;
    else if (ihigh > Py_SIZE(a))
        ihigh = Py_SIZE(a);
    len = ihigh - ilow;
    np = (PyListObject *) PyList_New(len);
    if (np == NULL)
        return NULL;

    dest = np->ob_item;
    for (i = 0; i < len; i++) {
        PyObject *v = PyLazyArray_GetItem(a, i);
        dest[i] = v;
    }
    return (PyObject *)np;
}

static int
lazyarray_contains(PyLazyArrayObject *a, PyObject *sub_obj)
{
    (void) sub_obj;
    //todo
    if (PyLazyArray_GET_TYPE(a) == TYPE_str) {
        return 0;
    } 
    if (PyLazyArray_GET_TYPE(a) == TYPE_hge) {
        return 0;
    }
    return -1;
}


#define COPY_BINARY_NUMPY_FUNCTION(name) static PyObject *lazyarray_##name(PyLazyArrayObject *m1, PyObject *m2) { return PyArray_Type.tp_as_number->nb_##name(PyLazyArray_AsNumpyArray(m1, 0, Py_SIZE(m1)), m2); }

COPY_BINARY_NUMPY_FUNCTION(add);
COPY_BINARY_NUMPY_FUNCTION(subtract);
COPY_BINARY_NUMPY_FUNCTION(multiply);
COPY_BINARY_NUMPY_FUNCTION(divide);
COPY_BINARY_NUMPY_FUNCTION(remainder);
COPY_BINARY_NUMPY_FUNCTION(divmod);
COPY_BINARY_NUMPY_FUNCTION(lshift);
COPY_BINARY_NUMPY_FUNCTION(rshift);
COPY_BINARY_NUMPY_FUNCTION(and);
COPY_BINARY_NUMPY_FUNCTION(xor);
COPY_BINARY_NUMPY_FUNCTION(or);
COPY_BINARY_NUMPY_FUNCTION(inplace_add);
COPY_BINARY_NUMPY_FUNCTION(inplace_subtract);
COPY_BINARY_NUMPY_FUNCTION(inplace_multiply);
COPY_BINARY_NUMPY_FUNCTION(inplace_divide);


#define COPY_UNARY_NUMPY_FUNCTION(name) static PyObject *lazyarray_##name(PyLazyArrayObject *v) { return PyArray_Type.tp_as_number->nb_##name(PyLazyArray_AsNumpyArray(v, 0, Py_SIZE(v))); }

COPY_UNARY_NUMPY_FUNCTION(int);
COPY_UNARY_NUMPY_FUNCTION(long);
COPY_UNARY_NUMPY_FUNCTION(float);
COPY_UNARY_NUMPY_FUNCTION(oct);
COPY_UNARY_NUMPY_FUNCTION(hex);
COPY_UNARY_NUMPY_FUNCTION(negative);
COPY_UNARY_NUMPY_FUNCTION(positive);
COPY_UNARY_NUMPY_FUNCTION(absolute);
COPY_UNARY_NUMPY_FUNCTION(invert);
// static PyObject *
// lazyarray_multiply(PyLazyArrayObject *m1, PyObject *m2)
// {
//     return PyArray_Type.tp_as_number->nb_multiply(PyLazyArray_AsNumpyArray(m1, 0, Py_SIZE(m1)), m2);
// }

// static PyObject *
// lazyarray_inplace_multiply(PyLazyArrayObject *m1, PyObject *m2)
// {
//     return PyArray_Type.tp_as_number->nb_inplace_multiply(PyLazyArray_AsNumpyArray(m1, 0, Py_SIZE(m1)), m2);
// }

static PyNumberMethods lazyarray_as_number = {
    (binaryfunc)lazyarray_add,                               /*nb_add*/
    (binaryfunc)lazyarray_subtract,                               /*nb_subtract*/
    (binaryfunc)lazyarray_multiply,  /*nb_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)lazyarray_divide,                               /*nb_divide*/
#endif
    (binaryfunc)lazyarray_remainder,                               /*nb_remainder*/
    (binaryfunc)lazyarray_divmod,                               /*nb_divmod*/
    0,                               /*nb_power*/
    (unaryfunc)lazyarray_negative,                               /*nb_neg*/
    (unaryfunc)lazyarray_positive,                               /*nb_pos*/
    (unaryfunc)lazyarray_absolute,                               /*(unaryfunc)array_abs,*/
    0,                               /*nb_nonzero*/
    (unaryfunc)lazyarray_invert,                               /*nb_invert*/
    (binaryfunc)lazyarray_lshift,                               /*nb_lshift*/
    (binaryfunc)lazyarray_rshift,                               /*nb_rshift*/
    (binaryfunc)lazyarray_and,                               /*nb_and*/
    (binaryfunc)lazyarray_xor,                               /*nb_xor*/
    (binaryfunc)lazyarray_or,                               /*nb_or*/
#if !defined(NPY_PY3K)
    0,                               /*nb_coerce*/
#endif
    (unaryfunc)lazyarray_int,                               /*nb_int*/
#if defined(NPY_PY3K)
    0,                               /*nb_reserved*/
#else
    (unaryfunc)lazyarray_long,                               /*nb_long*/
#endif
    (unaryfunc)lazyarray_float,                               /*nb_float*/
#if !defined(NPY_PY3K)
    (unaryfunc)lazyarray_oct,                               /*nb_oct*/
    (unaryfunc)lazyarray_hex,                               /*nb_hex*/
#endif
    (binaryfunc)lazyarray_inplace_add,                               /*inplace_add*/
    (binaryfunc)lazyarray_inplace_subtract,                               /*inplace_subtract*/
    (binaryfunc)lazyarray_inplace_multiply,         /*inplace_multiply*/
#if !defined(NPY_PY3K)
    (binaryfunc)lazyarray_inplace_divide,                               /*inplace_divide*/
#endif
    0,                               /*inplace_remainder*/
    0,                               /*inplace_power*/
    0,                               /*inplace_lshift*/
    0,                               /*inplace_rshift*/
    0,                               /*inplace_and*/
    0,                               /*inplace_xor*/
    0,                               /*inplace_or*/

    0,                               /*nb_floor_divide*/
    0,                               /*nb_true_divide*/
    0,                               /*nb_inplace_floor_divide*/
    0,                               /*nb_inplace_true_divide*/
    0,                               /*nb_index */
#if PY_VERSION_HEX >= 0x03050000
    0,                               /*nb_matrix_multiply*/
    0,                               /*nb_inplacematrix_multiply*/
#endif
};

static PyObject *
lazyarray_item(PyLazyArrayObject *a, register Py_ssize_t i)
{
    if (i < 0 || i >= Py_SIZE(a)) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return NULL;
    }
    return PyLazyArray_GetItem(a, i);
}

static PySequenceMethods lazyarray_as_sequence = {
    (lenfunc)lazyarray_length,          /*sq_length*/
    0, //(binaryfunc)stringref_concat,       /*sq_concat*/
    0, //(ssizeargfunc)stringref_repeat,     /*sq_repeat*/
    (ssizeargfunc)lazyarray_item,       /*sq_item*/
    (ssizessizeargfunc)lazyarray_slice, /*sq_slice*/
    0,                                  /*sq_ass_item*/
    0,                                  /*sq_ass_slice*/
    (objobjproc)lazyarray_contains,      /*sq_contains*/
    0,
    0
};

static PyObject *
lazyarray_subscript(PyLazyArrayObject* self, PyObject* item)
{
    if (PyIndex_Check(item)) {
        Py_ssize_t i;
        i = PyNumber_AsSsize_t(item, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred())
            return NULL;
        if (i < 0)
            i += PyList_GET_SIZE(self);
        return lazyarray_item(self, i);
    }
    else if (PySlice_Check(item)) {
        Py_ssize_t start, stop, step, slicelength, cur, i;
        PyObject* result;
        PyObject* it;
        PyObject **dest;

        if (PySlice_GetIndicesEx((PySliceObject*)item, Py_SIZE(self),
                         &start, &stop, &step, &slicelength) < 0) {
            return NULL;
        }

        if (slicelength <= 0) {
            return PyList_New(0);
        }
        else if (step == 1) {
            return lazyarray_slice(self, start, stop);
        }
        else {
            result = PyList_New(slicelength);
            if (!result) return NULL;

            dest = ((PyListObject *)result)->ob_item;
            for (cur = start, i = 0; i < slicelength;
                 cur += step, i++) {
                it = PyLazyArray_GetItem(self, i);
                dest[i] = it;
            }

            return result;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "list indices must be integers, not %.200s",
                     item->ob_type->tp_name);
        return NULL;
    }
}

static PyMappingMethods lazyarray_as_mapping = {
    (lenfunc)lazyarray_length,
    (binaryfunc)lazyarray_subscript,
    0,
};

static int
lazyarray_traverse(PyLazyArrayObject *o, visitproc visit, void *arg)
{
    Py_ssize_t i;

    for (i = Py_SIZE(o); --i >= 0; )
        Py_VISIT(PyLazyArray_GetItem(o, i));
    return 0;
}

PyDoc_STRVAR(getitem_doc,
"x.__getitem__(y) <==> x[y]");
PyDoc_STRVAR(materialize_doc,
"L.materialize() -- Populate the LazyArray with Python Objects created from the BAT.");
PyDoc_STRVAR(ismaterialized_doc,
"L.ismaterialized() -- Returns true if the lazy array is fully materialized, false otherwise.");
PyDoc_STRVAR(asnumpyarray_doc,
"L.asnumpyarray() -- Creates a numpy array from the lazy array. If L.hasnil() is True, returns a numpy masked array, otherwise returns a numpy array.");
PyDoc_STRVAR(aslist_doc,
"L.aslist() -- Creates a Python list from the array. If L.hasnil() is true, throws an error.");
PyDoc_STRVAR(hasnil_doc,
"L.hasnil() -- Returns True if the BAT has a nill value in it, or false otherwise.");
PyDoc_STRVAR(type_doc,
"L.type() -- Returns the type of the BAT as a string. Either \"String\" or \"Huge\".");

static PyObject*
lazyarray_materialize(PyLazyArrayObject *seq, PyObject *unused)
{
    Py_ssize_t i;
    (void) unused;
    for (i = Py_SIZE(seq); --i >= 0; ) {
        PyLazyArray_GetItem(seq, i);
    }

    return (PyObject*)seq;
}

static PyObject*
lazyarray_ismaterialized(PyLazyArrayObject *a, PyObject *unused)
{
    Py_ssize_t i;
    (void) unused;
    if (a->array == NULL) Py_RETURN_FALSE;

    for (i = Py_SIZE(a); --i >= 0; ) {
        if (a->array[i] == NULL) {
            Py_RETURN_FALSE;
        }
    }
    Py_RETURN_TRUE;
}

PyObject *PyLazyArray_AsNumpyArray(PyLazyArrayObject *a, size_t start, size_t end)
{
    if (a->numpy_array != NULL) {
        Py_INCREF(a->numpy_array);
        return a->numpy_array; 
    } else {
        PyObject *result;
        char *msg = NULL;
        PyInput input;

        input.bat = LAZYARRAY_BAT(a);
        input.count = BATcount(LAZYARRAY_BAT(a));
        input.bat_type = PyLazyArray_GET_TYPE(a);
        result = PyArrayObject_FromBAT(&input, start, end, &msg);

        if (result == NULL) {
            PyErr_Format(PyExc_Exception, "%s", msg);
            return NULL;
        }
        a->numpy_array = result;
        Py_INCREF(result);
        return result;
    }
}
static PyObject*
lazyarray_asnumpyarray(PyLazyArrayObject *a, PyObject *unused)
{
   (void) unused;
   return PyLazyArray_AsNumpyArray(a, 0, Py_SIZE(a));
}

static PyObject*
lazyarray_aslist(PyLazyArrayObject *a, PyObject *unused)
{
    (void) unused; (void) a;
    PyErr_Format(PyExc_TypeError, "Not implemented yet");
    return NULL;
}

static PyObject*
lazyarray_hasnil(PyLazyArrayObject *a, PyObject *unused)
{
    (void) unused;
    if (LAZYARRAY_BAT(a)->T->nil) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject*
lazyarray_type(PyLazyArrayObject *a, PyObject *unused)
{
    (void) unused;
    return PyString_FromString(BatType_Format(PyLazyArray_GET_TYPE(a)));
}

static PyObject* 
lazyarray_getattr(PyObject *o, PyObject *attr_name)
{
    if (PyString_Check(attr_name) && _PyString_Eq(attr_name, PyString_FromString("mask"))) {
        return PyNullMask_FromBAT(LAZYARRAY_BAT(o) ,0, Py_SIZE(o));
    }

    return PyObject_GenericGetAttr((PyObject*)o, attr_name);
}

static PyMethodDef lazyarray_methods[] = {
    {"__getitem__", (PyCFunction)lazyarray_subscript, METH_NOARGS, getitem_doc},
    {"materialize",(PyCFunction)lazyarray_materialize, METH_NOARGS, materialize_doc},
    {"ismaterialized",(PyCFunction)lazyarray_ismaterialized, METH_NOARGS, ismaterialized_doc},
    {"asnumpyarray",(PyCFunction)lazyarray_asnumpyarray, METH_NOARGS, asnumpyarray_doc},
    {"aslist",(PyCFunction)lazyarray_aslist, METH_NOARGS, aslist_doc},
    {"hasnil",(PyCFunction)lazyarray_hasnil, METH_NOARGS, hasnil_doc},
    {"type",(PyCFunction)lazyarray_type, METH_NOARGS, type_doc},
    {NULL,              NULL, 0, NULL}           /* sentinel */
};

PyTypeObject PyLazyArray_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "lazyarray",
    sizeof(PyLazyArrayObject),
    0,
    (destructor)lazyarray_dealloc,                   /* tp_dealloc */
    (printfunc)lazyarray_print,                      /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    (reprfunc)lazyarray_repr,                        /* tp_repr */
    &lazyarray_as_number,                                          /* tp_as_number */
    &lazyarray_as_sequence,                          /* tp_as_sequence */
    &lazyarray_as_mapping,                           /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    lazyarray_getattr,                    /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT      | Py_TPFLAGS_CHECKTYPES
     | Py_TPFLAGS_BASETYPE,         /* tp_flags */
    0,//lazyarray_doc,                                   /* tp_doc */
    (traverseproc)lazyarray_traverse,                /* tp_traverse */
    0,//(inquiry)lazyarray_clear,                        /* tp_clear */
    0,//list_richcompare,                           /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,//lazyarray_iter,                                  /* tp_iter */
    0,                                          /* tp_iternext */
    lazyarray_methods,                               /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,//(initproc)list_init,                        /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    PyType_GenericNew,                          /* tp_new */
    PyObject_Del,                            /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};

PyObject *PyLazyArray_FromBAT(BAT *b)
{
    register PyLazyArrayObject *op;
    size_t size = BATcount((BAT*)b);

    op = (PyLazyArrayObject *)PyObject_MALLOC(sizeof(PyLazyArrayObject));
    if (op == NULL)
        return PyErr_NoMemory();
    PyObject_INIT_VAR(op, &PyLazyArray_Type, size);
    op->bat = b;
    op->array = NULL;
    op->numpy_array = NULL;

    return (PyObject*) op;
}
