
#include "lazyarray.h"
#include "pytypes.h"
#include "type_conversion.h"

// Numpy Library
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef __INTEL_COMPILER
// Intel compiler complains about trailing comma's in numpy source code
#pragma warning(disable:271)
#endif
#include <numpy/arrayobject.h>

PyTypeObject *_maskedarray_type = NULL;
PyTypeObject *_lazyarray_type = NULL;
size_t _lazyarray_size = -1;

#define PyBatObject_GETBAT(op) ((PyBATObject*)op)->bat
#define PyBatObject_GETTYPE(op)  ATOMstorage(getColumnType(PyBatObject_GETBAT(op)->T->type))

static PyObject*
lazyarray_materialize(PyObject *lazyarray, PyObject *unused)
{
    (void) unused;
    if (PyObject_GetAttrString(lazyarray, "materialized") == Py_True) {
        PyErr_Format(PyExc_Exception, "lazyarray has already been materialized!");
        Py_RETURN_NONE;
    } else {
        PyObject *bat = PyObject_GetAttrString(lazyarray, "bat");
        if (bat == NULL || !PyBAT_CheckExact(bat) || PyBatObject_GETBAT(bat) == NULL) {
            PyErr_Format(PyExc_Exception, "lazyarray does not have a BAT to load data from!");
            Py_RETURN_NONE;
        } else { 
            PyObject *array;
            PyObject *mask;
            char *msg;
            PyInput input;
            input.bat = PyBatObject_GETBAT(bat);
            input.count = BATcount(PyBatObject_GETBAT(bat));
            input.bat_type = PyBatObject_GETTYPE(bat);
            input.scalar = false;

            array = (PyObject*)PyArrayObject_FromBAT(&input, 0, input.count, &msg, false);
            if (array == NULL) {
                PyErr_Format(PyExc_Exception, "%s", msg);
                Py_RETURN_NONE;
            }
            ((PyArrayObject_fields*)lazyarray)->data = ((PyArrayObject_fields*)array)->data;
            ((PyArrayObject_fields*)lazyarray)->nd = ((PyArrayObject_fields*)array)->nd;
            ((PyArrayObject_fields*)lazyarray)->dimensions = ((PyArrayObject_fields*)array)->dimensions;
            ((PyArrayObject_fields*)lazyarray)->strides = ((PyArrayObject_fields*)array)->strides;
            ((PyArrayObject_fields*)lazyarray)->base = ((PyArrayObject_fields*)array)->base;
            ((PyArrayObject_fields*)lazyarray)->descr = ((PyArrayObject_fields*)array)->descr;
            ((PyArrayObject_fields*)lazyarray)->flags = ((PyArrayObject_fields*)array)->flags;
            ((PyArrayObject_fields*)lazyarray)->weakreflist = ((PyArrayObject_fields*)array)->weakreflist;

            PyObject_SetAttrString(lazyarray, "materialized", Py_True);

            mask = PyNullMask_FromBAT(input.bat, 0, input.count);
            PyObject_SetAttrString(lazyarray, "hasnil", Py_True);
            PyObject_SetAttrString(lazyarray, "_mask", mask);
            PyErr_Print();
        }
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(materialize_doc,
"L.materialize() -- Populate the LazyArray with Python Objects created from the BAT.");

static PyMethodDef lazyarray_methods[] = {
    {"materialize",(PyCFunction)lazyarray_materialize, METH_NOARGS, materialize_doc},
    {NULL,              NULL, 0, NULL}           /* sentinel */
};

char lazyarray_code[] = "\n\
class LazyArray(numpy.ma.core.MaskedArray):\n\
    materialized = False\n\
    __mask = numpy.ma.core.nomask\n\
    def materialize(self):\n\
        self.materialized = True\n\
    @property\n\
    def data(self):\n\
        return self._data\n\
    @property\n\
    def _data(self):\n\
        if not self.materialized: self.materialize()\n\
        return self._get_data()\n\
    @property\n\
    def shape(self):\n\
        if not self.materialized: self.materialize()\n\
        return super(numpy.ma.core.MaskedArray, self).shape\n\
    @property\n\
    def _mask(self):\n\
        return self._mask\n\
    @property\n\
    def _mask(self):\n\
        if not self.materialized: self.materialize()\n\
        return self.__mask\n\
    @_mask.setter\n\
    def _mask(self, value):\n\
        self.__mask = value\n\
    def __getitem__(self, index):\n\
        if not self.materialized: self.materialize()\n\
        return numpy.ma.core.MaskedArray.__getitem__(self, index)\n\
    def __init__(self, *args, **kwargs):\n\
        self.bat = None\n\
        self.materialized = True\n\
        super(numpy.ma.core.MaskedArray, self).__init__(*args, **kwargs)\n\
        self._mask = numpy.ma.core.nomask\n\
        self._fill_value = None\n\
        self._hardmask = False\n\
        self._sharedmask = False\n\
        self.materialized = False\n\
        self.hasnil = False";

void lazyarray_init(void)
{
    PyObject *pModule;
    import_array();

    pModule = PyImport_Import(PyString_FromString("numpy.ma.core"));
    if (pModule == NULL) {
        return;
    }
    _maskedarray_type = (PyTypeObject*)PyObject_GetAttrString(pModule, "MaskedArray");
    if (_maskedarray_type == NULL) {
        return;
    }

    pModule = PyImport_Import(PyString_FromString("__main__"));
    if (pModule == NULL) {
        return;
    }
    if (PyRun_SimpleString(lazyarray_code) != 0) {
        return;
    }
    _lazyarray_type = (PyTypeObject*)PyObject_GetAttrString(pModule, "LazyArray");
    if (_lazyarray_type == NULL) {
        return;
    }
    (void) lazyarray_methods;
    //_lazyarray_type->tp_methods = lazyarray_methods;
    _lazyarray_size = _lazyarray_type->tp_basicsize;
}


PyObject *PyLazyArray_AsNumpyArray(PyObject *lazyarray)
{
    if (PyObject_GetAttrString(lazyarray, "materialized") == Py_False) {
        lazyarray_materialize(lazyarray, NULL);
    }
    return lazyarray;
}

PyTypeObject PyBAT_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "monetdb.bat",
    sizeof(PyBAT_Type),
    0,
    0,                   /* tp_dealloc */
    0,                      /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                        /* tp_repr */
    0,                                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                    /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_BASETYPE,         /* tp_flags */
    0,//lazyarray_doc,                                   /* tp_doc */
    0,                /* tp_traverse */
    0,//(inquiry)lazyarray_clear,                        /* tp_clear */
    0,//list_richcompare,                           /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,//lazyarray_ iter,                                  /* tp_iter */
    0,                                          /* tp_iternext */
    0,                               /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                        /* tp_init */
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
    register PyObject *op;
    PyObject *bat, *materialize;
    if (_lazyarray_type == NULL) {
        PyErr_Format(PyExc_Exception, "lazyarray was not initialized!");
        return NULL;
    }

    op = _lazyarray_type->tp_alloc(_lazyarray_type, 0);
    if (op == NULL) {
        printf("Failed to alloc"); fflush(stdout);
        return PyErr_NoMemory();
    }

    if (_lazyarray_type->tp_init(op, PyTuple_New(0), PyDict_New()) < 0) {
        printf("Failed to set"); fflush(stdout);
        return PyErr_NoMemory();        
    }

    bat = PyBAT_FromBAT(b);
    if (bat == NULL || PyObject_SetAttrString(op, "bat", bat) < 0) {
        printf("Failed to set"); fflush(stdout);
        return PyErr_NoMemory();
    }

    materialize = PyCFunction_NewEx(&lazyarray_methods[0], op, NULL);
    if (materialize == NULL || PyObject_SetAttrString(op, "materialize", PyCFunction_NewEx(&lazyarray_methods[0], op, NULL)) < 0) {
        printf("Failed to set"); fflush(stdout);
        return PyErr_NoMemory();
    }

    ((PyArrayObject_fields*)op)->data = NULL;
    ((PyArrayObject_fields*)op)->descr = PyArray_DescrFromType(BatType_ToPyType(ATOMstorage(getColumnType(b->T->type))));
    ((PyArrayObject_fields*)op)->nd = 1;
    ((PyArrayObject_fields*)op)->dimensions = GDKzalloc(sizeof(npy_intp) * 1);
    ((PyArrayObject_fields*)op)->dimensions[0] = 0;
    ((PyArrayObject_fields*)op)->strides = GDKzalloc(sizeof(npy_intp) * 1);
    ((PyArrayObject_fields*)op)->strides[0] = ((PyArrayObject_fields*)op)->descr->elsize;
    ((PyArrayObject_fields*)op)->weakreflist = NULL;
    ((PyArrayObject_fields*)op)->base = NULL;
    ((PyArrayObject_fields*)op)->flags = 0;

    PyErr_Print();
    return (PyObject*)op;
}


PyObject *PyBAT_FromBAT(BAT *b)
{
    register PyBATObject *op;

    op = (PyBATObject *)PyObject_MALLOC(sizeof(PyBAT_Type));
    if (op == NULL)
        return PyErr_NoMemory();
    PyObject_INIT(op, &PyBAT_Type);

    op->bat = b;

    return (PyObject*) op;
}

