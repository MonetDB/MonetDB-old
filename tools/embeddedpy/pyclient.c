
#include "pyclient.h"
#include "embedded.h"


static void
pyclient_dealloc(PyClientObject *self) {
    monetdb_disconnect(self->cntxt);
    self->cntxt = NULL;
}

PyTypeObject PyClientType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "monetdb._connection",
    sizeof(PyClientObject),
    0,
    (destructor)pyclient_dealloc,               /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    (hashfunc)PyObject_HashNotImplemented,      /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    "MonetDB Client Object",                    /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    PyType_GenericNew,                          /* tp_new */
    PyObject_Del,                               /* tp_free */
    0,
    0,
    0,
    0,
    0,
    0, 
    0,
    0
};


PyObject *PyClient_Create(Client cntxt)
{
    register PyClientObject *op;

    op = (PyClientObject *)PyObject_MALLOC(sizeof(PyClientObject));
    if (op == NULL)
        return PyErr_NoMemory();
    PyObject_INIT(op, &PyClientType);

    op->cntxt = cntxt;

    return (PyObject*) op;
}

void monetdbclient_init(void)
{
    if (PyType_Ready(&PyClientType) < 0)
        return;
}
