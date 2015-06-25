
#include "bytearray.h"


PyCFunction replaced_functions[10];
int replaced_index = 0;

PyObject *PyByteArray_FromString(char *str)
{
    PyByteArrayObject *new;
    size_t size = strlen(str);

    new = PyObject_New(PyByteArrayObject, &PyByteArray_Type);
    if (new == NULL)
        return NULL;

    Py_SIZE(new) = size;
    new->ob_alloc = 0;
    new->ob_exports = 1;
    new->ob_bytes = str;

    return (PyObject *)new;
}

ssizeobjargproc setitem_func = NULL;
static int
bytearray_setitem(PyByteArrayObject *self, Py_ssize_t i, PyObject *value)
{
    if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not assign to it.");
        return -1;
    }
    return setitem_func((PyObject*)self, i, value);
}

objobjargproc assignsubscript_func = NULL;
static int
bytearray_ass_subscript(PyByteArrayObject *self, PyObject *index, PyObject *values)
{
    if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not assign to it.");
        return -1;
    }
    return assignsubscript_func((PyObject*)self, index, values);
}

int reverse_index = -1;
static PyObject *
bytearray_reverse(PyByteArrayObject *self, PyObject *unused)
{
    if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not reverse it.");
        Py_RETURN_NONE;
    }
    return replaced_functions[reverse_index]((PyObject*)self, unused);
}

int remove_index = -1;
static PyObject *
bytearray_remove(PyByteArrayObject *self, PyObject *arg)
{
    if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not remove anything from it.");
        Py_RETURN_NONE;
    }
    return replaced_functions[remove_index]((PyObject*)self, arg);
}

int pop_index = -1;
static PyObject *
bytearray_pop(PyByteArrayObject *self, PyObject *args)
{
    if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not remove anything from it.");
        Py_RETURN_NONE;
    }
    return replaced_functions[pop_index]((PyObject*)self, args);
}

int extend_index = -1;
static PyObject *
bytearray_extend(PyByteArrayObject *self, PyObject *arg)
{
	if (self->ob_alloc == 0) {
        PyErr_SetString(PyExc_TypeError, "This ByteArray references to a BAT in the database, you may not extend it.");
        Py_RETURN_NONE;
    }
    return replaced_functions[extend_index]((PyObject*)self, arg);
}

int replace_method(char *name, PyCFunction method);
int replace_method(char *name, PyCFunction method)
{
	int i = 0, index = -1;
	while((&PyByteArray_Type)->tp_methods[i].ml_name != NULL)
	{
		if (strcmp(name, (&PyByteArray_Type)->tp_methods[i].ml_name) == 0) 
		{
			index = replaced_index++;
			replaced_functions[index] = (&PyByteArray_Type)->tp_methods[i].ml_meth;
        	(&PyByteArray_Type)->tp_methods[i].ml_meth = method;
        	i = -1;
        	break;
	    } 
		i++;
	}
	if (i >= 0)
	{
        fprintf(stderr, "WARNING: Trying to replace function %s in ByteArrayObject with a new object but the function could not be found. Maybe this is because of a different/newer Python version?", name);
	}
	return index;
}

void PyByteArray_Override(void)
{
	// We override all the Python ByteArray methods that modify the Byte Array
	// We replace them by methods that check if the ob_alloc property is set to 0 (this should only happen with ByteArrays created by PyByteArray_FromString)
	// and if it is throw an error message
	
	// Set item: bytearray.__setitem__(0, 'a')
	setitem_func = (&PyByteArray_Type)->tp_as_sequence->sq_ass_item;
    (&PyByteArray_Type)->tp_as_sequence->sq_ass_item = (ssizeobjargproc)bytearray_setitem;
    // Subscript assign: bytearray[0] = 'a'
	assignsubscript_func = (&PyByteArray_Type)->tp_as_mapping->mp_ass_subscript;
    (&PyByteArray_Type)->tp_as_mapping->mp_ass_subscript = (objobjargproc)bytearray_ass_subscript;
    // Reverse: bytearray.reverse()
    reverse_index = replace_method("reverse", (PyCFunction)bytearray_reverse);
    // Remove: bytearray.remove(1)
    remove_index = replace_method("remove", (PyCFunction)bytearray_remove);
    // Pop: bytearray.pop()
    pop_index = replace_method("pop", (PyCFunction)bytearray_pop);
    // Extend: bytearray.extend('extend')
    extend_index = replace_method("extend", (PyCFunction)bytearray_extend);
}
