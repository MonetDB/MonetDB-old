
#include "type_conversion.h"
#include "unicode.h"

#include <longintrepr.h>


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

void dbl_to_string(char* str, dbl value)
{
    sprintf(str, "%lf", value);
}

#ifdef HAVE_HGE
int hge_to_string(char * str, hge x)
{
    int i = 0;
    size_t size = 2;
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
    i = size - 2;
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


bool utf32_to_lng(uint32_t *utf32, lng *value)
{
    size_t length = utf32_strlen(utf32);
    int i = length;
    size_t factor = 1;
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

bool utf32_to_dbl(uint32_t *utf32, dbl *value)
{
    size_t length = utf32_strlen(utf32);
    int i = length;
    size_t factor = 1;
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
bool utf32_to_hge(uint32_t *utf32, hge *value)
{
    size_t length = utf32_strlen(utf32);
    int i = length;
    size_t factor = 1;
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
#endif


//py_to_hge and py_to_lng are almost identical, so use a generic #define to make them
#define PY_TO_(type)                                                                                             \
bool py_to_##type(PyObject *ptr, type *value)                                                                    \
{                                                                                                                \
    if (PyLong_Check(ptr)) {                                                                                     \
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
    } else if (PyInt_Check(ptr) || PyBool_Check(ptr)) {                                                          \
        *value = (type)((PyIntObject*)ptr)->ob_ival;                                                             \
        return true;                                                                                             \
    } else if (PyFloat_Check(ptr)) {                                                                             \
        *value = (type) ((PyFloatObject*)ptr)->ob_fval;                                                          \
        return true;                                                                                             \
    } else if (PyString_Check(ptr)) {                                                                            \
        return s_to_##type(((PyStringObject*)ptr)->ob_sval, strlen(((PyStringObject*)ptr)->ob_sval), value);     \
    }  else if (PyByteArray_Check(ptr)) {                                                                        \
        return s_to_##type(((PyByteArrayObject*)ptr)->ob_bytes, strlen(((PyByteArrayObject*)ptr)->ob_bytes), value);\
    } else if (PyUnicode_Check(ptr)) {                                                                           \
        return utf32_to_##type(((PyUnicodeObject*)ptr)->str, value);                                             \
    }                                                                                                            \
    return false;                                                                                                \
}

PY_TO_(lng);
#ifdef HAVE_HGE
PY_TO_(hge);

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
#endif

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
        (void) size;                                               \
        if (!utf32_to_##strval((uint32_t*)ptr, &val)) return false;         \
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
