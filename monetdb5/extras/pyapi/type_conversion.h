/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * M. Raasveldt
 * This file contains a number of helper functions for converting between types, mainly used to convert from an object from a numpy array to the type requested by the BAT.
 */

#ifndef _TYPE_CONVERSION_
#define _TYPE_CONVERSION_


//! Copies the string of size up to max_size from the source to the destination, returns FALSE if "source" is not a legal ASCII string (i.e. a character is >= 128)
bool string_copy(char * source, char* dest, size_t max_size);
//! Converts a long to a string and writes it into the string "str"
void lng_to_string(char* str, lng value);
//! Converts a double to a string and writes it into the string "str"
void dbl_to_string(char* str, dbl value);
//! Converts a hge to a string and writes it into the string "str" [base 16], size specifies the maximum size of the string
int hge_to_string(char *, int, hge );
//! Converts a base-10 string to a hge value
bool s_to_hge(char *ptr, size_t size, hge *value);
//! Converts a base-10 string to a dbl value
bool s_to_dbl(char *ptr, size_t size, dbl *value);
//! Converts a base-10 string to a lng value
bool s_to_lng(char *ptr, size_t size, lng *value);
//! Converts a base-10 utf32-encoded string to a lng value
bool utf32_to_lng(uint32_t *utf32, lng *value);
//! Converts a base-10 utf32-encoded string to a dbl value
bool utf32_to_dbl(uint32_t *utf32, dbl *value);
//! Converts a base-10 utf32-encoded string to a hge value
bool utf32_to_hge(uint32_t *utf32, hge *value);


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

int hge_to_string(char * str, int size, hge x)
{
    int i = 0;
    for(i = 0; i < size - 1; i++) str[i] = '0';
    if (x < 0) 
    {
        x *= -1;
        str[0] = '-';
    }
    str[size - 1] = '\0';
    while(x > 0)
    {
        int v = x % 16;
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
        else if (v == 10) str[i] = 'a';
        else if (v == 11) str[i] = 'b';
        else if (v == 12) str[i] = 'c';
        else if (v == 13) str[i] = 'd';
        else if (v == 14) str[i] = 'e';
        else if (v == 15) str[i] = 'f';
        x = x / 16;
    }
    return TRUE;
}

//using macros, create a number of str_to_<type> and unicode_to_<type> functions
#define CONVERSION_FUNCTION_FACTORY(tpe, strconv, utfconv, strval)          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value);          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value)           \
    {                                                              \
        strval val;                                                \
        if (!strconv((char*)ptr, size, &val)) return false;   \
        *value = (tpe)val;                                         \
        return true;                                               \
    }                                                              \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value);                  \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value)                   \
    {                                                              \
        strval val;                                                \
        (void) size;                                               \
        if (!utfconv((uint32_t*)ptr, &val)) return false;         \
        *value = (tpe)val;                                         \
        return true;                                               \
    }                                                              
    
CONVERSION_FUNCTION_FACTORY(bit, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_FACTORY(sht, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_FACTORY(int, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_FACTORY(lng, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_FACTORY(hge, s_to_hge, utf32_to_hge, hge)
CONVERSION_FUNCTION_FACTORY(flt, s_to_dbl, utf32_to_dbl, dbl)
CONVERSION_FUNCTION_FACTORY(dbl, s_to_dbl, utf32_to_dbl, dbl)

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

#endif /* _TYPE_CONVERSION_ */
