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

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "monetdb_config.h"
#include "mal.h"
#include "mal_stack.h"
#include "mal_linker.h"
#include "gdk_utils.h"
#include "gdk.h"

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

//using macros, create a number of str_to_<type> and unicode_to_<type> functions
#define CONVERSION_FUNCTION_HEADER_FACTORY(tpe, strconv, utfconv, strval)          \
    bool str_to_##tpe(void *ptr, size_t size, tpe *value);          \
    bool unicode_to_##tpe(void *ptr, size_t size, tpe *value);                  \

CONVERSION_FUNCTION_HEADER_FACTORY(bit, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_HEADER_FACTORY(sht, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_HEADER_FACTORY(int, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_HEADER_FACTORY(lng, s_to_lng, utf32_to_lng, lng)
CONVERSION_FUNCTION_HEADER_FACTORY(hge, s_to_hge, utf32_to_hge, hge)
CONVERSION_FUNCTION_HEADER_FACTORY(flt, s_to_dbl, utf32_to_dbl, dbl)
CONVERSION_FUNCTION_HEADER_FACTORY(dbl, s_to_dbl, utf32_to_dbl, dbl)

#endif /* _TYPE_CONVERSION_ */
