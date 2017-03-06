/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */

#include "monetdb_config.h"

#include "mal.h"
#include "mal_exception.h"
#include "gdk_bbp.h"

#include "nested_table.h"

// errors
#if !defined(NDEBUG) /* debug only */
#define _CHECK_ERRLINE_EXPAND(LINE) #LINE
#define _CHECK_ERRLINE(LINE) _CHECK_ERRLINE_EXPAND(LINE)
#define _CHECK_ERRMSG(EXPR, ERROR) "[" __FILE__ ":" _CHECK_ERRLINE(__LINE__) "] " ERROR ": `" #EXPR "'"
#else /* release mode */
#define _CHECK_ERRMSG(EXPR, ERROR) ERROR
#endif
#define CHECK( EXPR, ERROR ) if ( !(EXPR) ) \
	{ rc = createException(MAL, function_name /*__FUNCTION__?*/, _CHECK_ERRMSG( EXPR, ERROR ) ); goto error; }

// prototypes
mal_export str NESTEDTABLEnest1_oid(bat*, const bat*, const bat*);
mal_export str NESTEDTABLEunnest101_oid(bat*, bat*, const bat*);
mal_export void NESTEDTABLEheap(Heap*, size_t);
mal_export var_t NESTEDTABLEput(Heap *h, var_t *bun, nested_table *val);
mal_export str NESTEDTABLEprelude(void*);

// index in the BATatoms table
int TYPE_nested_table;

void NESTEDTABLEheap(Heap* heap, size_t capacity){
	(void) capacity; // capacity refers to the #number of elements in the non virtual heap
	HEAP_initialize(heap, 0, 0, sizeof(oid));
}

// initializer
str NESTEDTABLEprelude(void* ret) {
	(void) ret;
	TYPE_nested_table = ATOMindex("nestedtable");

	BATatoms[TYPE_nested_table].linear = FALSE;
	return MAL_SUCCEED;
}

var_t NESTEDTABLEput(Heap *h, var_t *bun, nested_table *value) {
	const size_t size = (value->count+1) * sizeof(oid);
	const var_t offset = HEAP_malloc(h, size) << GDK_VARSHIFT; // longjmp in case of error O.o
	memcpy(h->base + offset, value, size);
	return (*bun = offset);
}

// MAL interface
mal_export str NESTEDTABLEnest1_oid(bat* id_out, const bat* id_group_mapping, const bat* id_histogram) {
	const char* function_name = "nestedtable.nest1";
	str rc = MAL_SUCCEED;
	BAT* group_mapping = NULL;
	BAT* histogram = NULL;
	lng* __restrict histogram_values = NULL;
	BAT* output = NULL;
	var_t heap_alloc_offset = 0;
	var_t* /*__restrict*/ output_offsets = NULL;
	char* /*__restrict*/ output_content = NULL;
	size_t output_sz = 0;
	size_t output_content_sz = 0;
	oid sum = 0;

	// input arguments
	CHECK(id_group_mapping != NULL, ILLEGAL_ARGUMENT);
	group_mapping = BATdescriptor(*id_group_mapping);
	CHECK(group_mapping != NULL, RUNTIME_OBJECT_MISSING);
	CHECK(id_histogram != NULL, ILLEGAL_ARGUMENT);
	histogram = BATdescriptor(*id_histogram);
	CHECK(histogram != NULL, RUNTIME_OBJECT_MISSING);

	// output
	assert(group_mapping->hseqbase == 0 && "This implementation is limited to a single partition");
	output = COLnew(group_mapping->hseqbase, TYPE_nested_table, BATcount(histogram), TRANSIENT);
	CHECK(output != NULL, MAL_MALLOC_FAIL);

	// allocate the virtual heap
	output_sz = BATcount(histogram);
	output_content_sz = output_sz + BATcount(group_mapping);
	heap_alloc_offset = HEAP_malloc(output->tvheap, output_content_sz * sizeof(oid)) << GDK_VARSHIFT;
	output_content = output->tvheap->base;
	memset(output_content + heap_alloc_offset, 0, output_content_sz * sizeof(oid));

	// edge case, the input is empty
	if(output_sz == 0) goto success;

	// compute the offsets
	histogram_values = (lng*) histogram->T.heap.base;
	output_offsets = (var_t*) output->T.heap.base;
	sum = heap_alloc_offset;
	output_offsets[0] = sum >> GDK_VARSHIFT;
	for (size_t i = 1; i < output_sz; i++){
		sum += (histogram_values[i-1] + /* length */ 1) * sizeof(oid);
		output_offsets[i] = ((var_t) sum) >> GDK_VARSHIFT;
	}
	sum += (histogram_values[output_sz -1] +1) * sizeof(oid);
	assert((sum - heap_alloc_offset) == output_content_sz * sizeof(oid) && "computed sum != #groups + #values");
	output->tvarsized = 1;
	output->tkey = 1;
	output->tdense = output->tnodense = 0;
	output->tsorted = 1;
	output->trevsorted = output->tnorevsorted = 0;
	output->tnonil = 1; output->tnil = 0;
	BATsetcount(output, output_sz);

	// insert the actual values into the vheap
	switch(group_mapping->ttype){
	case TYPE_oid: { // regular case
		oid* __restrict group_mapping_values = (oid*) group_mapping->theap.base;
		for(size_t i = 0, sz = BATcount(group_mapping); i < sz; i++){
			var_t offset = output_offsets[ group_mapping_values[i] ] << GDK_VARSHIFT;
			oid* __restrict values = (oid*) (output_content + offset);
			oid pos = ++values[0];
			values[pos] = i;
		}
		break;
	}
	case TYPE_void: { // extreme case causing seg~ faults
		for(size_t i = group_mapping->T.seq /* tseqbase */, sz = BATcount(group_mapping); i < sz; i++){
			var_t offset = output_offsets[ i ] << GDK_VARSHIFT;
			oid* __restrict values = (oid*) (output_content + offset);
			oid pos = ++values[0];
			values[pos] = i;
		}
		break;
	}
	default:
		assert(false && "Invalid input type for the parameter group_mapping");
		CHECK(false, ILLEGAL_ARGUMENT); // in case it skips the above assertion
	}

success:
	BBPunfix(group_mapping->batCacheid);
	BBPunfix(histogram->batCacheid);
	*id_out = output->batCacheid;
	BBPkeepref(output->batCacheid);

	return rc;
error:
	if(group_mapping) { BBPunfix(group_mapping->batCacheid); }
	if(histogram) { BBPunfix(histogram->batCacheid); }
	if(output) { BBPunfix(output->batCacheid); }

//	goto success;
	return rc;
}

mal_export str NESTEDTABLEunnest101_oid(bat* out_jl, bat* out_jr, const bat* in_nested_attribute){
	const char* function_name = "nestedtable.unnest1";
	str rc = MAL_SUCCEED;
	BAT *nested_attribute = NULL;
	var_t* __restrict na_offsets = NULL;
	char* __restrict na_values = NULL;
	oid na_count = 0;
	BAT *jl = NULL, *jr = NULL;

	// input argument
	CHECK(in_nested_attribute != NULL, ILLEGAL_ARGUMENT);
	nested_attribute = BATdescriptor(*in_nested_attribute);
	CHECK(nested_attribute != NULL, RUNTIME_OBJECT_MISSING);
	CHECK(ATOMtype(nested_attribute->ttype) == TYPE_nested_table, ILLEGAL_ARGUMENT);

	// output arguments
	assert(nested_attribute->hseqbase == 0 && "Partition unexpected");
	jl = COLnew(nested_attribute->hseqbase, TYPE_oid, BATcount(nested_attribute) * 5, TRANSIENT); // *5 <= arbitrary value
	jr = COLnew(nested_attribute->hseqbase, TYPE_oid, BATcount(nested_attribute) * 5, TRANSIENT);
	CHECK(jl != NULL && jr != NULL, MAL_MALLOC_FAIL);

	na_count = BATcount(nested_attribute);
	na_offsets = (var_t*) nested_attribute->T.heap.base;
	assert(nested_attribute->T.vheap != NULL);
	na_values =  nested_attribute->T.vheap->base;
	assert(na_values != NULL);
	for(oid i = 0; i < na_count; i++){
		var_t offset = na_offsets[i] << GDK_VARSHIFT;
		oid* __restrict base = (oid*) (na_values + offset);
		oid off_count = *(base++);
		assert((offset + (off_count * sizeof(oid)) <= nested_attribute->T.vheap->size) && "Index out of bounds");
		for(oid j = 0; j < off_count; j++){
			BUNappend(jl, &i, false);
		}
		for(oid j = 0; j < off_count; j++){
			oid off_value = base[j];
			BUNappend(jr, &off_value, false);
		}
	}

//success:
	BBPunfix(nested_attribute->batCacheid);
	*out_jl = jl->batCacheid;
	BBPkeepref(jl->batCacheid);
	*out_jr = jr->batCacheid;
	BBPkeepref(jr->batCacheid);

	return rc;
error:
	if(nested_attribute){ BBPunfix(nested_attribute->batCacheid); }
	if(jl){ BBPunfix(jl->batCacheid); }
	if(jr){ BBPunfix(jr->batCacheid); }

	return rc;
}

