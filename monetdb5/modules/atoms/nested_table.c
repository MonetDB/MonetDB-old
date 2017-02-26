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
mal_export str NESTEDTABLEprelude(void*);

// index in the BATatoms table
int TYPE_nested_table;

// only needed for its side effect: it asks the gdk initialise the vheap storage
void NESTEDTABLEheap(Heap *heap, size_t capacity){
	(void) capacity;
	heap->base = heap->base -1; // fool the gdk -> ATOMheap
}

// initializer
str NESTEDTABLEprelude(void* ret) {
	atomDesc* descriptor = NULL;

	(void) ret;
	TYPE_nested_table = ATOMindex("nestedtable");
	descriptor = BATatoms + TYPE_nested_table;
	descriptor->linear = FALSE;
	return MAL_SUCCEED;
}

// MAL interface
mal_export str NESTEDTABLEnest1_oid(bat* id_out, const bat* id_group_mapping, const bat* id_histogram) {
	const char* function_name = "nestedtable.nest1";
	str rc = MAL_SUCCEED;
	BAT* group_mapping = NULL;
	oid* __restrict group_mapping_values = NULL;
	BAT* histogram = NULL;
	lng* __restrict histogram_values = NULL;
	BAT* output = NULL;
	var_t* __restrict output_offsets = NULL;
	oid* __restrict output_content = NULL;
	size_t output_sz = 0;
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

	// edge case, the input is empty
	output_sz = BATcount(histogram);
	if(output_sz == 0) goto success;

	// compute the offsets
	histogram_values = (lng*) histogram->T.heap.base;
	output_offsets = (var_t*) output->T.heap.base;
	output_offsets[0] = 0;
	for (size_t i = 1; i < output_sz; i++){
		sum += histogram_values[i-1] + /* length */ 1;
		output_offsets[i] = (var_t) sum;
	}
	sum += histogram_values[output_sz -1] +1;
	assert(sum == BATcount(histogram) + BATcount(group_mapping) && "#groups + #values != computed sum");
	output->tvarsized = 1;
	output->tkey = 1;
	output->tdense = output->tnodense = 0;
	output->tsorted = 1;
	output->trevsorted = output->tnorevsorted = 0;
	output->tnonil = 1; output->tnil = 0;
	BATsetcount(output, output_sz);

	// allocate the virtual heap
	HEAP_initialize(output->tvheap, 0, sum, sizeof(oid));
	CHECK(output->tvheap != NULL, MAL_MALLOC_FAIL);
    memset(output->tvheap->base, 0, output->tvheap->size);

    // insert the actual values into the vheap
    group_mapping_values = (oid*) group_mapping->theap.base;
    output_content = (oid*) output->T.vheap->base;
    for(size_t i = 0, sz = BATcount(group_mapping); i < sz; i++){
    	oid count = output_offsets[ group_mapping_values[i] ];
    	oid pos = ++output_content[count];
    	output_content[pos] = i;
    }
    output->tvheap->free = sum * sizeof(oid);

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
	oid* __restrict na_values = NULL;
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

	// empty argument?
	na_count = BATcount(nested_attribute);
	if(na_count == 0) goto success; // skip here as the vheap may not be effectively allocated

	na_offsets = (var_t*) nested_attribute->T.heap.base;
	assert(nested_attribute->T.vheap != NULL);
	na_values = (oid*) nested_attribute->T.vheap->base;
	assert(na_values != NULL && ((lng) na_values) != -1);
	for(oid i = 0; i < na_count; i++){
		var_t offset = na_offsets[i];
		oid* __restrict base = na_values + offset;
		oid off_count = *(base++);
		assert(((offset+off_count) * sizeof(oid) <= nested_attribute->T.vheap->size) && "Index out of bounds");
		for(oid j = 0; j < off_count; j++){
			BUNappend(jl, &i, false);
		}
		for(oid j = 0; j < off_count; j++){
			oid off_value = base[j];
			BUNappend(jr, &off_value, false);
		}
	}

success:
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

