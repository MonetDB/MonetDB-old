/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
 */
#include <inttypes.h> // uint64_t
#include <stdlib.h> // rand

#include "monetdb_config.h"
#include "sql_mem.h"
#include "sql_hash.h"

static unsigned int
log_base2(unsigned int n)
{
	unsigned int l ;

	for (l = 0; n; l++)
		n >>= 1 ;
	return l ;
}


sql_hash *
hash_new(sql_allocator *sa, int size, fkeyvalue key)
{
	int i;
	sql_hash *ht = SA_ZNEW(sa, sql_hash);

	if (ht == NULL)
		return NULL;
	ht->sa = sa;
	ht->size = (1<<log_base2(size-1));
	ht->key = key;
	ht->buckets = SA_NEW_ARRAY(sa, sql_hash_e*, ht->size);
	for(i = 0; i < ht->size; i++)
		ht->buckets[i] = NULL;
	return ht;
}

sql_hash_e*
hash_add(sql_hash *h, int key, void *value)
{
	sql_hash_e *e = SA_ZNEW(h->sa, sql_hash_e);

	if (e == NULL)
		return NULL;
	e->chain = h->buckets[key&(h->size-1)];
	h->buckets[key&(h->size-1)] = e;
	e->key = key;
	e->value = value;
	return e;
}

void
hash_del(sql_hash *h, int key, void *value)
{
	sql_hash_e *e = h->buckets[key&(h->size-1)], *p = NULL;

	while (e && (e->key != key || e->value != value)) {
		p = e;
		e = e->chain;
	}
	if (e) {
		if (p) 
			p->chain = e->chain;
		else
			h->buckets[key&(h->size-1)] = e->chain;
	}
}

unsigned int
hash_key(const char *k)
{
	unsigned int h = 0;

	while (*k) {
		h += *k;
		h += (h << 10);
		h ^= (h >> 6);
		k++;
	}
	h += (h << 3);
	h ^= (h >> 11);
	h += (h << 15);
	return h;
}

//unsigned int
//hash_key_ptr(void* ptr)
//{
//	static uint64_t a = 0;
//	static uint64_t b = 0;
//	const uint64_t p = 2386660291; // prime number
//	if(a == 0 && b == 0){
//		srand(time(NULL));
//		a = rand() +1;
//		b = rand() +1;
//	}
//
//	return (unsigned int) (( a * ((uint64_t) ptr) + b ) % p);
//}
