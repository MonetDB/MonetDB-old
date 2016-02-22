/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

#ifndef _K3M_LIB_
#define _K3M_LIB_

#include "monetdb_config.h"
#include "mal.h"
#include "mal_client.h"
#include "mal_interpreter.h"
#include <math.h>

#include "k3match.h"

#ifdef WIN32
#define k3m_export extern __declspec(dllexport)
#else
#define k3m_export extern
#endif

k3m_export str K3Mprelude(void *ret);
k3m_export str K3Mbuild(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		 InstrPtr pci);
k3m_export str K3Mfree(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		 InstrPtr pci);
k3m_export str K3Mquery(Client cntxt, MalBlkPtr mb, MalStkPtr stk,
		 InstrPtr pci);

typedef struct {
	node_t *tree;
	real_t *values;
	point_t **catalog;
} k3m_tree_tpe;

static k3m_tree_tpe *k3m_tree = NULL;
static MT_Lock k3m_lock;

#define K3M_ALLOCS_DEFAULT_SIZE 10

static size_t k3m_allocs_size = K3M_ALLOCS_DEFAULT_SIZE;
static size_t k3m_allocs_pos = 0;
static k3m_tree_tpe **k3m_allocs = NULL;

k3m_export str K3Mprelude(void *ret) {
	(void) ret;
	MT_lock_init(&k3m_lock, "k3m_lock");
	return MAL_SUCCEED;
}

str K3Mbuild(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
	BAT *ids, *ra, *dec, *ret;
	int *ids_a;
	dbl *ra_a, *dec_a;
	size_t N_a;
	int_t i;
	int_t npool = 0;
	bit b = bit_nil;
	bit newtree = !k3m_tree;
	k3m_tree_tpe *k3m_tree_alloc = NULL;
	(void) cntxt;
	MT_lock_set(&k3m_lock);

	if (!isaBatType(getArgType(mb,pci,0)) || !isaBatType(getArgType(mb,pci,1)) ||
			!isaBatType(getArgType(mb,pci,2)) || !isaBatType(getArgType(mb,pci,3))) {
		return createException(MAL, "k3m.build", "Can only deal with BAT types. Sorry.");
	}
	for (i = 0; i <= 3; i++) {
		if (!isaBatType(getArgType(mb,pci,i))) {
			return createException(MAL, "k3m.build", "Can only deal with BAT types. Sorry.");
		}
	}
	ids = BATdescriptor(*getArgReference_bat(stk, pci, 1));
	ra  = BATdescriptor(*getArgReference_bat(stk, pci, 2));
	dec = BATdescriptor(*getArgReference_bat(stk, pci, 3));

	N_a = BATcount(ids);
	assert(ids && ra && dec); // TODO: dynamic checks & errors instead of asserts

	ids_a = ((int*) Tloc(ids, BUNfirst(ids)));
	ra_a  = ((dbl*) Tloc(ra,  BUNfirst(ra)));
	dec_a = ((dbl*) Tloc(dec, BUNfirst(dec)));

	assert(ids_a && ra_a && dec_a); // TODO: dynamic checks & errors instead of asserts

	if (newtree) {
		k3m_tree = GDKmalloc(sizeof(k3m_tree_tpe));
		k3m_allocs = GDKmalloc(k3m_allocs_size);
		if (!k3m_allocs) {
			// yes I know we should probably free some stuff here but its very unlikely this fails
			return createException(MAL, "k3m.build", "Memory allocation failed 1.");
		}
		k3m_tree_alloc = k3m_tree;
	} else {
		k3m_tree_alloc = GDKmalloc(sizeof(k3m_tree_tpe));
		// enlarge malloc pointer array size if neccessary
		if (k3m_allocs_pos >= k3m_allocs_size) {
			k3m_allocs_size *= 2;
			k3m_allocs = GDKrealloc(k3m_allocs, k3m_allocs_size);
			if (!k3m_allocs) {
				// see above
				return createException(MAL, "k3m.build", "Memory allocation failed 2.");
			}
		}
	}
	k3m_allocs[k3m_allocs_pos++] = k3m_tree_alloc;

	if (!k3m_tree || !k3m_tree_alloc) {
		if (k3m_tree) {
			GDKfree(k3m_tree);
		}
		if (k3m_tree_alloc) {
			GDKfree(k3m_tree_alloc);
		}
		return createException(MAL, "k3m.build", "Memory allocation failed 3.");
	}

	k3m_tree_alloc->values = GDKmalloc(3 * N_a * sizeof(real_t));
	k3m_tree_alloc->catalog = GDKmalloc(N_a * sizeof(point_t*));
	*k3m_tree_alloc->catalog = GDKmalloc(N_a * sizeof(point_t));
	k3m_tree_alloc->tree = (node_t*) GDKmalloc(N_a * sizeof(node_t));

	if (!k3m_tree_alloc->values || !k3m_tree_alloc->catalog || !*k3m_tree_alloc->catalog) {
		if (k3m_tree_alloc->values) {
			GDKfree(k3m_tree_alloc->values);
		}
		if (k3m_tree_alloc->catalog) {
			GDKfree(k3m_tree_alloc->catalog);
		}
		if (*k3m_tree_alloc->catalog) {
			GDKfree(*k3m_tree_alloc->catalog);
		}
		if (k3m_tree_alloc->tree) {
			GDKfree(k3m_tree_alloc->tree);
		}
		return createException(MAL, "k3m.build", "Memory allocation failed 4.");
	}

	for (i=0; i<N_a; i++) {
		k3m_tree_alloc->catalog[i] = k3m_tree_alloc->catalog[0] + i;
		k3m_tree_alloc->catalog[i]->id = ids_a[i];
		k3m_tree_alloc->catalog[i]->value = k3m_tree_alloc->values + 3 * i;
		k3m_tree_alloc->catalog[i]->value[0] = cos(dec_a[i]) * cos(ra_a[i]);
		k3m_tree_alloc->catalog[i]->value[1] = cos(dec_a[i]) * sin(ra_a[i]);
		k3m_tree_alloc->catalog[i]->value[2] = sin(dec_a[i]);
	}

	if (newtree) {
		k3m_tree->tree->parent = NULL;
		k3m_build_balanced_tree(k3m_tree->tree, k3m_tree->catalog, N_a, 0, &npool);
	} else {
		for (i=0; i<N_a; i++) {
			k3m_tree_alloc->tree[i].point = k3m_tree_alloc->catalog[i];
			k3m_tree->tree = k3m_insert_node(k3m_tree->tree, &k3m_tree_alloc->tree[i]);
		}
	}
	MT_lock_unset(&k3m_lock);
	ret = BATnew(TYPE_void, TYPE_bit, 0, TRANSIENT);
	BUNappend(ret, &b, 0);
	*getArgReference_bat(stk, pci, 0) = ret->batCacheid;
	BBPkeepref(ret->batCacheid);
	return MAL_SUCCEED;

}


str K3Mfree(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {

	BAT *ret;
	bit b = bit_nil;
	size_t i;

	(void) cntxt;
	(void) mb;
	(void) stk;
	(void) pci;

	MT_lock_set(&k3m_lock);

	for (i = 0; i < k3m_allocs_pos; i++) {
		GDKfree(k3m_allocs[i]->tree);
		GDKfree(k3m_allocs[i]->values);
		GDKfree(k3m_allocs[i]->catalog);
	}
	GDKfree(k3m_allocs);
	k3m_allocs = NULL;
	k3m_allocs_pos = 0;
	k3m_allocs_size = K3M_ALLOCS_DEFAULT_SIZE;
	k3m_tree = NULL;

	MT_lock_unset(&k3m_lock);

	ret = BATnew(TYPE_void, TYPE_bit, 0, TRANSIENT);
	BUNappend(ret, &b, 0);
	*getArgReference_bat(stk, pci, 0) = ret->batCacheid;
	BBPkeepref(ret->batCacheid);

	return MAL_SUCCEED;
}

str K3Mquery(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
	size_t i;
	BAT *in_ids, *in_ra, *in_dec, *in_dist, *out_id_cat, *out_id_sl, *out_dist;
	int *in_ids_a;
	dbl *in_ra_a, *in_dec_a, *in_dist_a;
	point_t search, *match;
	size_t N_b;
	point_t *mi = NULL;
	int_t nmatch = 0;

	(void) cntxt;

	for (i = 0; i <= 6; i++) {
		if (!isaBatType(getArgType(mb,pci,i))) {
			return createException(MAL, "k3m.build", "Can only deal with BAT types. Sorry.");
		}
	}

	if (!k3m_tree) {
		return createException(MAL, "k3m.build", "Tree not built!");
	}

	in_ids     = BATdescriptor(*getArgReference_bat(stk, pci, 3));
	in_ra      = BATdescriptor(*getArgReference_bat(stk, pci, 4));
	in_dec     = BATdescriptor(*getArgReference_bat(stk, pci, 5));
	in_dist    = BATdescriptor(*getArgReference_bat(stk, pci, 6));

	assert(in_ids && in_ra && in_dec && in_dist);

	in_ids_a = ((int*) Tloc(in_ids, BUNfirst(in_ids)));
	in_ra_a  = ((dbl*) Tloc(in_ra, BUNfirst(in_ra)));
	in_dec_a = ((dbl*) Tloc(in_dec, BUNfirst(in_dec)));
	in_dist_a = ((dbl*) Tloc(in_dist, BUNfirst(in_dist)));

	assert(in_ids_a && in_ra_a && in_dec_a && in_dist_a);

	out_id_cat = BATnew(TYPE_void, TYPE_int, 0, TRANSIENT);
	out_id_sl  = BATnew(TYPE_void, TYPE_int, 0, TRANSIENT);
	out_dist   = BATnew(TYPE_void, TYPE_dbl, 0, TRANSIENT);

	N_b = BATcount(in_ids);

	search.value = GDKmalloc(3 * sizeof(real_t));

	for (i=0; i<N_b; i++) {
		search.id = in_ids_a[i];
		search.value[0] = cos(in_dec_a[i]) * cos(in_ra_a[i]);
		search.value[1] = cos(in_dec_a[i]) * sin(in_ra_a[i]);
		search.value[2] = sin(in_dec_a[i]);

		match = NULL;
		nmatch = k3m_in_range(k3m_tree->tree, &match, &search, in_dist_a[i]);

		mi = match;
		nmatch++;
		while (--nmatch) {
			BUNappend(out_id_sl, &search.id, 0);
			BUNappend(out_id_cat, &mi->id, 0);
			BUNappend(out_dist, &mi->ds, 0);
			mi = mi->neighbour;
		}
	}
	GDKfree(search.value);

	BBPkeepref(out_id_cat->batCacheid);
	BBPkeepref(out_id_sl->batCacheid);
	BBPkeepref(out_dist->batCacheid);

	*getArgReference_bat(stk, pci, 0) = out_id_cat->batCacheid;
	*getArgReference_bat(stk, pci, 1) = out_id_sl->batCacheid;
	*getArgReference_bat(stk, pci, 2) = out_dist->batCacheid;

	return MAL_SUCCEED;
}

#endif /* _K3M_LIB_ */
