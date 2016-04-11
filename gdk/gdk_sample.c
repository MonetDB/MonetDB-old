/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 */

/*
 * @a Lefteris Sidirourgos, Hannes Muehleisen, Abe Wits
 * @* Low level sample facilities
 *
 * This sampling implementation generates a sorted set of OIDs by
 * calling the random number generator, and uses a binary tree to
 * eliminate duplicates.  The elements of the tree are then used to
 * create a sorted sample BAT.  This implementation has a logarithmic
 * complexity that only depends on the sample size.
 *
 * There is a pathological case when the sample size is almost the
 * size of the BAT.  Then, many collisions occur and performance
 * degrades. To catch this, we switch to antiset semantics when the
 * sample size is larger than half the BAT size. Then, we generate the
 * values that should be omitted from the sample.
 */

#include "monetdb_config.h"
#include "gdk.h"
#include "gdk_private.h"

#include "mtwist.h"

#undef BATsample

/* this is a straightforward implementation of a binary tree */
struct oidtreenode {
	oid o;
	struct oidtreenode *left;
	struct oidtreenode *right;
};

static int
OIDTreeMaybeInsert(struct oidtreenode *tree, oid o, BUN allocated)
{
	struct oidtreenode **nodep;

	if (allocated == 0) {
		tree->left = tree->right = NULL;
		tree->o = o;
		return 1;
	}
	nodep = &tree;
	while (*nodep) {
		if (o == (*nodep)->o)
			return 0;
		if (o < (*nodep)->o)
			nodep = &(*nodep)->left;
		else
			nodep = &(*nodep)->right;
	}
	*nodep = &tree[allocated];
	tree[allocated].left = tree[allocated].right = NULL;
	tree[allocated].o = o;
	return 1;
}

/* inorder traversal, gives us a sorted BAT */
static void
OIDTreeToBAT(struct oidtreenode *node, BAT *bat)
{
	if (node->left != NULL)
		OIDTreeToBAT(node->left, bat);
	((oid *) bat->T->heap.base)[bat->batFirst + bat->batCount++] = node->o;
	if (node->right != NULL )
		OIDTreeToBAT(node->right, bat);
}

/* Antiset traversal, give us all values but the ones in the tree */
static void
OIDTreeToBATAntiset(struct oidtreenode *node, BAT *bat, oid start, oid stop)
{
	oid noid;

	if (node->left != NULL)
			OIDTreeToBATAntiset(node->left, bat, start, node->o);
	else
		for (noid = start; noid < node->o; noid++)
			((oid *) bat->T->heap.base)[bat->batFirst + bat->batCount++] = noid;

		if (node->right != NULL)
 		OIDTreeToBATAntiset(node->right, bat, node->o + 1, stop);
	else
		for (noid = node->o+1; noid < stop; noid++)
						((oid *) bat->T->heap.base)[bat->batFirst + bat->batCount++] = noid;
}

/* BATsample implements sampling for void headed BATs */
BAT *
BATsample(BAT *b, BUN n)
{
	BAT *bn;
	BUN cnt, slen;
	BUN rescnt;
	struct oidtreenode *tree = NULL;
	mtwist *mt_rng;
	unsigned int range;

	BATcheck(b, "BATsample", NULL);
	assert(BAThdense(b));
	ERRORcheck(n > BUN_MAX, "BATsample: sample size larger than BUN_MAX\n", NULL);
	ALGODEBUG
		fprintf(stderr, "#BATsample: sample " BUNFMT " elements.\n", n);

	cnt = BATcount(b);
	/* empty sample size */
	if (n == 0) {
		bn = BATnew(TYPE_void, TYPE_void, 0, TRANSIENT);
		if (bn == NULL) {
			return NULL;
		}
		BATsetcount(bn, 0);
		BATseqbase(bn, 0);
		BATseqbase(BATmirror(bn), 0);
	/* sample size is larger than the input BAT, return all oids */
	} else if (cnt <= n) {
		bn = BATnew(TYPE_void, TYPE_void, cnt, TRANSIENT);
		if (bn == NULL) {
			return NULL;
		}
		BATsetcount(bn, cnt);
		BATseqbase(bn, 0);
		BATseqbase(BATmirror(bn), b->H->seq);
	} else {
		oid minoid = b->hseqbase;
		oid maxoid = b->hseqbase + cnt;
		/* if someone samples more than half of our tree, we
		 * do the antiset */
		bit antiset = n > cnt / 2;
		slen = n;
		if (antiset)
			n = cnt - n;

		tree = GDKmalloc(n * sizeof(struct oidtreenode));
		if (tree == NULL) {
			return NULL;
		}
		bn = BATnew(TYPE_void, TYPE_oid, slen, TRANSIENT);
		if (bn == NULL) {
			GDKfree(tree);
			return NULL;
		}
		/* while we do not have enough sample OIDs yet */
		
		/* create and seed Mersenne Twister */
		mt_rng = mtwist_new();

		mtwist_seed(mt_rng, rand());
		
		range = maxoid - minoid;
		
		for (rescnt = 0; rescnt < n; rescnt++) {
			oid candoid;
			do {
				/* generate a new random OID in [minoid, maxoid[
				 * that is including minoid, excluding maxoid*/
				candoid = (oid) ( minoid + (mtwist_u32rand(mt_rng)%range) );
				/* if that candidate OID was already
				 * generated, try again */
			} while (!OIDTreeMaybeInsert(tree, candoid, rescnt));
		}
		if (!antiset) {
			OIDTreeToBAT(tree, bn);
		} else {
			OIDTreeToBATAntiset(tree, bn, minoid, maxoid);
		}
		GDKfree(tree);

		BATsetcount(bn, slen);
		bn->trevsorted = bn->batCount <= 1;
		bn->tsorted = 1;
		bn->tkey = 1;
		bn->tdense = bn->batCount <= 1;
		if (bn->batCount == 1)
			bn->tseqbase = *(oid *) Tloc(bn, BUNfirst(bn));
		bn->hdense = 1;
		bn->hseqbase = 0;
		bn->hkey = 1;
		bn->hrevsorted = bn->batCount <= 1;
		bn->hsorted = 1;
	}
	return bn;
}
