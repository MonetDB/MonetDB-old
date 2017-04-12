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
		for (noid = node->o + 1; noid < stop; noid++)
			((oid *) bat->T->heap.base)[bat->batFirst + bat->batCount++] = noid;
}

/* inorder traversal, gives us a bit BAT */
/*BAT *bat OIDTreeToBITBAT(struct oidtreenode)
{
	//TODO create this function
}*/

/* 
 * _BATsample is the internal (weighted) sampling function without replacement
 * If cdf=NULL, an uniform sample is taken
 * Otherwise it is assumed the cdf increases monotonically
 */
static BAT *
_BATsample(BAT *b, BUN n, BAT *cdf)
{
	BAT *bn;
	BUN cnt, slen;
	BUN rescnt;
	struct oidtreenode *tree = NULL;
	mtwist *mt_rng;
	unsigned int range;
	dbl random;
	dbl cdf_max;
	dbl* cdf_ptr;

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
		
		/* create and seed Mersenne Twister */
		mt_rng = mtwist_new();

		mtwist_seed(mt_rng, rand());
		
		range = maxoid - minoid;
		
		/* sample OIDs (method depends on w) */
		if(cdf == NULL) {
			/* no weights, hence do uniform sampling */

			/* while we do not have enough sample OIDs yet */
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

		} else {
			/* do weighted sampling */
			
			cdf_ptr = (dbl*) Tloc(cdf, BUNfirst(cdf));
			if (!antiset)
				cdf_max = cdf_ptr[cnt-1];
			else
				cdf_max = cdf_ptr[0];
			   //TODO how to type/cast cdf_max?

			/* generate candoids, using CDF */
			for (rescnt = 0; rescnt < n; rescnt++) {
				oid candoid;

				do {
					random = mtwist_drand(mt_rng)*cdf_max;
					/* generate a new random OID in [minoid, maxoid[
					 * that is including minoid, excluding maxoid*/
					/* note that cdf has already been adjusted for antiset case */
					candoid = (oid) ( minoid + (oid) SORTfndfirst(cdf, &random) );
					/* if that candidate OID was already
					 * generated, try again */
				} while (!OIDTreeMaybeInsert(tree, candoid, rescnt));
			}
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


/* BATsample takes uniform samples of void headed BATs */
BAT *
BATsample(BAT *b, BUN n)
{
	return _BATsample(b, n, NULL);
}

/* BATweightedsample takes weighted samples of void headed BATs */
/* Note that the type of w should be castable to doubles */
BAT *
BATweightedsample(BAT *b, BUN n, BAT *w)
{
	BAT* cdf;
	BAT* sample;
	dbl* w_ptr;//TODO types of w
	dbl* cdf_ptr;
	BUN cnt, i;
	bit antiset;

	BATcheck(b, "BATsample", NULL);
	BATcheck(w, "BATsample", NULL);

	ERRORcheck(w->ttype == TYPE_str || w->ttype == TYPE_void,
					"BATsample: type of weights not castable to doubles\n", NULL);
	ERRORcheck(w->ttype != TYPE_dbl,
					"BATsample: type of weights must be doubles\n", NULL);//TODO types of w (want to remove this)

	cnt = BATcount(b);

	antiset = n > cnt / 2;

	cdf = BATnew(TYPE_void, TYPE_dbl, cnt, TRANSIENT);
	BATsetcount(cdf, cnt);
	
	/* calculate cumilative distribution function */
	w_ptr = (dbl*) Tloc(w, BUNfirst(w));//TODO support different types w
	cdf_ptr = (dbl*) Tloc(cdf, BUNfirst(cdf));

	cdf_ptr[0] = (dbl)w_ptr[0];
	for (i = 1; i < cnt; i++) {
		if((dbl)w_ptr[i] == dbl_nil) {//TODO fix NULL-test if w can have different types
			cdf_ptr[i] = cdf_ptr[i-1];
		} else {
			cdf_ptr[i] = ((dbl)w_ptr[i]) + cdf_ptr[i-1];
		}
	}
	if (!antiset) {
		cdf->tsorted = 1;
		cdf->trevsorted = cnt <= 1;
	} else {
		/* in antiset notation, we have to flip probabilities */
		for (i = 0; i < cnt; i++) {
			 cdf_ptr[i] = cdf_ptr[cnt-1] - cdf_ptr[i];
		}
		cdf->tsorted = cnt <= 1;
		cdf->trevsorted = 1;
	}
	
	/* obtain sample */
	sample = _BATsample(b, n, cdf);
	
	BATdelete(cdf);

	return sample;
}


/* BATweightedbitbat creates a bit BAT of length cnt containing n 1s and cnt-n 0s */
/* Note that the type of w should be castable to doubles */
/*BAT *
BATweightedbitbat(BUN cnt, BUN n, BAT *w)
{
	BAT* res;
	res = BATnew(TYPE_void, TYPE_dbl, cnt, TRANSIENT);
	BATsetcount(res, cnt);
	
	//Need to adjust _BATsample so it will return a bit BAT with bools denoting if element is selected
	//Now it will rather return a subset
	//TODO rewrite _BATsample to support this, add call to _BATsample
	//Why did we choose for this UDF notation?
	//+ easier to implement (no parsing addition)
	//- slow
	//- actually yields uglier code
	//Why implement something like that? Hence we should choose for the other notation?
	
	
	return res;
}
*/


