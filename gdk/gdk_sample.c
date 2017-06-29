/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2017 MonetDB B.V.
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

#include <math.h>

#include "mtwist.h"

//TODO share these with gkd_firstn.c
#define siftup(OPER, START, SWAP)					\
	do {								\
		pos = (START);						\
		childpos = (pos << 1) + 1;				\
		while (childpos < n) {					\
			/* find most extreme child */			\
			if (childpos + 1 < n &&				\
			    !(OPER(childpos + 1, childpos)))		\
				childpos++;				\
			/* compare parent with most extreme child */	\
			if (!OPER(pos, childpos)) {			\
				/* already correctly ordered */		\
				break;					\
			}						\
			/* exchange parent with child and sift child */	\
			/* further */					\
			SWAP(pos, childpos);				\
			pos = childpos;					\
			childpos = (pos << 1) + 1;			\
		}							\
	} while (0)

#define heapify(OPER, SWAP)				\
	do {						\
		for (i = n / 2; i > 0; i--)		\
			siftup(OPER, i - 1, SWAP);	\
	} while (0)


/* CUSTOM SWAP AND COMPARE FUNCTION */
#define SWAP3(p1, p2)				\
	do {					\
		item = oids[p1];		\
		oids[p1] = oids[p2];		\
		oids[p2] = item;		\
		key = keys[p1];		\
		keys[p1] = keys[p2];		\
		keys[p2] = key;		\
	} while (0)

#define compKeysGT(p1, p2)							\
	((keys[p1] > keys[p2]) || 						\
	((keys[p1] == keys[p2]) && (oids[p1] > oids[p2])))

/* this is a straightforward implementation of a binary tree */
struct oidtreenode {
	oid o;
	struct oidtreenode *left;
	struct oidtreenode *right;
};

mtwist *mt_rng = NULL;

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
	((oid *) bat->theap.base)[bat->batCount++] = node->o;
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
			((oid *) bat->theap.base)[bat->batCount++] = noid;

	if (node->right != NULL)
		OIDTreeToBATAntiset(node->right, bat, node->o + 1, stop);
	else
		for (noid = node->o+1; noid < stop; noid++)
			((oid *) bat->theap.base)[bat->batCount++] = noid;
}


/* BATsample takes uniform samples of void headed BATs */
BAT *
BATsample(BAT *b, BUN n)
{
	BAT *bn;
	BUN cnt, slen;
	BUN rescnt;
	struct oidtreenode *tree = NULL;

	BUN range;

	BATcheck(b, "BATsample", NULL);
	ERRORcheck(n > BUN_MAX, "BATsample: sample size larger than BUN_MAX\n", NULL);
	ALGODEBUG
		fprintf(stderr, "#BATsample: sample " BUNFMT " elements.\n", n);

	cnt = BATcount(b);
	/* empty sample size */
	if (n == 0) {
		bn = COLnew(0, TYPE_void, 0, TRANSIENT);
		if (bn == NULL) {
			return NULL;
		}
		BATsetcount(bn, 0);
		BATtseqbase(bn, 0);
	/* sample size is larger than the input BAT, return all oids */
	} else if (cnt <= n) {
		bn = COLnew(0, TYPE_void, cnt, TRANSIENT);
		if (bn == NULL) {
			return NULL;
		}
		BATsetcount(bn, cnt);
		BATtseqbase(bn, b->hseqbase);
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
		bn = COLnew(0, TYPE_oid, slen, TRANSIENT);
		if (bn == NULL) {
			GDKfree(tree);
			return NULL;
		}
		
		if(!mt_rng) {
			/* create and seed Mersenne Twister */
			mt_rng = mtwist_new();

			mtwist_seed(mt_rng, rand());
		}
		
		range = (BUN) (maxoid - minoid);
		
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
			bn->tseqbase = *(oid *) Tloc(bn, 0);
	}
	return bn;
}

/* BATweightedsample takes weighted samples of void headed BATs */
/* Note that the type of w should be castable to doubles */
/* based on Alg-A-exp from 'Weighted random sampling with a reservoir' by Efraimidis and Spirakis (2006) */
BAT *
BATweightedsample(BAT *b, BUN n, BAT *w)
{//TODO test correctness extensively
	BAT* weights = NULL;
	bit weights_are_cast = 0;
	BAT* sample = NULL;
	oid* oids = NULL;  /* points to the oids in sample */
	dbl* w_ptr = NULL; //TODO types of w
	dbl* keys = NULL;  /* keys as defined in Alg-A-exp */
	BUN cnt, i, j;
	BUN pos, childpos;
	oid item;
	dbl r, xw, r2, key, tw;

	oid minoid = b->hseqbase;

	ERRORcheck(n > BATcount(b), "BATsample: Sample size bigger than table!", NULL);

	BATcheck(b, "BATsample", NULL);
	BATcheck(w, "BATsample", NULL);

	ERRORcheck(w->ttype == TYPE_str || w->ttype == TYPE_void,
					"BATsample: type of weights not castable to doubles\n", NULL);

	if(w->ttype != TYPE_dbl) {
		weights = BATconvert(w, NULL, TYPE_dbl, 0);
		ERRORcheck(weights == NULL, "BATsample: could not cast weights to doubles\n", NULL);
		weights_are_cast = 1;
	} else {
		weights = w;
		weights_are_cast = 0;
	}
	//ERRORcheck(w->ttype != TYPE_dbl,
	//				"BATsample: type of weights must be doubles\n", NULL);//TODO types of w (want to remove this)
	//TODO: handle NULL values in w_ptr

	cnt = BATcount(b);

	sample = COLnew(0, TYPE_oid, n, TRANSIENT);
	if(!sample)
		goto bailout;

	if(n == 0) {
		if(weights_are_cast)
			BBPunfix(weights->batCacheid);
		return sample;
	}

	keys = (dbl*) GDKmalloc(sizeof(dbl)*n);
	if(!keys)
		goto bailout;

	oids = (oid *) Tloc(sample, 0);
	w_ptr = (dbl*) Tloc(weights, 0);

	if(!mt_rng) {
		mt_rng = mtwist_new();
		mtwist_seed(mt_rng, rand());
	}

	BATsetcount(sample, n);
	/* obtain sample */

	/* Initialize prioqueue */
	i = 0; /* i indices the initial sample (filled with elements with non-zero weight) */
		  /* j indices the oids and weights */
	for(j = 0; i < n && j < cnt; j++) {
		if(w_ptr[j] == 0.0)
			continue;
		if(w_ptr[j] < 0.0) {
			GDKerror("BATsample: w contains negative weights\n");
			goto bailout;
		}
		oids[i] = (oid)(j + minoid);
		keys[i] = pow(mtwist_drand(mt_rng), 1.0 / w_ptr[j]);//TODO cast 1.0 to dbl?
		if (keys[i] == 1) {
			GDKerror("BATsample: weight overflow\n");
			goto bailout;
		}
		i++;
	}

	if(i < n) {
		GDKerror("BATsample: sample size bigger than number of non-zero weights\n");
		goto bailout;
	}

	heapify(compKeysGT, SWAP3);

	while(true) {
		r = mtwist_drand(mt_rng);
		xw = log(r)/log(keys[0]);
		for(; j < cnt && xw >= w_ptr[j]; j++) {
			if(w_ptr[j] < 0.0) {
				GDKerror("BATsample: w contains negative weights\n");
				goto bailout;
			}
			xw -= w_ptr[j];
		}
		if(j >= cnt) break;

		/* At this point:
		 * 		w_ptr[c]+w_ptr[c+1]+...+w_ptr[i-1]
		 *   <  xw (the initial value, log(r)/log(keys[0]))
		 * 	 <= w_ptr[c]+w_ptr[c+1]+...+w_ptr[i] */
		tw = pow(keys[0], w_ptr[j]);
		r2 = mtwist_drand(mt_rng)*(1-tw)+tw;
		key = pow(r2, 1/w_ptr[j]);

		/* Replace element with lowest key in prioqueue */
		oids[0] = (oid)(j+minoid);
		keys[0] = key;
		siftup(compKeysGT, 0, SWAP3);

		j++;/* Increment j so j=c (c is defined in Alg-A-exp) */
	}

	GDKfree(keys);
	if(weights_are_cast)
		BBPunfix(weights->batCacheid);

	sample->trevsorted = sample->batCount <= 1;
	sample->tsorted = sample->batCount <= 1;
	sample->tkey = 1;
	sample->tdense = sample->batCount <= 1;
	if (sample->batCount == 1)
		sample->tseqbase = *(oid *) Tloc(sample, 0);

	return sample;

bailout:
    if(weights_are_cast && weights)//if weights where converted, delete converted BAT
    	BBPunfix(weights->batCacheid);
    if(keys)
    	GDKfree(keys);
    if(sample)
    	BBPunfix(sample->batCacheid);
    return NULL;
}



