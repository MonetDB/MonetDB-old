/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/* monetdb_config.h must be the first include in each .c file */
#include "monetdb_config.h"
#include "udf.h"

/* Reverse a string */

/* actual implementation */
/* all non-exported functions must be declared static */
static char *
UDFreverse_(char **ret, const char *src)
{
	size_t len = 0;
	char *dst = NULL;

	/* assert calling sanity */
	assert(ret != NULL);

	/* handle NULL pointer and NULL value */
	if (src == NULL || strcmp(src, str_nil) == 0) {
		*ret = GDKstrdup(str_nil);
		if (*ret == NULL)
			throw(MAL, "udf.reverse",
			      "failed to create copy of str_nil");

		return MAL_SUCCEED;
	}

	/* allocate result string */
	len = strlen(src);
	*ret = dst = GDKmalloc(len + 1);
	if (dst == NULL)
		throw(MAL, "udf.reverse",
		      "failed to allocate string of length " SZFMT, len + 1);

	/* copy characters from src to dst in reverse order */
	dst[len] = 0;
	while (len > 0)
		*dst++ = src[--len];

	return MAL_SUCCEED;
}

/* MAL wrapper */
char *
UDFreverse(char **ret, const char **arg)
{
	/* assert calling sanity */
	assert(ret != NULL && arg != NULL);

	return UDFreverse_ ( ret, *arg );
}


/* Reverse a BAT of strings */
/*
 * Generic "type-oblivious" version,
 * using generic "type-oblivious" BAT access interface.
 */

/* actual implementation */
static char *
UDFBATreverse_(BAT **ret, BAT *src)
{
	BATiter li;
	BAT *bn = NULL;
	BUN p = 0, q = 0;

	/* assert calling sanity */
	assert(ret != NULL);

	/* handle NULL pointer */
	if (src == NULL)
		throw(MAL, "batudf.reverse", RUNTIME_OBJECT_MISSING);

	/* check tail type */
	if (src->ttype != TYPE_str) {
		throw(MAL, "batudf.reverse",
		      "tail-type of input BAT must be TYPE_str");
	}

	/* allocate void-headed result BAT */
	bn = BATnew(TYPE_void, TYPE_str, BATcount(src), TRANSIENT);
	if (bn == NULL) {
		throw(MAL, "batudf.reverse", MAL_MALLOC_FAIL);
	}
	BATseqbase(bn, src->hseqbase);

	/* create BAT iterator */
	li = bat_iterator(src);

	/* the core of the algorithm, expensive due to malloc/frees */
	BATloop(src, p, q) {
		char *tr = NULL, *err = NULL;

		const char *t = (const char *) BUNtail(li, p);

		/* revert tail value */
		err = UDFreverse_(&tr, t);
		if (err != MAL_SUCCEED) {
			/* error -> bail out */
			BBPunfix(bn->batCacheid);
			return err;
		}

		/* assert logical sanity */
		assert(tr != NULL);

		/* append reversed tail in result BAT */
		BUNappend(bn, tr, FALSE);		

		/* free memory allocated in UDFreverse_() */
		GDKfree(tr);
	}

	*ret = bn;

	return MAL_SUCCEED;
}

/* MAL wrapper */
char *
UDFBATreverse(bat *ret, const bat *arg)
{
	BAT *res = NULL, *src = NULL;
	char *msg = NULL;

	/* assert calling sanity */
	assert(ret != NULL && arg != NULL);

	/* bat-id -> BAT-descriptor */
	if ((src = BATdescriptor(*arg)) == NULL)
		throw(MAL, "batudf.reverse", RUNTIME_OBJECT_MISSING);

	/* do the work */
	msg = UDFBATreverse_ ( &res, src );

	/* release input BAT-descriptor */
	BBPunfix(src->batCacheid);

	if (msg == MAL_SUCCEED) {
		/* register result BAT in buffer pool */
		BBPkeepref((*ret = res->batCacheid));
	}

	return msg;
}



/* fuse */

/* instantiate type-specific functions */

#define UI bte
#define UU unsigned char
#define UO sht
#include "udf_impl.h"
#undef UI
#undef UU
#undef UO

#define UI sht
#define UU unsigned short
#define UO int
#include "udf_impl.h"
#undef UI
#undef UU
#undef UO

#define UI int
#define UU unsigned int
#define UO lng
#include "udf_impl.h"
#undef UI
#undef UU
#undef UO

#ifdef HAVE_HGE
#define UI lng
#ifdef HAVE_LONG_LONG
#define UU unsigned long long
#else
#ifdef HAVE___INT64
#define UU unsigned __int64
#endif
#endif
#define UO hge
#include "udf_impl.h"
#undef UI
#undef UU
#undef UO
#endif

/* BAT fuse */

/* actual implementation */
static char *
UDFBATfuse_(BAT **ret, const BAT *bone, const BAT *btwo)
{
	BAT *bres = NULL;
	bit two_tail_sorted_unsigned = FALSE;
	bit two_tail_revsorted_unsigned = FALSE;
	BUN n;
	char *msg = NULL;

	/* assert calling sanity */
	assert(ret != NULL);

	/* handle NULL pointer */
	if (bone == NULL || btwo == NULL)
		throw(MAL, "batudf.fuse", RUNTIME_OBJECT_MISSING);

	/* check for dense & aligned heads */
	if (!BAThdense(bone) ||
	    !BAThdense(btwo) ||
	    BATcount(bone) != BATcount(btwo) ||
	    bone->hseqbase != btwo->hseqbase) {
		throw(MAL, "batudf.fuse",
		      "heads of input BATs must be aligned");
	}
	n = BATcount(bone);

	/* check tail types */
	if (bone->ttype != btwo->ttype) {
		throw(MAL, "batudf.fuse",
		      "tails of input BATs must be identical");
	}

	/* allocate result BAT */
	switch (bone->ttype) {
	case TYPE_bte:
		bres = BATnew(TYPE_void, TYPE_sht, n, TRANSIENT);
		break;
	case TYPE_sht:
		bres = BATnew(TYPE_void, TYPE_int, n, TRANSIENT);
		break;
	case TYPE_int:
		bres = BATnew(TYPE_void, TYPE_lng, n, TRANSIENT);
		break;
#ifdef HAVE_HGE
	case TYPE_lng:
		bres = BATnew(TYPE_void, TYPE_hge, n, TRANSIENT);
		break;
#endif
	default:
		throw(MAL, "batudf.fuse",
		      "tails of input BATs must be one of {bte, sht, int"
#ifdef HAVE_HGE
		      ", lng"
#endif
		      "}");
	}
	if (bres == NULL)
		throw(MAL, "batudf.fuse", MAL_MALLOC_FAIL);

	/* call type-specific core algorithm */
	switch (bone->ttype) {
	case TYPE_bte:
		msg = UDFBATfuse_bte_sht ( bres, bone, btwo, n,
			&two_tail_sorted_unsigned, &two_tail_revsorted_unsigned );
		break;
	case TYPE_sht:
		msg = UDFBATfuse_sht_int ( bres, bone, btwo, n,
			&two_tail_sorted_unsigned, &two_tail_revsorted_unsigned );
		break;
	case TYPE_int:
		msg = UDFBATfuse_int_lng ( bres, bone, btwo, n,
			&two_tail_sorted_unsigned, &two_tail_revsorted_unsigned );
		break;
#ifdef HAVE_HGE
	case TYPE_lng:
		msg = UDFBATfuse_lng_hge ( bres, bone, btwo, n,
			&two_tail_sorted_unsigned, &two_tail_revsorted_unsigned );
		break;
#endif
	default:
		BBPunfix(bres->batCacheid);
		throw(MAL, "batudf.fuse",
		      "tails of input BATs must be one of {bte, sht, int"
#ifdef HAVE_HGE
		      ", lng"
#endif
		      "}");
	}

	if (msg != MAL_SUCCEED) {
		BBPunfix(bres->batCacheid);
	} else {
		/* set number of tuples in result BAT */
		BATsetcount(bres, n);

		/* set result properties */
		bres->hdense = TRUE;              /* result head is dense */
		BATseqbase(bres, bone->hseqbase); /* result head has same seqbase as input */
		bres->hsorted = 1;                /* result head is sorted */
		bres->hrevsorted = (BATcount(bres) <= 1);
		BATkey(bres, TRUE);               /* result head is key (unique) */

		/* Result tail is sorted, if the left/first input tail is
		 * sorted and key (unique), or if the left/first input tail is
		 * sorted and the second/right input tail is sorted and the
		 * second/right tail values are either all >= 0 or all < 0;
		 * otherwise, we cannot tell.
		 */
		if (BATtordered(bone)
		    && (BATtkey(bone) || two_tail_sorted_unsigned))
			bres->tsorted = 1;
		else
			bres->tsorted = (BATcount(bres) <= 1);
		if (BATtrevordered(bone)
		    && (BATtkey(bone) || two_tail_revsorted_unsigned))
			bres->trevsorted = 1;
		else
			bres->trevsorted = (BATcount(bres) <= 1);
		/* result tail is key (unique), iff both input tails are */
		BATkey(BATmirror(bres), BATtkey(bone) || BATtkey(btwo));

		*ret = bres;
	}

	return msg;
}

/* MAL wrapper */
char *
UDFBATfuse(bat *ires, const bat *ione, const bat *itwo)
{
	BAT *bres = NULL, *bone = NULL, *btwo = NULL;
	char *msg = NULL;

	/* assert calling sanity */
	assert(ires != NULL && ione != NULL && itwo != NULL);

	/* bat-id -> BAT-descriptor */
	if ((bone = BATdescriptor(*ione)) == NULL)
		throw(MAL, "batudf.fuse", RUNTIME_OBJECT_MISSING);

	/* bat-id -> BAT-descriptor */
	if ((btwo = BATdescriptor(*itwo)) == NULL) {
		BBPunfix(bone->batCacheid);
		throw(MAL, "batudf.fuse", RUNTIME_OBJECT_MISSING);
	}

	/* do the work */
	msg = UDFBATfuse_ ( &bres, bone, btwo );

	/* release input BAT-descriptors */
	BBPunfix(bone->batCacheid);
	BBPunfix(btwo->batCacheid);

	if (msg == MAL_SUCCEED) {
		/* register result BAT in buffer pool */
		BBPkeepref((*ires = bres->batCacheid));
	}

	return msg;
}

char* UDFqrq(bat *q, const bat *c, const double *s) {
	BAT *cBAT, *qBAT;
	BUN start, end, i;
	BATiter c_iter;
	double *qVals;
	
	if(!(cBAT = BATdescriptor(*c))) {
		return createException(MAL, "batudf.qrq", "Problem retrieving BAT");
	}

    if( !BAThdense(cBAT) ) {
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "batudf.qrq", "BATs must have dense heads");
    }

	c_iter = bat_iterator(cBAT);
	start = BUNfirst(cBAT);
	end = BUNlast(cBAT);

	if(!(qBAT = BATnew(TYPE_void, TYPE_dbl, BATcount(cBAT), TRANSIENT))) {
		BBPunfix(cBAT->batCacheid);
		return createException(MAL, "batudf.qrq", MAL_MALLOC_FAIL);
	}

	BATseqbase(qBAT, cBAT->hseqbase);

	qVals = (double*)Tloc(qBAT, BUNfirst(qBAT));
	for(i=start; i<end; i++) {
		*(qVals++) = *(double*)BUNtail(c_iter, i)/(*s);
	}

	BATsetcount(qBAT, BATcount(cBAT));
	qBAT->tsorted = 0;
	qBAT->trevsorted = 0;
	qBAT->tkey = 0;
	qBAT->tdense = 0;

	BBPunfix(cBAT->batCacheid);
	BBPkeepref(*q = qBAT->batCacheid);

	return MAL_SUCCEED;
}

static double computeR(BAT *col) {
	BATiter col_iter = bat_iterator(col);
	BUN start = BUNfirst(col);
	BUN last = BUNlast(col);
	BUN i;
	double s = 0.0;

	for(i=start; i<last; i++) {
		double val = *(double*)BUNtail(col_iter, i);
		s += val*val;
	}

	return sqrt(s);
}

static double computeR2(BAT *col, BAT *q) {
	BATiter col_iter = bat_iterator(col);
	BATiter q_iter = bat_iterator(q);
	BUN start = BUNfirst(col);
	BUN last = BUNlast(col);
	BUN i;
	double s = 0.0;

	for(i=start; i<last; i++) {
		s += (*(double*)BUNtail(col_iter, i))*(*(double*)BUNtail(q_iter, i));
	}

	return s;
}

static BAT* setQ(BAT *col, double r) {
	BATiter col_iter = bat_iterator(col);
	double *vals;
	BUN i, start, end;

	BAT *q = BATnew(TYPE_void, TYPE_dbl, BATcount(col), TRANSIENT);
	if(!q)
		return NULL;

	start = BUNfirst(col);
	end = BUNlast(col);
	vals = (double*)Tloc(q, BUNfirst(q));

	for(i=start; i<end; i++) {
		double val = *(double*)BUNtail(col_iter, i);
		*(vals++) = val/r;
	}

	BATsetcount(q, BATcount(col));
	BATseqbase(q, col->hseqbase);
	q->trevsorted = q->tsorted = 0;
    q->tkey = 0;
    q->T->nil = 1;
    q->T->nonil = 0;
	return q;
}

static BAT* updateCol(BAT *col, double r, BAT* q) {
	double *colVals, *qVals;
	oid i;

	colVals = (double*)Tloc(col, BUNfirst(col));
	qVals = (double*)Tloc(q, BUNfirst(q));

	for(i=0; i<BATcount(col); i++) {
		colVals[i] -= r*qVals[i];
	}

	return col;	
}

char* qrUDF_bulk(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci) {
	/* iterate over tha columns */
	int colNum =0;
	int colsNum = pci->argc-pci->retc;
	/* we need to update the column without destroying the original relation */
	BAT **cols_copy= (BAT**)malloc(sizeof(BAT*)*colsNum);
	/* the first input column is the rowN */
	BAT *rowsCol = BATdescriptor(*getArgReference_bat(stk, pci, pci->retc));
	BAT *rowsCol_sorted = NULL, *order = NULL;
	bat *rowsRes;


	(void)cntxt;
	(void)mb;

	if(!rowsCol)
		return createException(MAL, "udf.qr", MAL_MALLOC_FAIL);
	
	/* sort all columns according to rowNum */
	if(BATsubsort(&rowsCol_sorted, &order, NULL, rowsCol, NULL, NULL, 0, 0) != GDK_SUCCEED) {
		BBPunfix(rowsCol->batCacheid);
		return createException(MAL, "udf.qr", "Problem in BATsubsort");
	}
	BBPunfix(rowsCol->batCacheid);

	/* copy the columns  and sort them at the same time*/
	for(colNum=1; colNum<colsNum; colNum++) {
		BAT *col = BATdescriptor(*getArgReference_bat(stk, pci, colNum+pci->retc));
		if (col == NULL) {
			int i;
			BBPunfix(rowsCol_sorted->batCacheid);
			for(i=0; i<colNum; i++)
				BBPunfix(cols_copy[colNum]->batCacheid);
        	return createException(MAL, "udf.qr", MAL_MALLOC_FAIL);
        }

		cols_copy[colNum] = BATproject(order, col);  //BATcopy(col, TYPE_void, BATttype(col), FALSE, TRANSIENT);
		if (cols_copy[colNum] == NULL) {
			int i;
			BBPunfix(rowsCol_sorted->batCacheid);
			for(i=0; i<=colNum; i++)
				BBPunfix(cols_copy[colNum]->batCacheid);
        	return createException(MAL, "udf.qr", "BATproject error");
        }
		BBPunfix(col->batCacheid);
	}

	/* process the columns */
	for(colNum=1 ; colNum<colsNum ; colNum++) {
		int colNum2 =0 ;
		BAT *col = cols_copy[colNum];
		double s = computeR(col);
		bat *res = getArgReference_bat(stk, pci, colNum);
		BAT *q = setQ(col,s);

		if(q == NULL) {
			int i;
			BBPunfix(rowsCol_sorted->batCacheid);
			for(i=0; i<colsNum; i++)
				BBPunfix(cols_copy[i]->batCacheid);
			//TODO: clean res also
			return createException(MAL, "udf.qr", "Problem in setQ");
		}

		/* update the other columns */
		for(colNum2 = colNum+1; colNum2<colsNum; colNum2++) {
			s = computeR2(cols_copy[colNum2], q);
			cols_copy[colNum2] = updateCol(cols_copy[colNum2], s, q);
		}

		BBPkeepref(*res = q->batCacheid);
		/* we do not need this column anymore */
		BBPunfix(cols_copy[colNum]->batCacheid);
	}

	rowsRes = getArgReference_bat(stk, pci, 0);
	BBPkeepref(*rowsRes = rowsCol_sorted->batCacheid);

    return MAL_SUCCEED;
}

char* narrowqrUDF_bulk(bat *rowsRes, bat* columnsRes, bat* valuesRes, const bat *rows, const bat *columns, const bat *values) {
	BAT *cBAT_in, *cBAT_out, *rBAT_in, *rBAT_out, *vBAT_in, *vBAT_out;

	BAT *rBAT_sorted, *cBAT_sortedR, *vBAT_sortedR;
	BAT *order;

	int *minColNum, *maxColNum, colNum, colsNum;
	oid i,j;

	BAT **cols;
	BAT **q;

	int *rVals, *cVals;
	double *vVals, *vcVals;

	BUN totalElementsNum;

	if(!(rBAT_in = BATdescriptor(*rows)))
		return createException(MAL, "udf.qr", MAL_MALLOC_FAIL);

	if(!(cBAT_in = BATdescriptor(*columns))) {
		BBPunfix(rBAT_in->batCacheid);
		return createException(MAL, "udf.qr", MAL_MALLOC_FAIL);
	}

	if(!(vBAT_in = BATdescriptor(*values))) {
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		return createException(MAL, "udf.qr", MAL_MALLOC_FAIL);

	}
	
	totalElementsNum = BATcount(vBAT_in);
	
	/*get the minimum and maximum column number */
	minColNum = (int*)BATmin(cBAT_in, NULL);
	maxColNum = (int*)BATmax(cBAT_in, NULL);
	if(!minColNum || !maxColNum) {
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		BBPunfix(vBAT_in->batCacheid);

		return createException(MAL, "udf.qr", "Problem in BATmin or BATmax");
	}
	colsNum = *maxColNum-*minColNum+1;

	/* sort the values according to rows */
	if(BATsubsort(&rBAT_sorted, &order, NULL, rBAT_in, NULL, NULL, 0, 0) != GDK_SUCCEED) {
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		BBPunfix(vBAT_in->batCacheid);
		return createException(MAL, "udf.qr", "Problem in BATsubsort");
	}

	if(!(cBAT_sortedR = BATproject(order, cBAT_in))) { 
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		BBPunfix(vBAT_in->batCacheid);

		BBPunfix(rBAT_sorted->batCacheid);
	
		return createException(MAL, "udf.qr", "Problem in BATproject");
	}

	if(!(vBAT_sortedR = BATproject(order, vBAT_in))) {
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		BBPunfix(vBAT_in->batCacheid);

		BBPunfix(rBAT_sorted->batCacheid);
		BBPunfix(cBAT_sortedR->batCacheid);

		return createException(MAL, "udf.qr", "Problem in BATproject");
	}
	
	BBPunfix(order->batCacheid);
	
	cols = (BAT**)GDKmalloc(sizeof(BAT*)*colsNum);
	q = (BAT**)GDKmalloc(sizeof(BAT*)*colsNum);
	if(!cols || !q) {
		BBPunfix(rBAT_in->batCacheid);
		BBPunfix(cBAT_in->batCacheid);
		BBPunfix(vBAT_in->batCacheid);
	
		BBPunfix(rBAT_sorted->batCacheid);
		BBPunfix(cBAT_sortedR->batCacheid);
		BBPunfix(vBAT_sortedR->batCacheid);

		if(cols)
			GDKfree(cols);

		return createException(MAL, "udf.qr", "Problem allocating space");
	}
	
	/* create a copy of all the columns */
	for(i=0,colNum=*minColNum; colNum<=*maxColNum; i++, colNum++) {
		BAT *vals;
		/*get the oids that correspond to this colNum */
		BAT *colOidsBAT = BATsubselect(cBAT_sortedR, NULL, &colNum, &colNum, 1, 1, 0);
		if(colOidsBAT == NULL) {
			BBPunfix(rBAT_in->batCacheid);
			BBPunfix(cBAT_in->batCacheid);
			BBPunfix(vBAT_in->batCacheid);

			BBPunfix(rBAT_sorted->batCacheid);
			BBPunfix(cBAT_sortedR->batCacheid);
			BBPunfix(vBAT_sortedR->batCacheid);

			return createException(MAL, "udf.qr", "Problem in BATsubselect");
		}

		/* get the values that correspond to these oids */
		if(!(vals = BATproject(colOidsBAT, vBAT_sortedR))) {
			BBPunfix(rBAT_in->batCacheid);
			BBPunfix(cBAT_in->batCacheid);
			BBPunfix(vBAT_in->batCacheid);

			BBPunfix(rBAT_sorted->batCacheid);
			BBPunfix(cBAT_sortedR->batCacheid);
			BBPunfix(vBAT_sortedR->batCacheid);
	
			for(j=0; j<i; j++)
				BBPunfix(cols[j]->batCacheid);

			return createException(MAL, "udf.qr", "Problem in BATproject");
		}

		cols[i] = vals;

		/*create a BAT for the output */
		q[i] = BATnew(TYPE_void, BATttype(vals), BATcount(vals), TRANSIENT);
		BBPunfix(colOidsBAT->batCacheid)
	}

	/* I do not need these BATs anymore */
	BBPunfix(rBAT_sorted->batCacheid);
	BBPunfix(cBAT_sortedR->batCacheid);
	BBPunfix(vBAT_sortedR->batCacheid);

	/*start the algorithm*/
	for(colNum=0 ; colNum<colsNum; colNum++) {
		int colNum2 =0 ;
        BAT *col = cols[colNum];
        double s = computeR(col);
        q[colNum] = setQ(col,s);

        if(q[colNum] == NULL) {
			BBPunfix(rBAT_in->batCacheid);
			BBPunfix(cBAT_in->batCacheid);
			BBPunfix(vBAT_in->batCacheid);

			for(i=0; i<(unsigned int)colsNum; i++)
				BBPunfix(cols[i]->batCacheid);
			for(i=0; i<(unsigned int)colNum; i++)
				BBPunfix(q[i]->batCacheid);

			GDKfree(cols);
			GDKfree(q);

            return createException(MAL, "udf.qr", "Problem in setQ");
        }
            
        /* update the other columns */
        for(colNum2 = colNum+1; colNum2<colsNum; colNum2++) {
        	s = computeR2(cols[colNum2], q[colNum]);
            cols[colNum2] = updateCol(cols[colNum2], s, q[colNum]);
        }
	}

	/* merge the results in a single column */
	/* the results are ordered by column and subortered by row
 	* because this is how we processed them */
	rBAT_out = BATnew(TYPE_void, BATttype(rBAT_in), totalElementsNum, TRANSIENT);
	cBAT_out = BATnew(TYPE_void, BATttype(cBAT_in), totalElementsNum, TRANSIENT);
	vBAT_out = BATnew(TYPE_void, BATttype(vBAT_in), totalElementsNum, TRANSIENT);

	rVals = (int*)Tloc(rBAT_out, BUNfirst(rBAT_out));
	cVals = (int*)Tloc(cBAT_out, BUNfirst(cBAT_out));
	vVals = (double*)Tloc(vBAT_out, BUNfirst(vBAT_out));

	for(i=0, colNum=0; colNum<colsNum; colNum++) {
		vcVals = (double*)Tloc(q[colNum], BUNfirst(q[colNum]));
		
		for(j=0; j<BATcount(q[colNum]); j++, i++) {
			rVals[i] = j;
			cVals[i] = colNum;
			vVals[i] = vcVals[j];
		}

		BBPunfix(cols[colNum]->batCacheid);
		BBPunfix(q[colNum]->batCacheid);
	}

	GDKfree(cols);
	GDKfree(q);

	BATsetcount(rBAT_out, i);
    BATseqbase(rBAT_out, 0);
    rBAT_out->trevsorted = rBAT_out->tsorted = 0;
    rBAT_out->tkey = 0;
    rBAT_out->T->nil = 0;
    rBAT_out->T->nonil = 0;

	BATsetcount(cBAT_out, i);
    BATseqbase(cBAT_out, 0);
    cBAT_out->trevsorted = cBAT_out->tsorted = 0;
    cBAT_out->tkey = 0;
    cBAT_out->T->nil = 0;
    cBAT_out->T->nonil = 0;

	BATsetcount(vBAT_out, i);
    BATseqbase(vBAT_out, 0);
    vBAT_out->trevsorted = vBAT_out->tsorted = 0;
    vBAT_out->tkey = 0;
    vBAT_out->T->nil = 0;
    vBAT_out->T->nonil = 0;

	BBPunfix(rBAT_in->batCacheid);
	BBPunfix(cBAT_in->batCacheid);
	BBPunfix(vBAT_in->batCacheid);


	BBPkeepref(*rowsRes = rBAT_out->batCacheid);
	BBPkeepref(*columnsRes = cBAT_out->batCacheid);
	BBPkeepref(*valuesRes = vBAT_out->batCacheid);

	return MAL_SUCCEED;
}

