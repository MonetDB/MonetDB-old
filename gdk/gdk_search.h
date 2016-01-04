/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

#ifndef _GDK_SEARCH_H_
#define _GDK_SEARCH_H_
/*
 * @+ Hash indexing
 *
 * This is a highly efficient implementation of simple bucket-chained
 * hashing.
 *
 * In the past, we used integer modulo for hashing, with bucket chains
 * of mean size 4.  This was shown to be inferior to direct hashing
 * with integer anding. The new implementation reflects this.
 */
gdk_export void HASHdestroy(BAT *b);
gdk_export BUN HASHprobe(Hash *h, const void *v);
gdk_export BUN HASHlist(Hash *h, BUN i);


#define HASHfnd_str(x,y,z)						\
	do {								\
		(x) = BUN_NONE;						\
		if (BAThash((y).b) == GDK_SUCCEED) {			\
			BUN _i;						\
			BUN _prb = HASHprobe((y).b->T->hash, (z));	\
			int _pcs;					\
			HASHloop_str((y), (y).b->T->hash, _prb, (z), _i, _pcs) { \
				(x) = _i;				\
				_pcs = 0; /* causes exit of outer loop */ \
				break;					\
			}						\
		} else							\
			goto hashfnd_failed;				\
	} while (0)
#define HASHfnd(x,y,z)							\
	do {								\
		(x) = BUN_NONE;						\
		if (BAThash((y).b) == GDK_SUCCEED) {			\
			BUN _i;						\
			BUN _prb = HASHprobe((y).b->T->hash, (z));	\
			int _pcs;					\
			HASHloop((y), (y).b->T->hash, _prb, (z), _i, _pcs) { \
				(x) = _i;				\
				_pcs = 0; /* causes exit of outer loop */ \
				break;					\
			}						\
		} else							\
			goto hashfnd_failed;				\
	} while (0)
#define HASHfnd_TYPE(x,y,z,TYPE)					\
	do {								\
		(x) = BUN_NONE;						\
		if (BAThash((y).b) == GDK_SUCCEED) {			\
			BUN _i;						\
			BUN _prb = HASHprobe((y).b->T->hash, (z));	\
			int _pcs;					\
			HASHloop_##TYPE((y), (y).b->T->hash, _prb, (z), _i, _pcs) { \
				(x) = _i;				\
				_pcs = 0; /* causes exit of outer loop */ \
				break;					\
			}						\
		} else							\
			goto hashfnd_failed;				\
	} while (0)
#define HASHfnd_bte(x,y,z)	HASHfnd_TYPE(x,y,z,bte)
#define HASHfnd_sht(x,y,z)	HASHfnd_TYPE(x,y,z,sht)
#define HASHfnd_int(x,y,z)	HASHfnd_TYPE(x,y,z,int)
#define HASHfnd_lng(x,y,z)	HASHfnd_TYPE(x,y,z,lng)
#ifdef HAVE_HGE
#define HASHfnd_hge(x,y,z)	HASHfnd_TYPE(x,y,z,hge)
#endif

/* Functions to perform a binary search on a sorted BAT.
 * See gdk_search.c for details. */
gdk_export BUN SORTfnd(BAT *b, const void *v);
gdk_export BUN SORTfndfirst(BAT *b, const void *v);
gdk_export BUN SORTfndlast(BAT *b, const void *v);

#endif /* _GDK_SEARCH_H_ */
