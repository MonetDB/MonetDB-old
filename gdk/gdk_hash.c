/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/* Hash table maintenance */

#include "monetdb_config.h"
#include "gdk.h"
#include "gdk_private.h"

#define HASH_VERSION		2 /* if mismatch, discard */
#define HASH_HEADER_SIZE	6 /* nr of size_t fields in header */

/* we won't split the table into pieces smaller than HASHMINSIZE */
#define HASHMINSIZE		(1 << 20)

/* we won't split the table into more than HASHMAXCHUNKS number of pieces */
#define HASHMAXCHUNKSLOG	4
#define HASHMAXCHUNKS		(1 << HASHMAXCHUNKSLOG)

/* #define DISABLE_HASHSYNC */

static BUN
HASHmask(BUN cnt, int tpe)
{
	BUN m;

	if (BATatoms[tpe].atomHash == BATatoms[TYPE_bte].atomHash)
		return 1 << 8;
	else if (BATatoms[tpe].atomHash == BATatoms[TYPE_sht].atomHash)
		return 1 << 16;
	/* find largest power of 2 smaller than or equal to cnt */
	m = cnt;
	m |= m >> 1;
	m |= m >> 2;
	m |= m >> 4;
	m |= m >> 8;
	m |= m >> 16;
#ifdef BUN8
	m |= m >> 32;
#endif
	m -= m >> 1;

	/* if cnt is more than 1/3 into the gap between m and 2*m,
	   double m */
	if (m + m - cnt < 2 * (cnt - m))
		m += m;
	if (m < BATTINY)
		m = BATTINY;	/* minimum size */
	return m;
}

BUN
HASHprobe(Hash *h, const void *v)
{
	return ATOMhash(h->type, v) & h->mask;
}

void
HASHins(BAT *b, BUN i, const void *v)
{
	Hash *h = b->T->hash;
	if (h) {
		/* must be last piece */
		assert(i >= (h->pieces - 1) * h->chunk);

		if (i >= h->cap) {
			/* doesn't fit anymore, so destroy hash */
			HASHdestroy(b);
		} else {
			BUN c = ATOMhash(b->ttype, v) & h->mask;
			if (((size_t *) h->heap->base)[0] & 1 << 24) {
				int fd;
				if ((fd = GDKfdlocate(h->heap->farmid, h->heap->filename, "rb+", NULL)) >= 0) {
					((size_t *) h->heap->base)[0] &= ~(1 << 24);
					if (write(fd, h->heap->base, SIZEOF_SIZE_T) < 0)
						perror("write hash");
					if (!(GDKdebug & FORCEMITOMASK)) {
#if defined(NATIVE_WIN32)
						_commit(fd);
#elif defined(HAVE_FDATASYNC)
						fdatasync(fd);
#elif defined(HAVE_FSYNC)
						fsync(fd);
#endif
					}
					close(fd);
				}
			}
			switch (h->width) {
			case BUN2:
				((BUN2type *)h->Link)[i] = ((BUN2type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c];
				((BUN2type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c] = (BUN2type) i;
				break;
			case BUN4:
				((BUN4type *)h->Link)[i] = ((BUN4type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c];
				((BUN4type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c] = (BUN4type) i;
				break;
#ifdef BUN8
			case BUN8:
				((BUN8type *)h->Link)[i] = ((BUN8type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c];
				((BUN8type *)h->Hash)[(h->pieces - 1) * (h->mask + 1) + c] = (BUN8type) i;
				break;
#endif
			}
			ALGODEBUG fprintf(stderr, "#HASHins(b=%s)\n", BATgetId(b));
		}
	}
}

static int
HASHwidth(BUN hashsize)
{
	if (hashsize <= (BUN) BUN2_NONE)
		return BUN2;
#ifndef BUN8
	return BUN4;
#else
	if (hashsize <= (BUN) BUN4_NONE)
		return BUN4;
	return BUN8;
#endif
}

Hash *
HASHnew(Heap *hp, int tpe, int pieces, BUN cap, BUN count)
{
	Hash *h;
	int width = HASHwidth(cap);
	BUN chunk = count / pieces;
	BUN mask = HASHmask(chunk, tpe);

	if (count == 0 || pieces == 1) {
		assert(pieces == 1);
		chunk = cap;
	}
	h = GDKmalloc(sizeof(Hash));
	if (h == NULL)
		return NULL;
	if (HEAPalloc(hp, pieces * mask + cap + HASH_HEADER_SIZE * SIZEOF_SIZE_T / width, width) != GDK_SUCCEED) {
		GDKfree(h);
		return NULL;
	}
	hp->free = (pieces * mask + cap) * width + HASH_HEADER_SIZE * SIZEOF_SIZE_T;
	h->type = tpe;
	h->width = width;
	h->pieces = pieces;
	h->chunk = chunk;
	assert(chunk > 0);
	h->cap = cap;
	h->mask = mask - 1;
	assert((mask & h->mask) == 0); /* mask is power of two */
	h->Link = hp->base + HASH_HEADER_SIZE * SIZEOF_SIZE_T;
	switch (width) {
	case BUN2:
		h->Hash = (void *) ((BUN2type *) h->Link + h->cap);
		h->nil = (BUN) BUN2_NONE;
		break;
	case BUN4:
		h->Hash = (void *) ((BUN4type *) h->Link + h->cap);
		h->nil = (BUN) BUN4_NONE;
		break;
#ifdef BUN8
	case BUN8:
		h->Hash = (void *) ((BUN8type *) h->Link + h->cap);
		h->nil = (BUN) BUN8_NONE;
		break;
#endif
	default:
		assert(0);
	}
	h->heap = hp;
	((size_t *) hp->base)[0] = HASH_VERSION;
	((size_t *) hp->base)[1] = (tpe << 16) | (pieces << 8) | width;
	((size_t *) hp->base)[2] = chunk;
	((size_t *) hp->base)[3] = cap;
	((size_t *) hp->base)[4] = mask;
	((size_t *) hp->base)[5] = count;
#ifndef NDEBUG
	/* only clear to make valgrind happy */
	memset(h->Link, 0, cap * width);
#endif
	/* initialize the hash buckets with BUN?_NONE (all versions of
	 * which have all bits set by design) */
	memset(h->Hash, 0xFF, pieces * mask * width);
	ALGODEBUG fprintf(stderr, "#HASHnew: create hash(tpe %s, pieces %d, chunk " BUNFMT ", cap " BUNFMT ", mask " BUNFMT ", width %d, total " SZFMT " bytes);\n", ATOMname(tpe), pieces, chunk, cap, mask, width, hp->free);
	return h;
}

/* return TRUE if we have a hash on the tail, even if we need to read
 * one from disk */
int
BATcheckhash(BAT *b)
{
	int ret;
	lng t;

	t = GDKusec();
	MT_lock_set(&GDKhashLock(abs(b->batCacheid)), "BATcheckhash");
	t = GDKusec() - t;
	if (b->T->hash == NULL) {
		Hash *h;
		Heap *hp;
		const char *nme = BBP_physical(b->batCacheid);
		const char *ext = b->batCacheid > 0 ? "thash" : "hhash";
		int fd;

		if ((hp = GDKzalloc(sizeof(*hp))) != NULL &&
		    (hp->farmid = BBPselectfarm(b->batRole, b->ttype, hashheap)) >= 0 &&
		    (hp->filename = GDKmalloc(strlen(nme) + 12)) != NULL) {
			sprintf(hp->filename, "%s.%s", nme, ext);

			/* check whether a persisted hash can be found */
			if ((fd = GDKfdlocate(hp->farmid, nme, "rb+", ext)) >= 0) {
				size_t hdata[HASH_HEADER_SIZE];
				struct stat st;

				if ((h = GDKmalloc(sizeof(*h))) != NULL &&
				    read(fd, hdata, sizeof(hdata)) == sizeof(hdata) &&
				    hdata[0] == (((size_t) 1 << 24) | HASH_VERSION) &&
				    hdata[5] == (size_t) BATcount(b) &&
				    (h->type = (sht) (hdata[1] >> 16)) == b->ttype &&
				    fstat(fd, &st) == 0 &&
				    (hp->size = (size_t) st.st_size) >= (hp->free = ((hdata[1] >> 8 & 0xFF) * hdata[4] + hdata[3]) * (hdata[1] & 0xFF) + HASH_HEADER_SIZE * SIZEOF_SIZE_T) &&
				    HEAPload(hp, nme, ext, 0) == GDK_SUCCEED) {
					h->width = (sht) (hdata[1] & 0xFF);
					h->pieces = (sht) (hdata[1] >> 8 & 0xFF);
					h->chunk = hdata[2];
					h->cap = hdata[3];
					h->mask = (BUN) (hdata[4] - 1);
					h->heap = hp;
					h->Link = hp->base + HASH_HEADER_SIZE * SIZEOF_SIZE_T;
					switch (h->width) {
					case BUN2:
						h->Hash = (void *) ((BUN2type *) h->Link + h->cap);
						h->nil = (BUN) BUN2_NONE;
						break;
					case BUN4:
						h->Hash = (void *) ((BUN4type *) h->Link + h->cap);
						h->nil = (BUN) BUN4_NONE;
						break;
#ifdef BUN8
					case BUN8:
						h->Hash = (void *) ((BUN8type *) h->Link + h->cap);
						h->nil = (BUN) BUN8_NONE;
						break;
#endif
					default:
						HEAPfree(hp, 1);
						goto unusable;
					}
					close(fd);
					b->T->hash = h;
					ALGODEBUG fprintf(stderr, "#BATcheckhash: reusing persisted hash %d\n", b->batCacheid);
					MT_lock_unset(&GDKhashLock(abs(b->batCacheid)), "BATcheckhash");
					return 1;
				}
			  unusable:
				GDKfree(h);
				close(fd);
				/* unlink unusable file */
				GDKunlink(hp->farmid, BATDIR, nme, ext);
			}
			GDKfree(hp->filename);
		}
		GDKfree(hp);
	}
	ret = b->T->hash != NULL;
	MT_lock_unset(&GDKhashLock(abs(b->batCacheid)), "BATcheckhash");
	ALGODEBUG if (ret) fprintf(stderr, "#BATcheckhash: already has hash %d, waited " LLFMT " usec\n", b->batCacheid, t);
	return ret;
}

#define parthashloop(TYPE, N)						\
	do {								\
		const TYPE *restrict v = (const TYPE *) Tloc(b, 0);	\
		BUN##N##type *restrict links##N;			\
		BUN##N##type *restrict hashes##N;			\
		links##N = (BUN##N##type *) h->Link;			\
		hashes##N = (BUN##N##type *) h->Hash + piece * (mask + 1); \
		while (start < end) {					\
			c = (BUN) mix_##TYPE(v[start]) & mask;		\
			links##N[start] = hashes##N[c];			\
			hashes##N[c] = (BUN##N##type) start;		\
			start++;					\
		}							\
	} while (0)

#define parthashanyloop(N)						\
	do {								\
		BUN##N##type *restrict links##N;			\
		BUN##N##type *restrict hashes##N;			\
		links##N = (BUN##N##type *) h->Link;			\
		hashes##N = (BUN##N##type *) h->Hash + piece * (mask + 1); \
		while (start < end) {					\
			const void *v = BUNtail(bi, start);		\
			c = (*hashf)(v) & mask;				\
			links##N[start] = hashes##N[c];			\
			hashes##N[c] = (BUN##N##type) start;		\
			start++;					\
		}							\
	} while (0)

static void
BATparthash(BAT *b, Hash *h, int piece)
{
	BATiter bi = bat_iterator(b);
	BUN (*hashf)(const void *) = BATatoms[b->ttype].atomHash;
	BUN c;
	BUN start = piece * h->chunk;
	BUN end;
	BUN mask = h->mask;

	assert(piece < h->pieces);
	assert(piece >= 0);
	/* last piece is rest of BAT */
	if (piece == h->pieces - 1)
		end = BUNlast(b);
	else
		end = start + h->chunk;
	if (start < BUNfirst(b))
		start = BUNfirst(b);

	switch (h->width) {
	case BUN2:
		if (hashf == BATatoms[TYPE_bte].atomHash)
			parthashloop(bte, 2);
		else if (hashf == BATatoms[TYPE_sht].atomHash)
			parthashloop(sht, 2);
		else if (hashf == BATatoms[TYPE_int].atomHash)
			parthashloop(int, 2);
		else if (hashf == BATatoms[TYPE_lng].atomHash)
			parthashloop(lng, 2);
#ifdef HAVE_HGE
		else if (hashf == BATatoms[TYPE_hge].atomHash)
			parthashloop(hge, 2);
#endif
		else
			parthashanyloop(2);
		break;
	case BUN4:
		if (hashf == BATatoms[TYPE_bte].atomHash)
			parthashloop(bte, 4);
		else if (hashf == BATatoms[TYPE_sht].atomHash)
			parthashloop(sht, 4);
		else if (hashf == BATatoms[TYPE_int].atomHash)
			parthashloop(int, 4);
		else if (hashf == BATatoms[TYPE_lng].atomHash)
			parthashloop(lng, 4);
#ifdef HAVE_HGE
		else if (hashf == BATatoms[TYPE_hge].atomHash)
			parthashloop(hge, 4);
#endif
		else
			parthashanyloop(4);
		break;
#ifdef BUN8
	case BUN8:
		if (hashf == BATatoms[TYPE_bte].atomHash)
			parthashloop(bte, 8);
		else if (hashf == BATatoms[TYPE_sht].atomHash)
			parthashloop(sht, 8);
		else if (hashf == BATatoms[TYPE_int].atomHash)
			parthashloop(int, 8);
		else if (hashf == BATatoms[TYPE_lng].atomHash)
			parthashloop(lng, 8);
#ifdef HAVE_HGE
		else if (hashf == BATatoms[TYPE_hge].atomHash)
			parthashloop(hge, 8);
#endif
		else
			parthashanyloop(8);
		break;

#endif
	}
}

#ifndef DISABLE_HASHSYNC
static void
BAThashsync(void *arg)
{
	Heap *hp = arg;
	int fd;
	lng t0;

	t0 = GDKusec();
	if (HEAPsave(hp, hp->filename, NULL) != GDK_SUCCEED)
		return;
	if ((fd = GDKfdlocate(hp->farmid, hp->filename, "rb+", NULL)) < 0)
		return;
	((size_t *) hp->base)[0] |= 1 << 24;
	if (write(fd, hp->base, SIZEOF_SIZE_T) < 0)
		perror("write hash");
	if (!(GDKdebug & FORCEMITOMASK)) {
#if defined(NATIVE_WIN32)
		_commit(fd);
#elif defined(HAVE_FDATASYNC)
		fdatasync(fd);
#elif defined(HAVE_FSYNC)
		fsync(fd);
#endif
	}
	close(fd);
	ALGODEBUG fprintf(stderr, "#BAThash: persisting hash %s (" LLFMT " usec)\n", hp->filename, GDKusec() - t0);
}
#endif

static void
HASHcollisions(BAT *b, Hash *h)
{
	BUN entries = 0;
	BUN max = 0;
	BUN cnt;
	BUN total = 0;
	BUN nil = h->nil;
	int pcs;
	BUN i;
	BUN p;

	for (pcs = 0; pcs < h->pieces; pcs++) {
		for (i = 0; i <= h->mask; i++) {
			if ((p = HASHget(h, pcs, i)) != nil) {
				entries++;
				for (cnt = 0; p != nil; p = HASHgetlink(h, p))
					cnt++;
				if (cnt > max)
					max = cnt;
				total += cnt;
			}
		}
	}
	fprintf(stderr, "#HASH stats %s#" BUNFMT ", pieces: %d, "
		"chunksize: " BUNFMT ", mask: " BUNFMT ", entries: " BUNFMT ", "
		"max chain: " BUNFMT ", average chain: %2.6f\n",
		BATgetId(b), BATcount(b), h->pieces, h->chunk, h->mask,
		entries, max, entries == 0 ? 0.0 : (double) total / entries);
}

gdk_return
BAThash(BAT *b)
{
	if (BATtvoid(b)) {
		ALGODEBUG fprintf(stderr, "#BAThash: not creating hash on dense bat\n");
		return GDK_FAIL;
	}

	MT_lock_set(&GDKhashLock(abs(b->batCacheid)), "BAThash");
	if (b->T->hash == NULL) {
		BUN cnt = BATcount(b);
		int pieces = 1;
		int i;
		Hash *h;
		Heap *hp;
		const char *nme = BBP_physical(b->batCacheid);
		const char *ext = b->batCacheid > 0 ? "thash" : "hash";
		lng t0;

		if ((hp = GDKzalloc(sizeof(*hp))) == NULL ||
		    (hp->farmid = BBPselectfarm(b->batRole, b->ttype, hashheap)) < 0 ||
		    (hp->filename = GDKmalloc(strlen(nme) + strlen(ext) + 2)) == NULL) {
			GDKfree(hp);
			goto bailout;
		}
		sprintf(hp->filename, "%s.%s", nme, ext);
		if (cnt >= HASHMINSIZE * HASHMAXCHUNKS) {
			pieces = HASHMAXCHUNKS;
		} else {
			pieces = 1;
			while (cnt > 2 * (BUN) HASHMINSIZE * pieces) {
				pieces <<= 1;
			}
			assert(pieces < HASHMAXCHUNKS);
		}
		if ((h = HASHnew(hp, b->ttype, pieces, BATcapacity(b), cnt)) == NULL) {
			GDKfree(hp->filename);
			GDKfree(hp);
			goto bailout;
		}
		t0 = GDKusec();
		for (i = 0; i < pieces; i++)
			BATparthash(b, h, i);
		ALGODEBUG {
			fprintf(stderr, "#BAThash(%s#" BUNFMT "[%s]): hash construction " LLFMT " usec\n", BATgetId(b), BATcount(b), ATOMname(b->ttype), GDKusec() - t0);
			HASHcollisions(b, h);
		}
#if 0
		/* check that the hash table is correct */
		{
			BATiter bi = bat_iterator(b);
			BUN p, q;
			BATloop(b, p, q) {
				const void *v = BUNtail(bi, p);
				BUN prb = HASHprobe(h, v);
				BUN hb;
				BUN phb = BUN_NONE;
				int pcs;
				int seen = 0;

				HASHloop(bi, h, prb, v, hb, pcs) {
					seen += hb == p;
					assert(hb < phb);
					phb = hb;
				}
				assert(seen == 1);
			}
		}
#endif
		b->T->hash = h;
		/* unlock before potentially expensive sync */
		MT_lock_unset(&GDKhashLock(abs(b->batCacheid)), "BAThash");
#ifndef DISABLE_HASHSYNC
		if (BBP_status(b->batCacheid) & BBPEXISTING) {
			MT_Id tid;
			MT_create_thread(&tid, BAThashsync, hp, MT_THR_DETACHED);
		}
#endif
		return GDK_SUCCEED;
	}
  bailout:
	MT_lock_unset(&GDKhashLock(abs(b->batCacheid)), "BAThash");
	return b->T->hash ? GDK_SUCCEED : GDK_FAIL;
}

BUN
HASHlist(Hash *h, BUN prb)
{
	int pcs;
	BUN cnt = 0;
	BUN i;

	assert(prb <= h->mask);
	for (pcs = 0; pcs < h->pieces; pcs++)
		for (i = HASHget(h, pcs, prb); i != h->nil; i = HASHgetlink(h, i))
			cnt++;
	return cnt;
}
