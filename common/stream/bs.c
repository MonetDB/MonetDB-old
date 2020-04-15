/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2020 MonetDB B.V.
 */

#include "monetdb_config.h"
#include "stream.h"
#include "stream_internal.h"


/* ------------------------------------------------------------------ */

/* A buffered stream consists of a sequence of blocks.  Each block
 * consists of a count followed by the data in the block.  A flush is
 * indicated by an empty block (i.e. just a count of 0).
 */

static bs *
bs_create(stream *s)
{
	/* should be a binary stream */
	bs *ns;

	if ((ns = malloc(sizeof(*ns))) == NULL)
		return NULL;
	*ns = (bs) {
		.s = s,
	};
	return ns;
}

/* Collect data until the internal buffer is filled, then write the
 * filled buffer to the underlying stream.
 * Struct field usage:
 * s - the underlying stream;
 * buf - the buffer in which data is collected;
 * nr - how much of buf is already filled (if nr == sizeof(buf) the
 *      data is written to the underlying stream, so upon entry nr <
 *      sizeof(buf));
 * itotal - unused.
 */
ssize_t
bs_write(stream *restrict ss, const void *restrict buf, size_t elmsize, size_t cnt)
{
	bs *s;
	size_t todo = cnt * elmsize;
	uint16_t blksize;

	s = (bs *) ss->stream_data.p;
	if (s == NULL)
		return -1;
	assert(!ss->readonly);
	assert(s->nr < sizeof(s->buf));
	while (todo > 0) {
		size_t n = sizeof(s->buf) - s->nr;

		if (todo < n)
			n = todo;
		memcpy(s->buf + s->nr, buf, n);
		s->nr += (unsigned) n;
		todo -= n;
		buf = ((const char *) buf + n);
		if (s->nr == sizeof(s->buf)) {
			/* block is full, write it to the stream */
#ifdef BSTREAM_DEBUG
			{
				unsigned i;

				fprintf(stderr, "W %s %u \"", ss->name, s->nr);
				for (i = 0; i < s->nr; i++)
					if (' ' <= s->buf[i] && s->buf[i] < 127)
						putc(s->buf[i], stderr);
					else
						fprintf(stderr, "\\%03o", (unsigned char) s->buf[i]);
				fprintf(stderr, "\"\n");
			}
#endif
			/* since the block is at max BLOCK (8K) - 2 size we can
			 * store it in a two byte integer */
			blksize = (uint16_t) s->nr;
			s->bytes += s->nr;
			/* the last bit tells whether a flush is in
			 * there, it's not at this moment, so shift it
			 * to the left */
			blksize <<= 1;
			if (!mnstr_writeSht(s->s, (int16_t) blksize) ||
			    s->s->write(s->s, s->buf, 1, s->nr) != (ssize_t) s->nr) {
				ss->errnr = MNSTR_WRITE_ERROR;
				s->nr = 0; /* data is lost due to error */
				return -1;
			}
			s->blks++;
			s->nr = 0;
		}
	}
	return (ssize_t) cnt;
}

/* If the internal buffer is partially filled, write it to the
 * underlying stream.  Then in any case write an empty buffer to the
 * underlying stream to indicate to the receiver that the data was
 * flushed.
 */
static int
bs_flush(stream *ss)
{
	uint16_t blksize;
	bs *s;

	s = (bs *) ss->stream_data.p;
	if (s == NULL)
		return -1;
	assert(!ss->readonly);
	assert(s->nr < sizeof(s->buf));
	if (!ss->readonly) {
		/* flush the rest of buffer (if s->nr > 0), then set the
		 * last bit to 1 to to indicate user-instigated flush */
#ifdef BSTREAM_DEBUG
		if (s->nr > 0) {
			unsigned i;

			fprintf(stderr, "W %s %u \"", ss->name, s->nr);
			for (i = 0; i < s->nr; i++)
				if (' ' <= s->buf[i] && s->buf[i] < 127)
					putc(s->buf[i], stderr);
				else
					fprintf(stderr, "\\%03o", (unsigned char) s->buf[i]);
			fprintf(stderr, "\"\n");
			fprintf(stderr, "W %s 0\n", ss->name);
		}
#endif
		blksize = (uint16_t) (s->nr << 1);
		s->bytes += s->nr;
		/* indicate that this is the last buffer of a block by
		 * setting the low-order bit */
		blksize |= 1;
		/* allways flush (even empty blocks) needed for the protocol) */
		if ((!mnstr_writeSht(s->s, (int16_t) blksize) ||
		     (s->nr > 0 &&
		      s->s->write(s->s, s->buf, 1, s->nr) != (ssize_t) s->nr))) {
			ss->errnr = MNSTR_WRITE_ERROR;
			s->nr = 0; /* data is lost due to error */
			return -1;
		}
		s->blks++;
		s->nr = 0;
	}
	return 0;
}

/* Read buffered data and return the number of items read.  At the
 * flush boundary we will return 0 to indicate the end of a block.
 *
 * Structure field usage:
 * s - the underlying stream;
 * buf - not used;
 * itotal - the amount of data in the current block that hasn't yet
 *          been read;
 * nr - indicates whether the flush marker has to be returned.
 */
ssize_t
bs_read(stream *restrict ss, void *restrict buf, size_t elmsize, size_t cnt)
{
	bs *s;
	size_t todo = cnt * elmsize;
	size_t n;

	s = (bs *) ss->stream_data.p;
	if (s == NULL)
		return -1;
	assert(ss->readonly);
	assert(s->nr <= 1);

	if (s->itotal == 0) {
		int16_t blksize = 0;

		if (s->nr) {
			/* We read the closing block but hadn't
			 * returned that yet. Return it now, and note
			 * that we did by setting s->nr to 0. */
			assert(s->nr == 1);
			s->nr = 0;
			return 0;
		}

		assert(s->nr == 0);

		/* There is nothing more to read in the current block,
		 * so read the count for the next block */
		switch (mnstr_readSht(s->s, &blksize)) {
		case -1:
			ss->errnr = s->s->errnr;
			return -1;
		case 0:
			return 0;
		case 1:
			break;
		}
		if ((uint16_t) blksize > (BLOCK << 1 | 1)) {
			ss->errnr = MNSTR_READ_ERROR;
			return -1;
		}
#ifdef BSTREAM_DEBUG
		fprintf(stderr, "RC size: %u, final: %s\n", (uint16_t) blksize >> 1, (uint16_t) blksize & 1 ? "true" : "false");
		fprintf(stderr, "RC %s %u\n", ss->name, (uint16_t) blksize);
#endif
		s->itotal = (uint16_t) blksize >> 1;	/* amount readable */
		/* store whether this was the last block or not */
		s->nr = (uint16_t) blksize & 1;
		s->bytes += s->itotal;
		s->blks++;
	}

	/* Fill the caller's buffer. */
	cnt = 0;		/* count how much we put into the buffer */
	while (todo > 0) {
		/* there is more data waiting in the current block, so
		 * read it */
		n = todo < s->itotal ? todo : s->itotal;
		while (n > 0) {
			ssize_t m = s->s->read(s->s, buf, 1, n);

			if (m <= 0) {
				ss->errnr = s->s->errnr;
				return -1;
			}
#ifdef BSTREAM_DEBUG
			{
				ssize_t i;

				fprintf(stderr, "RD %s %zd \"", ss->name, m);
				for (i = 0; i < m; i++)
					if (' ' <= ((char *) buf)[i] &&
					    ((char *) buf)[i] < 127)
						putc(((char *) buf)[i], stderr);
					else
						fprintf(stderr, "\\%03o", ((unsigned char *) buf)[i]);
				fprintf(stderr, "\"\n");
			}
#endif
			buf = (void *) ((char *) buf + m);
			cnt += (size_t) m;
			n -= (size_t) m;
			s->itotal -= (unsigned) m;
			todo -= (size_t) m;
		}

		if (s->itotal == 0) {
			int16_t blksize = 0;

			/* The current block has been completely read,
			 * so read the count for the next block, only
			 * if the previous was not the last one */
			if (s->nr)
				break;
			switch (mnstr_readSht(s->s, &blksize)) {
			case -1:
				ss->errnr = s->s->errnr;
				return -1;
			case 0:
				return 0;
			case 1:
				break;
			}
			if ((uint16_t) blksize > (BLOCK << 1 | 1)) {
				ss->errnr = MNSTR_READ_ERROR;
				return -1;
			}
#ifdef BSTREAM_DEBUG
			fprintf(stderr, "RC size: %d, final: %s\n", (uint16_t) blksize >> 1, (uint16_t) blksize & 1 ? "true" : "false");
			fprintf(stderr, "RC %s %d\n", ss->name, s->nr);
			fprintf(stderr, "RC %s %d\n", ss->name, blksize);
#endif
			s->itotal = (uint16_t) blksize >> 1;	/* amount readable */
			/* store whether this was the last block or not */
			s->nr = (uint16_t) blksize & 1;
			s->bytes += s->itotal;
			s->blks++;
		}
	}
	/* if we got an empty block with the end-of-sequence marker
	 * set (low-order bit) we must only return an empty read once,
	 * so we must squash the flag that we still have to return an
	 * empty read */
	if (todo > 0 && cnt == 0)
		s->nr = 0;
	return (ssize_t) (elmsize > 0 ? cnt / elmsize : 0);
}

static void
bs_update_timeout(stream *ss)
{
	bs *s;

	if ((s = ss->stream_data.p) != NULL && s->s) {
		s->s->timeout = ss->timeout;
		s->s->timeout_func = ss->timeout_func;
		if (s->s->update_timeout)
			s->s->update_timeout(s->s);
	}
}

static int
bs_isalive(const stream *ss)
{
	struct bs *s;

	if ((s = ss->stream_data.p) != NULL && s->s) {
		if (s->s->isalive)
			return s->s->isalive(s->s);
		return 1;
	}
	return 0;
}

static void
bs_close(stream *ss)
{
	bs *s;

	s = (bs *) ss->stream_data.p;
	assert(s);
	if (s == NULL)
		return;
	if (!ss->readonly && s->nr > 0)
		bs_flush(ss);
	if (s->s)
		s->s->close(s->s);
}

void
bs_destroy(stream *ss)
{
	bs *s;

	s = (bs *) ss->stream_data.p;
	assert(s);
	if (s) {
		if (s->s)
			s->s->destroy(s->s);
		free(s);
	}
	destroy_stream(ss);
}

void
bs_clrerr(stream *s)
{
	if (s->stream_data.p)
		mnstr_clearerr(((bs *) s->stream_data.p)->s);
}

stream *
bs_stream(stream *s)
{
	assert(isa_block_stream(s));
	return ((bs *) s->stream_data.p)->s;
}

stream *
block_stream(stream *s)
{
	stream *ns;
	bs *b;

	if (s == NULL)
		return NULL;
#ifdef STREAM_DEBUG
	fprintf(stderr, "block_stream %s\n", s->name ? s->name : "<unnamed>");
#endif
	if ((ns = create_stream(s->name)) == NULL)
		return NULL;
	if ((b = bs_create(s)) == NULL) {
		destroy_stream(ns);
		return NULL;
	}
	/* blocksizes have a fixed little endian byteorder */
#ifdef WORDS_BIGENDIAN
	s->swapbytes = true;
#endif
	ns->binary = s->binary;
	ns->readonly = s->readonly;
	ns->close = bs_close;
	ns->clrerr = bs_clrerr;
	ns->destroy = bs_destroy;
	ns->flush = bs_flush;
	ns->read = bs_read;
	ns->write = bs_write;
	ns->update_timeout = bs_update_timeout;
	ns->isalive = bs_isalive;
	ns->stream_data.p = (void *) b;

	return ns;
}

