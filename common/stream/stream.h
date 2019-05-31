/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef _STREAM_H_
#define _STREAM_H_

/*
 * File: stream.h
 * Auteur: Niels J. Nes
 * Date: 09-01-2001
 *
 * Version 0.1: start
 *
 * This is the general interface to input/output. Each stream will
 * contains some stream info (for now only byteorder). This is
 * required for proper conversion on different byte order platforms.
 */

#include "monetdb_config.h"
#ifndef NATIVE_WIN32
# include <unistd.h>
#endif
#include <ctype.h>
#include <limits.h>

/* avoid using "#ifdef WIN32" so that this file does not need our config.h */
#if defined(_MSC_VER) || defined(__CYGWIN__) || defined(__MINGW32__)
# ifndef LIBSTREAM
#  define stream_export extern __declspec(dllimport)
# else
#  define stream_export extern __declspec(dllexport)
# endif
#else
# define stream_export extern
#endif

#define EOT 4

/* fwf gets turned into a csv with these parameters */
#define STREAM_FWF_FIELD_SEP '|'
#define STREAM_FWF_ESCAPE '\\'
#define STREAM_FWF_RECORD_SEP '\n'
#define STREAM_FWF_FILLER ' '

#ifdef _MSC_VER
/* use intrinsic functions on Windows */
#define short_int_SWAP(s)	((int16_t) _byteswap_ushort((uint16_t) (s)))
/* on Windows, long is the same size as int */
#define normal_int_SWAP(i)	((int) _byteswap_ulong((unsigned long) (i)))
#define long_int_SWAP(l)	((int64_t) _byteswap_uint64((unsigned __int64) (l)))
#else
#define short_int_SWAP(s)				\
	((int16_t) (((0x00ff & (uint16_t) (s)) << 8) |	\
		  ((0xff00 & (uint16_t) (s)) >> 8)))

#define normal_int_SWAP(i)						\
	((int) (((((unsigned) 0xff <<  0) & (unsigned) (i)) << 24) |	\
		((((unsigned) 0xff <<  8) & (unsigned) (i)) <<  8) |	\
		((((unsigned) 0xff << 16) & (unsigned) (i)) >>  8) |	\
		((((unsigned) 0xff << 24) & (unsigned) (i)) >> 24)))

#define long_int_SWAP(l)						\
	((int64_t) (((((uint64_t) 0xff <<  0) & (uint64_t) (l)) << 56) | \
		((((uint64_t) 0xff <<  8) & (uint64_t) (l)) << 40) |	\
		((((uint64_t) 0xff << 16) & (uint64_t) (l)) << 24) |	\
		((((uint64_t) 0xff << 24) & (uint64_t) (l)) <<  8) |	\
		((((uint64_t) 0xff << 32) & (uint64_t) (l)) >>  8) |	\
		((((uint64_t) 0xff << 40) & (uint64_t) (l)) >> 24) |	\
		((((uint64_t) 0xff << 48) & (uint64_t) (l)) >> 40) |	\
		((((uint64_t) 0xff << 56) & (uint64_t) (l)) >> 56)))
#endif

#ifdef HAVE_HGE
#define huge_int_SWAP(h)					\
	((hge) (((((uhge) 0xff <<   0) & (uhge) (h)) << 120) |	\
		((((uhge) 0xff <<   8) & (uhge) (h)) << 104) |	\
		((((uhge) 0xff <<  16) & (uhge) (h)) <<  88) |	\
		((((uhge) 0xff <<  24) & (uhge) (h)) <<  72) |	\
		((((uhge) 0xff <<  32) & (uhge) (h)) <<  56) |	\
		((((uhge) 0xff <<  40) & (uhge) (h)) <<  40) |	\
		((((uhge) 0xff <<  48) & (uhge) (h)) <<  24) |	\
		((((uhge) 0xff <<  56) & (uhge) (h)) <<   8) |	\
		((((uhge) 0xff <<  64) & (uhge) (h)) >>   8) |	\
		((((uhge) 0xff <<  72) & (uhge) (h)) >>  24) |	\
		((((uhge) 0xff <<  80) & (uhge) (h)) >>  40) |	\
		((((uhge) 0xff <<  88) & (uhge) (h)) >>  56) |	\
		((((uhge) 0xff <<  96) & (uhge) (h)) >>  72) |	\
		((((uhge) 0xff << 104) & (uhge) (h)) >>  88) |	\
		((((uhge) 0xff << 112) & (uhge) (h)) >> 104) |	\
		((((uhge) 0xff << 120) & (uhge) (h)) >> 120)))
#endif

typedef struct stream stream;

/* some os specific initialization */
stream_export int mnstr_init(void);

/* all mnstr_readX/mnstr_writeX return
 *  0 on error
 * !0 on success
 */
stream_export int mnstr_readBte(stream *restrict s, int8_t *restrict val);
stream_export int mnstr_readChr(stream *restrict s, char *restrict val);
stream_export int mnstr_writeChr(stream *s, char val);

stream_export int mnstr_writeBte(stream *s, int8_t val);
stream_export int mnstr_readSht(stream *restrict s, int16_t *restrict val);
stream_export int mnstr_writeSht(stream *s, int16_t val);
stream_export int mnstr_readInt(stream *restrict s, int *restrict val);
stream_export int mnstr_writeInt(stream *s, int val);
stream_export int mnstr_readLng(stream *restrict s, int64_t *restrict val);
stream_export int mnstr_writeLng(stream *s, int64_t val);

stream_export int mnstr_writeFlt(stream *s, float val);
stream_export int mnstr_writeDbl(stream *s, double val);

#ifdef HAVE_HGE
stream_export int mnstr_readHge(stream *restrict s, hge *restrict val);
stream_export int mnstr_writeHge(stream *s, hge val);
#endif

stream_export int mnstr_readBteArray(stream *restrict s, int8_t *restrict val, size_t cnt);
stream_export int mnstr_writeBteArray(stream *restrict s, const int8_t *restrict val, size_t cnt);
stream_export int mnstr_writeStr(stream *restrict s, const char *restrict val);
stream_export int mnstr_readStr(stream *restrict s, char *restrict val);

stream_export int mnstr_readShtArray(stream *restrict s, int16_t *restrict val, size_t cnt);
stream_export int mnstr_writeShtArray(stream *restrict s, const int16_t *restrict val, size_t cnt);
stream_export int mnstr_readIntArray(stream *restrict s, int *restrict val, size_t cnt);
stream_export int mnstr_writeIntArray(stream *restrict s, const int *restrict val, size_t cnt);
stream_export int mnstr_readLngArray(stream *restrict s, int64_t *restrict val, size_t cnt);
stream_export int mnstr_writeLngArray(stream *restrict s, const int64_t *restrict val, size_t cnt);
#ifdef HAVE_HGE
stream_export int mnstr_readHgeArray(stream *restrict s, hge *restrict val, size_t cnt);
stream_export int mnstr_writeHgeArray(stream *restrict s, const hge *restrict val, size_t cnt);
#endif
stream_export int mnstr_printf(stream *restrict s, _In_z_ _Printf_format_string_ const char *restrict format, ...)
	__attribute__((__format__(__printf__, 2, 3)));
stream_export ssize_t mnstr_read(stream *restrict s, void *restrict buf, size_t elmsize, size_t cnt);
stream_export ssize_t mnstr_readline(stream *restrict s, void *restrict buf, size_t maxcnt);
stream_export ssize_t mnstr_write(stream *restrict s, const void *restrict buf, size_t elmsize, size_t cnt);
stream_export void mnstr_close(stream *s);
stream_export void mnstr_destroy(stream *s);
stream_export char *mnstr_error(stream *s);
stream_export int mnstr_flush(stream *s);
stream_export int mnstr_fsync(stream *s);
stream_export int mnstr_fgetpos(stream *restrict s, fpos_t *restrict p);
stream_export int mnstr_fsetpos(stream *restrict s, fpos_t *restrict p);
stream_export char *mnstr_name(stream *s);
stream_export int mnstr_errnr(stream *s);
stream_export void mnstr_clearerr(stream *s);
stream_export bool mnstr_isbinary(stream *s);
stream_export bool mnstr_get_swapbytes(stream *s);
stream_export void mnstr_set_bigendian(stream *s, bool bigendian);
stream_export void mnstr_settimeout(stream *s, unsigned int ms, bool (*func)(void));
stream_export int mnstr_isalive(stream *s);

stream_export stream *open_rstream(const char *filename);
stream_export stream *open_wstream(const char *filename);

/* open in ascii stream in read mode */
stream_export stream *open_rastream(const char *filename);

/* open in ascii stream in write mode*/
stream_export stream *open_wastream(const char *filename);

stream_export void close_stream(stream *s);

stream_export stream *open_urlstream(const char *url);

stream_export stream *file_rstream(FILE *restrict fp, const char *restrict name);
stream_export stream *file_wstream(FILE *restrict fp, const char *restrict name);
stream_export stream *file_rastream(FILE *restrict fp, const char *restrict name);
stream_export stream *file_wastream(FILE *restrict fp, const char *restrict name);

stream_export FILE *getFile(stream *s);
stream_export int getFileNo(stream *s);	/* fileno(getFile(s)) */
stream_export size_t getFileSize(stream *s);

stream_export stream *iconv_rstream(stream *restrict ss, const char *restrict charset, const char *restrict name);
stream_export stream *iconv_wstream(stream *restrict ss, const char *restrict charset, const char *restrict name);

typedef struct buffer {
	char *buf;
	size_t pos;
	size_t len;
} buffer;

stream_export void buffer_init(buffer *restrict b, char *restrict buf, size_t size);
stream_export buffer *buffer_create(size_t size);
stream_export char *buffer_get_buf(buffer *b);
stream_export void buffer_destroy(buffer *b);

stream_export stream *buffer_rastream(buffer *restrict b, const char *restrict name);
stream_export stream *buffer_wastream(buffer *restrict b, const char *restrict name);
stream_export buffer *mnstr_get_buffer(stream *s);

/* note, the size is fixed to 8K, you cannot simply change it to any
 * value */
#define BLOCK (8 * 1024 - 2)

/* Block stream is a stream which sends data in blocks of a known size
 * (BLOCK size or dynamically changed using CHANGE_BLOCK_SIZE msg).
 *
 * A block is written once more than BLOCK size data has been written
 * using the write commands or when the flush command is sent.
 *
 * All full blocks together with a single not full block form a major
 * block. Major blocks can be used to synchronize the communication.
 * Example server sends some reply, ie a major block consisting of
 * various minor blocks. The header of the major block can contain
 * special info which the client can interpret.
 *
 * Each read attempt tries to return the number of bytes. Once a lower
 * number of bytes can be read the end of the major block is
 * found. The next read will then start with a new major block.
 */
stream_export stream *block_stream(stream *s);
stream_export bool isa_block_stream(stream *s);
stream_export stream *bs_stream(stream *s);

typedef enum {
	PROTOCOL_AUTO = 0,
	PROTOCOL_9 = 1,
	PROTOCOL_10 = 2
} protocol_version;

typedef enum {
	COMPRESSION_NONE = 0,
	COMPRESSION_SNAPPY = 1,
	COMPRESSION_LZ4 = 2,
	COMPRESSION_AUTO = 255
} compression_method;

stream_export stream *block_stream2(stream *s, size_t bufsiz, compression_method comp);
stream_export int bs2_resizebuf(stream *ss, size_t bufsiz);
stream_export buffer bs2_buffer(stream *s);
stream_export void bs2_setpos(stream *ss, size_t pos);

/* read block of data including the end of block marker */
stream_export ssize_t mnstr_read_block(stream *restrict s, void *restrict buf, size_t elmsize, size_t cnt);

typedef struct bstream {
	stream *s;
	char *buf;
	size_t size;		/* size of buf */
	size_t pos;		/* the data cursor (ie read until pos) */
	size_t len;		/* len of the data (<= size) */
	size_t mode;		/* 0 line mode else size for block mode */
	bool eof;
} bstream;

stream_export bstream *bstream_create(stream *rs, size_t chunk_size);
stream_export void bstream_destroy(bstream *s);
stream_export ssize_t bstream_read(bstream *s, size_t size);
stream_export ssize_t bstream_next(bstream *s);

typedef enum mnstr_errors {
	MNSTR_NO__ERROR = 0,
	MNSTR_OPEN_ERROR,
	MNSTR_READ_ERROR,
	MNSTR_WRITE_ERROR,
	MNSTR_TIMEOUT
} mnstr_errors;

/* Callback stream is a read-only stream where the read function is
 * provided by the caller.  close and destroy are also provided.  The
 * private pointer is passed on to the callback functions when they
 * are invoked. */
stream_export stream *callback_stream(
	void *restrict priv,
	ssize_t (*read)(void *restrict priv, void *restrict buf, size_t elmsize, size_t cnt),
	void (*close)(void *priv),
	void (*destroy)(void *priv),
	const char *restrict name);

stream_export stream *stream_blackhole_create(void);

stream_export stream *stream_fwf_create(stream *restrict s, size_t num_fields, size_t *restrict widths, char filler);

#endif /*_STREAM_H_*/
