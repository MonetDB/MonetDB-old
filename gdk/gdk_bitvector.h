
/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 2008-2015 MonetDB B.V.
 */

/*
 * The primitives to create, maintain, and use a bit mask.
 * Each bitvector is represented by Vector,Cnt,Width 
 * where Vector is a bit sequence with CNT number of chunks of size Width
 * The Vector is aligned at sizeof(lng) =  64 bits
 * Cnt is a BUN, values are represented as lng
 */
#ifndef _GDK_MASK_H_
#define _GDK_MASK_H_

typedef unsigned int BitVectorChunk;

typedef BitVectorChunk *BitVector;

gdk_export void initBitMasks(void);
gdk_export size_t getBitVectorSize(const BUN cnt, const int width);
gdk_export BitVector newBitVector(BUN cnt, int width);
gdk_export void setBitVector(BitVector vector, const BUN i, const bte bits, const BitVectorChunk value);
gdk_export void clrBitVector(BitVector vector, BUN i, const bte bits);
gdk_export int tstBitVector(BitVector vector, BUN i, const bte bits);
gdk_export int getBitVector(BitVector vector, BUN i, const bte bits);

#define BitVectorSize(CNT, BITS) wordaligned(((CNT) * (BITS) / CHAR_BIT) + ( ((CNT) * (BITS)) % CHAR_BIT != 0 ), BitVectorChunk)

#endif /* _GDK_MASK_H_ */
