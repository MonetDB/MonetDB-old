#define OverShift ((sizeof(IPTpe(TPE)) - sizeof(PrefixTpe(TPE))) * CHAR_BIT)
#define getSuffixMask(SUFFIX_BITS) ((PrefixTpe(TPE)) (~(~((IPTpe(TPE)) (0)) << (SUFFIX_BITS))))
#define getPrefixMask(PREFIX_BITS) ((PrefixTpe(TPE)) ( (~(~((IPTpe(TPE)) (0)) >> (PREFIX_BITS))) >> OverShift))

MOSadvance_SIGNATURE(prefix, TPE)
{
	MOSBlockHeaderTpe(prefix, TPE)* parameters = (MOSBlockHeaderTpe(prefix, TPE)*) (task)->blk;
	BUN cnt = MOSgetCnt(task->blk);

	assert(cnt > 0);
	assert(MOSgetTag(task->blk) == MOSAIC_PREFIX);

	task->start += (oid) cnt;

	char* blk = (char*)task->blk;
	blk += sizeof(MOSBlockHeaderTpe(prefix, TPE));
	blk += BitVectorSize(cnt, parameters->suffix_bits);
	blk += GET_PADDING(task->blk, prefix, TPE);

	task->blk = (MosaicBlk) blk;
}

static inline void CONCAT2(determineDeltaParameters, TPE)
(MOSBlockHeaderTpe(prefix, TPE)* parameters, PrefixTpe(TPE)* src, BUN limit) {
	PrefixTpe(TPE) *val = (PrefixTpe(TPE)*) (src);
	const int type_size_in_bits = sizeof(PrefixTpe(TPE))  * CHAR_BIT;
	bte suffix_bits = 1;
	bte prefix_bits = type_size_in_bits - suffix_bits;
	PrefixTpe(TPE) prefix_mask = getPrefixMask(prefix_bits);
	PrefixTpe(TPE) prefix = *val & prefix_mask;
	/*TODO: add additional loop to find best bit wise upper bound*/
	BUN i;
	for(i = 0; i < (limit); i++, val++){
		bte current_prefix_bits = prefix_bits;
		bte current_suffix_bits = suffix_bits;
		PrefixTpe(TPE) current_prefix = prefix;
		PrefixTpe(TPE) current_prefix_mask =  prefix_mask;

		while ((current_prefix) != (current_prefix_mask & (*val))) {
			current_prefix_bits--;
			current_prefix_mask = getPrefixMask(current_prefix_bits);
			current_prefix = prefix & current_prefix_mask;
			current_suffix_bits++;
		}

		if (current_suffix_bits >= (int) ((sizeof(PrefixTpe(TPE)) * CHAR_BIT) / 2)) {
			/*If we can not compress better then the half of the original data type, we give up. */
			break;
		}
		if ((current_suffix_bits > (int) sizeof(BitVectorChunk) * CHAR_BIT)) {
			/*TODO: this extra condition should be removed once bitvector is extended to int64's*/
			break;
		}

		prefix = current_prefix;
		prefix_mask = current_prefix_mask;
		prefix_bits = current_prefix_bits;
		suffix_bits = current_suffix_bits;

		assert (suffix_bits + prefix_bits == type_size_in_bits);
		assert( (prefix | (getSuffixMask(suffix_bits) & (*val))) == *val);
	}

	parameters->rec.cnt = (unsigned int) i;
	parameters->suffix_bits = suffix_bits;
	parameters->prefix = prefix;
}

MOSestimate_SIGNATURE(prefix, TPE)
{
	(void) previous;
	current->is_applicable = true;
	current->compression_strategy.tag = MOSAIC_PREFIX;
	PrefixTpe(TPE) *src = ((PrefixTpe(TPE)*) task->src) + task->start;
	BUN limit = task->stop - task->start > MOSAICMAXCNT? MOSAICMAXCNT: task->stop - task->start;
	MOSBlockHeaderTpe(prefix, TPE) parameters;
	CONCAT2(determineDeltaParameters, TPE)(&parameters, src, limit);
	assert(parameters.rec.cnt > 0);/*Should always compress.*/

	BUN store;
	int bits;
	int i = parameters.rec.cnt;
	bits = i * parameters.suffix_bits;
	store = 2 * sizeof(MOSBlockHeaderTpe(prefix, TPE));
	store += wordaligned(bits/CHAR_BIT + ((bits % CHAR_BIT) > 0), lng);
	assert(i > 0);/*Should always compress.*/
	current->is_applicable = true;

	current->uncompressed_size += (BUN) (i * sizeof(TPE));
	current->compressed_size += store;
	current->compression_strategy.cnt = (unsigned int) parameters.rec.cnt;

	if (parameters.rec.cnt > *current->max_compression_length ) {
		*current->max_compression_length = parameters.rec.cnt;
	}

	return MAL_SUCCEED;
}

#define MOSpostEstimate_DEF(TPE)
MOSpostEstimate_SIGNATURE(prefix, TPE)
{
	(void) task;
}

// rather expensive simple value non-compressed store
#define MOScompress_DEF(TPE)
MOScompress_SIGNATURE(prefix, TPE)
{
	ALIGN_BLOCK_HEADER(task,  prefix, TPE);

	MosaicBlk blk = task->blk;
	MOSsetTag(blk,MOSAIC_PREFIX);
	MOSsetCnt(blk, 0);
	PrefixTpe(TPE)* src = (PrefixTpe(TPE)*) getSrc(TPE, task);
	BUN i = 0;
	BUN limit = estimate->cnt;
	MOSBlockHeaderTpe(prefix, TPE)* parameters = (MOSBlockHeaderTpe(prefix, TPE)*) (task)->blk;
	CONCAT2(determineDeltaParameters, TPE)(parameters, src, limit);
	BitVector base = MOScodevectorPrefix(task, TPE);
	task->dst = (char*) base;
	PrefixTpe(TPE) suffix_mask = getSuffixMask(parameters->suffix_bits);
	for(i = 0; i < MOSgetCnt(task->blk); i++, src++) {
		/*TODO: assert that prefix's actually does not cause an overflow. */
		PrefixTpe(TPE) suffix = *src & suffix_mask;
		setBitVector(base, i, parameters->suffix_bits, (BitVectorChunk) /*TODO: fix this once we have increased capacity of bitvector*/ suffix);
	}
	task->dst += BitVectorSize(i, parameters->suffix_bits);
}

#define MOSdecompress_DEF(TPE) 
MOSdecompress_SIGNATURE(prefix, TPE)
{
	MOSBlockHeaderTpe(prefix, TPE)* parameters = (MOSBlockHeaderTpe(prefix, TPE)*) (task)->blk;
	BUN lim = MOSgetCnt(task->blk);
    PrefixTpe(TPE) prefix = parameters->prefix;
	BitVector base = (BitVector) MOScodevectorPrefix(task, TPE);
	BUN i;
	for(i = 0; i < lim; i++){
		PrefixTpe(TPE) suffix = getBitVector(base, i, parameters->suffix_bits);
		/*TODO: assert that suffix's actually does not cause an overflow. */
		PrefixTpe(TPE) val = prefix | suffix;
		((PrefixTpe(TPE)*)task->src)[i] = val;
	}
	task->src += i * sizeof(TPE);
}

#undef OverShift
#undef getSuffixMask
#undef getPrefixMask
