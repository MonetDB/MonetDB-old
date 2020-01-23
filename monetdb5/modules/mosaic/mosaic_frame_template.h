MOSadvance_SIGNATURE(frame, TPE)
{
	MOSBlockHeaderTpe(frame, TPE)* parameters = (MOSBlockHeaderTpe(frame, TPE)*) (task)->blk;
	BUN cnt		= MOSgetCnt(task->blk);

	assert(cnt > 0);
	assert(MOSgetTag(task->blk) == MOSAIC_FRAME);

	task->start += (oid) cnt;

	char* blk = (char*)task->blk;
	blk += sizeof(MOSBlockHeaderTpe(frame, TPE));
	blk += BitVectorSize(cnt, parameters->bits);
	blk += GET_PADDING(task->blk, frame, TPE);

	task->blk = (MosaicBlk) blk;
}

static inline void CONCAT2(determineDeltaParameters, TPE)
(MOSBlockHeaderTpe(frame, TPE)* parameters, TPE* src, BUN limit) {
	TPE *val = src, max, min;
	bte bits = 1;
	unsigned int i;
	max = *val;
	min = *val;
	/*TODO: add additional loop to find best bit wise upper bound*/
	for(i = 0; i < limit; i++, val++){
		TPE current_max = max;
		TPE current_min = min;
		bool evaluate_bits = false;
		if (*val > current_max) {
			current_max = *val;
			evaluate_bits = true;
		}
		if (*val < current_min) {
			current_min = *val;
			evaluate_bits = true;
		}
		if (evaluate_bits) {
		 	DeltaTpe(TPE) width = GET_DELTA(TPE, current_max, current_min);
			bte current_bits = bits;
			while (width > ((DeltaTpe(TPE))(-1)) >> (sizeof(DeltaTpe(TPE)) * CHAR_BIT - current_bits) ) {/*keep track of number of BITS necessary to store difference*/
				current_bits++;
			}
			if ( (current_bits >= (int) ((sizeof(TPE) * CHAR_BIT) / 2))
				/*TODO: this extra condition should be removed once bitvector is extended to int64's*/
				|| (current_bits > (int) sizeof(BitVectorChunk) * CHAR_BIT) ) {
				/*If we can from here on not compress better then the half of the original data type, we give up. */
				break;
			}
			max = current_max;
			min = current_min;
			bits = current_bits;
		}
	}
	parameters->min = min;
	parameters->bits = bits;
	parameters->rec.cnt = i;
}

MOSestimate_SIGNATURE(frame, TPE)
{
	(void) previous;
	current->is_applicable = true;
	current->compression_strategy.tag = MOSAIC_FRAME;
	TPE *src = getSrc(TPE, task);
	BUN limit = task->stop - task->start > MOSAICMAXCNT? MOSAICMAXCNT: task->stop - task->start;
	MOSBlockHeaderTpe(frame, TPE) parameters;
	CONCAT2(determineDeltaParameters, TPE)(&parameters, src, limit);
	assert(parameters.rec.cnt > 0);/*Should always compress.*/
	current->uncompressed_size += (BUN) (parameters.rec.cnt * sizeof(TPE));
	current->compressed_size += 2 * sizeof(MOSBlockHeaderTpe(frame, TPE)) + wordaligned((parameters.rec.cnt * parameters.bits) / CHAR_BIT, lng);
	current->compression_strategy.cnt = (unsigned int) parameters.rec.cnt;

	if (parameters.rec.cnt > *current->max_compression_length ) {
		*current->max_compression_length = parameters.rec.cnt;
	}

	return MAL_SUCCEED;
}

MOSpostEstimate_SIGNATURE(frame, TPE)
{
	(void) task;
}

// rather expensive simple value non-compressed store
MOScompress_SIGNATURE(frame, TPE)
{
	ALIGN_BLOCK_HEADER(task,  frame, TPE);

	MosaicBlk blk = task->blk;
	MOSsetTag(blk,MOSAIC_FRAME);
	MOSsetCnt(blk, 0);
	TPE *src = getSrc(TPE, task);
	TPE delta;
	BUN i = 0;
	BUN limit = estimate->cnt;
	MOSBlockHeaderTpe(frame, TPE)* parameters = (MOSBlockHeaderTpe(frame, TPE)*) (task)->blk;
    CONCAT2(determineDeltaParameters, TPE)(parameters, src, limit);
	BitVector base = MOScodevectorFrame(task, TPE);
	task->dst = (char*) base;
	for(i = 0; i < MOSgetCnt(task->blk); i++, src++) {
		/*TODO: assert that delta's actually does not cause an overflow. */
		delta = *src - parameters->min;
		setBitVector(base, i, parameters->bits, (BitVectorChunk) /*TODO: fix this once we have increased capacity of bitvector*/ delta);
	}
	task->dst += BitVectorSize(i, parameters->bits);
}

MOSdecompress_SIGNATURE(frame, TPE)
{
	MOSBlockHeaderTpe(frame, TPE)* parameters = (MOSBlockHeaderTpe(frame, TPE)*) (task)->blk;
	BUN lim = MOSgetCnt(task->blk);
    TPE min = parameters->min;
	BitVector base = (BitVector) MOScodevectorFrame(task, TPE);
	BUN i;
	for(i = 0; i < lim; i++){
		TPE delta = getBitVector(base, i, parameters->bits);
		/*TODO: assert that delta's actually does not cause an overflow. */
		TPE val = min + delta;
		((TPE*)task->src)[i] = val;
	}
	task->src += i * sizeof(TPE);
}
