/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef STORE_SEQ_H
#define STORE_SEQ_H

#include "sql_catalog.h"

sql_extern void* sequences_init(void);
sql_extern void sequences_exit(void);

sql_extern int seq_next_value(sql_sequence *seq, lng *val);

/* for bulk next values, the API is split in 3 parts */

typedef struct seqbulk {
	void *internal_seq;
	sql_sequence *seq;
	BUN cnt;
	int save; 
} seqbulk;

sql_extern seqbulk *seqbulk_create(sql_sequence *seq, BUN cnt);
sql_extern int seqbulk_next_value(seqbulk *seq, lng *val);
sql_extern void seqbulk_destroy(seqbulk *seq);

sql_extern int seq_get_value(sql_sequence *seq, lng *val);
sql_extern int seq_restart(sql_sequence *seq, lng start);

#endif /* STORE_SEQ_H */
