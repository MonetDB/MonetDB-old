/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2019 MonetDB B.V.
 */

#ifndef LIST_H
#define LIST_H

#include "sql_mem.h"
#include "sql_hash.h"

typedef struct node {
	struct sql_hash_e e;
	struct node *next;
	void *data;
} node;

typedef void (*fdestroy) (void *);

typedef struct list {
	sql_allocator *sa;
	sql_hash *ht;
	MT_Lock ht_lock;	/* latch protecting ht */
	fdestroy destroy;
	node *h;
	node *t;
	int cnt;
	int expected_cnt;
} list;

typedef int (*traverse_func) (void *clientdata, int seqnr, void *data);

sql_extern list *list_create(fdestroy destroy);
sql_extern list *sa_list(sql_allocator *sa);
sql_extern list *list_new(sql_allocator *sa, fdestroy destroy);

sql_extern void list_destroy(list *l);
sql_extern int list_length(list *l);
sql_extern int list_empty(list *l);

sql_extern list *list_append(list *l, void *data);
sql_extern list *list_append_before(list *l, node *n, void *data);
sql_extern list *list_prepend(list *l, void *data);

sql_extern node *list_remove_node(list *l, node *n);
sql_extern void list_remove_data(list *l, void *data);
sql_extern void list_remove_list(list *l, list *data);
sql_extern void list_move_data(list *l, list *d, void *data);


sql_extern int list_traverse(list *l, traverse_func f, void *clientdata);

/* the compare function gets one element from the list and a key from the
 * as input from the find function 
 * Returns 0 if data and key are equal 
 * */
typedef int (*fcmp) (void *data, void *key);
typedef void *(*fcmpvalidate) (void *v1, void *v2, int *cmp);
typedef void *(*fvalidate) (void *v1, void *v2);
typedef int (*fcmp2) (void *data, void *v1, void *v2);
typedef void *(*fdup) (void *data);
typedef void *(*freduce) (void *v1, void *v2);
typedef void *(*freduce2) (sql_allocator *sa, void *v1, void *v2);
typedef void *(*fmap) (void *data, void *clientdata);

sql_extern void *list_traverse_with_validate(list *l, void *data, fvalidate cmp);
sql_extern void *list_append_with_validate(list *l, void *data, fvalidate cmp);
sql_extern void *list_append_sorted(list *l, void *data, fcmpvalidate cmp);
sql_extern node *list_find(list *l, void *key, fcmp cmp);
sql_extern int  list_position(list *l, void *val);
sql_extern void * list_fetch(list *l, int pos);
sql_extern list *list_select(list *l, void *key, fcmp cmp, fdup dup);
sql_extern list *list_order(list *l, fcmp cmp, fdup dup);
sql_extern list *list_distinct(list *l, fcmp cmp, fdup dup);
sql_extern void *list_reduce(list *l, freduce red, fdup dup);
sql_extern void *list_reduce2(list *l, freduce2 red, sql_allocator *sa);
sql_extern list *list_map(list *l, void *data, fmap f);
sql_extern int list_cmp(list *l1, list *l2, fcmp cmp);
/* cmp the lists in link order */
sql_extern int list_match(list *l1, list *l2, fcmp cmp);
/* match the lists (in any order) */
sql_extern list *list_sort(list *l, fkeyvalue key, fdup dup);
/* The sort function sorts the list using the key function, which 
 * translates the list item values into integer keyvalues. */
/* sometimes more complex functions are needed to compute a key, then
 * we can pass the keys via an array, to keysort */
sql_extern list *list_keysort(list *l, int *key, fdup dup);

sql_extern list *list_dup(list *l, fdup dup);
sql_extern list *list_merge(list *l, list *data, fdup dup);
sql_extern list *list_merge_destroy(list *l, list *data, fdup dup);

sql_extern void list_hash_delete(list *l, void *data, fcmp cmp);
sql_extern void* list_hash_add(list *l, void *data, fcmp cmp);

#endif /* LIST_H */
