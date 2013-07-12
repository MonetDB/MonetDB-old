/*
 * The contents of this file are subject to the MonetDB Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.monetdb.org/Legal/MonetDBLicense
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * The Original Code is the MonetDB Database System.
 *
 * The Initial Developer of the Original Code is CWI.
 * Portions created by CWI are Copyright (C) 1997-July 2008 CWI.
 * Copyright August 2008-2013 MonetDB B.V.
 * All Rights Reserved.
 */


/* Trie data structure for RDF
 * Main purpose is to check find maximum CS (characteristic set)
 *
 * */


#ifndef __TRIE_H__
#define __TRIE_H__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "monetdb_config.h"
#include "algebra.h"
#include <gdk.h>

#define TRIE_ERROR -1 /* Error state, used in return functions */
#define TRIE_PATH -1 /*Indicates that a node is just part of a path*/

struct trie_node{
  oid key;                      
  int value;                    
  struct trie_node* right;      
  struct trie_node* left;        
  struct trie_node* children;        
  struct trie_node* parent; 
  int parentFreqIdx;
};


/* `init_trie' 
   - Initialize a trie. 
   - The pointed node `head' will have '\0' as key value equals to TRIE_PATH 
   - Memory will be allocated to the pointed node
   
   - Always use init_trie in pair with delete_trie (like malloc and free)   
*/
int 
init_trie(struct trie_node** head);

/* `delete_trie'
   - Frees all memory allocated during trie operations 
   
   - Should be used in pair with init_trie function.
*/
int
delete_trie(struct trie_node** head);

int count_trieleaf(struct trie_node** head, int *numLeaf);

/* `trie_insert':
   - Inserts a new pair <char*, int> at the trie.
   - Params: root node, the key and the value
   - Returns: the value `val' if everthing got fine, TRIE_ERROR otherwise 
   
   - Negative values nor NULL key nor "" are not allowed 
   (to simplify the implementation)
   - Repeated keys are not allowed, the old value will be overwritten 
*/
int 
trie_insert(struct trie_node* root, oid* key, int key_len, int val);

//int 
//trie_insertwithFreqCS(struct trie_node* root, oid* key, int key_len, int val, CSset *freqCSset);

/* `trie_remove'
   - Removes a key from the trie.
   - Params
   - Returns: the value `val' of the key in case of success, 
   TRIE_ERROR otherwise 
 */
int 
trie_remove(struct trie_node* root, oid* key, int key_len);


/* `trie_search'
   - Gets the value of key.  
   - Returns: the value (greater then zero), TRIE_ERROR otherwise
*/
int 
trie_search(struct trie_node* root, oid* key, int key_len);

#endif /* __TRIE_H__ */

