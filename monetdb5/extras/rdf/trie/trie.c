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


#include <trie.h>

int 
init_trie(struct trie_node** head)
{
	(*head) = malloc(sizeof (struct trie_node));
	(*head)->key = 0;		/* Assume that no node has the key (pOid) = 0 */
	(*head)->value = TRIE_PATH;
	(*head)->right = NULL;
	(*head)->left = NULL;
	(*head)->parent = NULL;
	(*head)->children = NULL;

	return 1;
}


int 
trie_insert(struct trie_node* root, oid* key, int key_len,int val)
{

	int i, found_child;

	struct trie_node* curr_node = NULL;
	struct trie_node* new_node = NULL;
	struct trie_node* iter = NULL;

	/*
	printf("Insert: \n");
	for (i = 0; i < key_len; i++){
		printf(BUNFMT " ", key[i]);
	}
	printf("\n");
	*/

	/* Negative values nor NULL keys are allowed */
	if(val < 0 || root == NULL)
		return -1; 

	curr_node = root;

	/* iterates over all key's elements. For each one,
	tries to advance in the trie reusing the path. When 
	the reusable part ends, start to add new nodes.
	*/

	for(i=0; i <= key_len; i++){ 
		if(i == key_len)return curr_node->value;
	
		found_child = 0; 
		for(iter=curr_node->children; iter != NULL; iter=iter->right)
		{ 
			if(iter->key == key[i])
			{
				found_child = 1;
				curr_node = iter;
				break;
			}
		}

		/* Adds a new node on the trie */
		if(!found_child){	
			new_node = malloc(sizeof(struct trie_node));
			new_node->key = key[i];
			/*If we are in the end of key, this node should get the
			value*/
			new_node->value = i == key_len-1 ? val:TRIE_PATH;	
			new_node->parent = curr_node;
 			new_node->children = NULL; 		// DUC: CHECK. Force the children of newnode to be NULL
			/*Updates the children linked list*/
			new_node->left = NULL;
			new_node->right = curr_node->children; 	
			if(curr_node->children != NULL)
				curr_node->children->left = new_node;
	
			curr_node->children = new_node; 	
	
			/*Next loop iteration consider the new node*/
			curr_node = new_node; 
		}
		else {
			if(i == key_len -1)
				curr_node->key = key[i];
		}
	}

	return 1;
}


int 
trie_search(struct trie_node* root, oid* key, int key_len)
{
	int i, found_child;

	struct trie_node* curr_node = NULL;
	struct trie_node* iter = NULL;

	/* Invalid keys */
	if(root == NULL || key == NULL || key_len==0)
	return -1;

	else
	{
	/* Starting with root, we do a kind of DFS: for every
		 char in the a node should be found. Is this not apply, we 
		 got an error. */
	curr_node = root; 
	for(i=0; i <= key_len; i++)
		{
		/* If we got at the end of key without fails in node 
		search, we can safely return the value */
		if(i == key_len)	
		return curr_node->value;	
		
		found_child = 0;
		for(iter=curr_node->children; iter != NULL; iter=iter->right)
		{
		if(iter->key == key[i])
			{
			curr_node = iter;	
			found_child = 1;
			break;
			}
		} 	
		if(!found_child)
		return TRIE_ERROR; 
		}
	}
	return TRIE_ERROR;
}


int 
trie_remove(struct trie_node* root, oid* key, int key_len)
{

	struct trie_node* curr_node;
	struct trie_node* parent_node;
	struct trie_node* iter;

	int found_child, i, ret_value;

	// key_len = strlen(key);

	/*Invalid conditions*/
	if(root == NULL || key == NULL || key_len == 0)
	return TRIE_ERROR;

	curr_node = root;

	/* Iterates over chars in the key in order to find the 
	 last node (where the value is in). */
	for(i =0; i<key_len; i++)
	{
	found_child = 0;
	for(iter = curr_node->children; iter!=NULL; iter=iter->right)
		{
		if(iter->key == key[i])
		{
		curr_node = iter;
		found_child= 1;
		break;
		}	
		}
	if(!found_child)
		return TRIE_ERROR; 
	}

	/*If we are here, we found the node. Now it is necessary 
	to backtracking in the path (using parent pointer) and
	freeing every node marked with TRIE_PATH. If we found a 
	node with a value different from TRIE_PATH, we stop.*/
	i=0;
	ret_value = curr_node->value;
	curr_node->value = -1;	
	for(;;)
	{
	/* Double linked list deletion */
	if(curr_node->children == NULL)
		{
		if(curr_node->parent == NULL)
		break;	
		
		else if(curr_node->parent->children == curr_node)	
		curr_node->parent->children = curr_node->right;	
		
		else if(curr_node->right != NULL)
		{
		curr_node->left->right = curr_node->right;
		curr_node->right->left = curr_node->left;	
		}	
		
		else	
		curr_node->left->right = NULL;
		
		/* Update the path with the parent */
		parent_node = curr_node->parent;	
		free(curr_node);
		curr_node = parent_node;	
		}
	else
		break; 
	}

	return ret_value;
}


int
delete_trie(struct trie_node** head)
{
	struct trie_node* iter;

	if(*head == NULL)
	return 1;

	/* This massive node deletion is just a DFS 
	 implemented with recursion: first free all children 
	 and then free the father. Just remember that the 
	 father maybe is a member of a linked list with its brothers! */

	for(iter=(*head); iter != NULL; iter=iter->right)
	{
	delete_trie(&(iter->children));

	if(iter->parent != NULL)
		{
		if(iter->parent->children == iter)	
		iter->parent->children = iter->right;	

		else if(iter->right != NULL)
		{
		iter->left->right = iter->right;
		iter->right->left = iter->left;	
		}		
		else	
		iter->left->right = NULL;
		} 
	free(iter);	
	}	
	// free(*head);
	return 0;
}

int
count_trieleaf(struct trie_node** head, int *numLeaf)
{
	struct trie_node* iter;

	assert(*head != NULL);

	for(iter=(*head); iter != NULL; iter=iter->right)
	{

		if (iter->children == NULL){ 
			*numLeaf = *numLeaf + 1;
		}
		else{
			count_trieleaf(&(iter->children),numLeaf);
		}

	}	

	return *numLeaf; 
}
