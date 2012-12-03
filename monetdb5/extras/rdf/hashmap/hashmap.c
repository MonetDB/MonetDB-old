/*
 * Generic map implementation.
 */
#include <hashmap.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_SIZE (256)
#define MAX_CHAIN_LENGTH (8)



/*
 * Return an empty hashmap, or NULL on failure.
 */
map_t hashmap_new(void) {
	hashmap_map* m = (hashmap_map*) malloc(sizeof(hashmap_map));
	if(!m) goto err;

	m->data = (hashmap_element*) calloc(INITIAL_SIZE, sizeof(hashmap_element));
	if(!m->data) goto err;

	m->table_size = INITIAL_SIZE;
	m->size = 0;

	return m;
	err:
		if (m)
			hashmap_free(m);
		return NULL;
}

/*
 * Hashing function for a set of values
 * Rely on djb2 http://www.cse.yorku.ca/~oz/hash.html
 *
 */
static unsigned int hashmap_hash_int(hashmap_map * m, int* key, int num){
	unsigned int hashCode = 5381u; 
	int i; 

	for (i = 0; i < num; i++){
		hashCode = ((hashCode << 5) + hashCode) + key[i];
	}
	
	// return 0x7fffffff & hashCode 
	return hashCode % m->table_size;
}

/* Compare two list of integers 
 * Return 0 if they are same
 * Return 1 if they are different
 * */
static char intsetcmp(int* key1, int* key2, int num){
	int i; 
	 
	for (i = 0; i < num; i++){
		if (key1[i] != key2[i]) return 1; 	
	}
	return 0; 
}

/*
 * Return the integer of the location in data
 * to store the point to the item, or MAP_FULL.
 */
static int hashmap_hash(map_t in, int* key, int num){
	int curr;
	int i;

	/* Cast the hashmap */
	hashmap_map* m = (hashmap_map *) in;

	/* If full, return immediately */
	if(m->size >= (m->table_size/2)) return MAP_FULL;

	/* Find the best index */
	curr = hashmap_hash_int(m, key, num);

	/* Linear probing */
	for(i = 0; i< MAX_CHAIN_LENGTH; i++){
		if(m->data[curr].in_use == 0)
			return curr;

		if(m->data[curr].in_use == 1 && (m->data[curr].num == num) && (intsetcmp (m->data[curr].key,key, num)==0))
			return curr;

		curr = (curr + 1) % m->table_size;
	}

	return MAP_FULL;
}

/*
 * Doubles the size of the hashmap, and rehashes all the elements
 */
static int hashmap_rehash(map_t in){
	int i;
	int old_size;
	hashmap_element* curr;

	/* Setup the new elements */
	hashmap_map *m = (hashmap_map *) in;
	hashmap_element* temp = (hashmap_element *)
		calloc(2 * m->table_size, sizeof(hashmap_element));
	if(!temp) return MAP_OMEM;

	/* Update the array */
	curr = m->data;
	m->data = temp;

	/* Update the size */
	old_size = m->table_size;
	m->table_size = 2 * m->table_size;
	m->size = 0;

	/* Rehash the elements */
	for(i = 0; i < old_size; i++){
		int status;

		if (curr[i].in_use == 0)
			continue;
            
		status = hashmap_put(m, curr[i].key, curr[i].num, curr[i].data);
		if (status != MAP_OK)
			return status;
	}

	free(curr);

	return MAP_OK;
}

/*
 * Add a pointer to the hashmap with some key
 */
int hashmap_put(map_t in, int* key, int num,  any_t value){
	int index;
	hashmap_map* m;

	/* Cast the hashmap */
	m = (hashmap_map *) in;

	/* Find a place to put our value */
	index = hashmap_hash(in, key, num);
	while(index == MAP_FULL){
		if (hashmap_rehash(in) == MAP_OMEM) {
			return MAP_OMEM;
		}
		index = hashmap_hash(in, key, num);
	}

	/* Set the data */
	m->data[index].data = value;
	m->data[index].key = key;
	m->data[index].num = num; 
	m->data[index].in_use = 1;
	m->data[index].freq = 1; 
	m->size++; 

	return MAP_OK;
}

/*
 * Get your pointer out of the hashmap with a key
 */
int hashmap_get(map_t in, int* key, int num,  any_t *arg, char isUpdateFrequency, int *retfreq){
	int curr;
	int i;
	hashmap_map* m;

	/* Cast the hashmap */
	m = (hashmap_map *) in;

	/* Find data location */
	curr = hashmap_hash_int(m, key, num);

	/* Linear probing, if necessary */
	for(i = 0; i<MAX_CHAIN_LENGTH; i++){

		int in_use = m->data[curr].in_use;
		if (in_use == 1){
			if ((m->data[curr].num == num) && (intsetcmp(m->data[curr].key,key,num)==0)){
				*arg = (m->data[curr].data);
				if (isUpdateFrequency == 1){
					m->data[curr].freq++;
					*retfreq = m->data[curr].freq; 
				}
				return MAP_OK;
            		}
		}
		curr = (curr + 1) % m->table_size;
	}

	*arg = NULL;

	/* Not found */
	return MAP_MISSING;
}



/*
 * Iterate the function parameter over each element in the hashmap.  The
 * additional any_t argument is passed to the function as its first
 * argument and the hashmap element is the second.
 */
int hashmap_iterate(map_t in, PFany f, any_t item) {
	int i;

	/* Cast the hashmap */
	hashmap_map* m = (hashmap_map*) in;

	/* On empty hashmap, return immediately */
	if (hashmap_length(m) <= 0)
		return MAP_MISSING;	

	/* Linear probing */
	for(i = 0; i< m->table_size; i++)
		if(m->data[i].in_use != 0) {
			any_t data = (any_t) (m->data[i].data);
			int status = f(item, data);
			if (status != MAP_OK) {
				return status;
			}
		}

    return MAP_OK;
}

/*
 * This function is used for filtering the CSs only:
 *
 * Iterate over the hashmap and show the values from 
 * all the hash element that has the frequent value > freqthreshold
 *
 * */

int hashmap_iterate_threshold(map_t in, int freqthreshold){

	int i;
	int count = 0; 

	/* Cast the hashmap */
	hashmap_map* m = (hashmap_map*) in;
	
	/* On empty hashmap, return immediately */
	if (hashmap_length(m) <= 0)
		return MAP_MISSING;	

	/* Linear probing */
	for(i = 0; i< m->table_size; i++)
		if(m->data[i].in_use != 0) {
			if (m->data[i].freq > freqthreshold){
				//any_t data = (any_t) (m->data[i].data);
				count++; 
			}
		}

    return count;
}

/*
 * Remove an element with that key from the map
 */
int hashmap_remove(map_t in, int* key, int num){
	int i;
	int curr;
	hashmap_map* m;

	/* Cast the hashmap */
	m = (hashmap_map *) in;

	/* Find key */
	curr = hashmap_hash_int(m, key, num);

	/* Linear probing, if necessary */
	for(i = 0; i<MAX_CHAIN_LENGTH; i++){

        int in_use = m->data[curr].in_use;
        if (in_use == 1){
	    if ((m->data[curr].num == num) && (intsetcmp(m->data[curr].key,key,num)==0)){
                /* Blank out the fields */
                m->data[curr].in_use = 0;
		m->data[curr].num = 0; 
		m->data[curr].freq = 1; 
                m->data[curr].data = NULL;
                m->data[curr].key = NULL;

                /* Reduce the size */
                m->size--;
                return MAP_OK;
            }
		}
		curr = (curr + 1) % m->table_size;
	}

	/* Data not found */
	return MAP_MISSING;
}

/* Deallocate the hashmap */
void hashmap_free(map_t in){
	hashmap_map* m = (hashmap_map*) in;

	// Free all the element in the hashmap
	int i;
	for(i = 0; i< m->table_size; i++){
		free(m->data[i].key);
		free(m->data[i].data);
	}
	
	free(m->data);
	free(m);
}

/* Return the length of the hashmap */
int hashmap_length(map_t in){
	hashmap_map* m = (hashmap_map *) in;
	if(m != NULL) return m->size;
	else return 0;
}
