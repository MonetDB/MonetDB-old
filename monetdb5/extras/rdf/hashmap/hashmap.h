/* HashMap for the characteristic sets (CSs') in RDF */

/* Originally by Elliot C Back - http://elliottback.com/wp/hashmap-implementation-in-c/
 * Modified by Minh-Duc Pham to support key as a set of integer values 
 * + various functions for collecting data statistic
 * and for specific data analysis of RDF triples 
 *
 */

#ifndef __HASHMAP_H__
#define __HASHMAP_H__

#define MAP_MISSING -3  /* No such element */
#define MAP_FULL -2 	/* Hashmap is full */
#define MAP_OMEM -1 	/* Out of Memory */
#define MAP_OK 0 	/* OK */

/*
 * any_t is a pointer.  This allows you to put arbitrary structures in
 * the hashmap.
 */
typedef void *any_t;

/*
 * PFany is a pointer to a function that can take two any_t arguments
 * and return an integer. Returns status code..
 */
typedef int (*PFany)(any_t, any_t);

/*
 * map_t is a pointer to an internally maintained data structure.
 * Clients of this package do not need to know how hashmaps are
 * represented.  They see and manipulate only map_t's.
 */
typedef any_t map_t;

/* We need to keep keys and values */
typedef struct _hashmap_element{
	int* key;
	int num; 
	int freq; 	
	char in_use;
	any_t data;
} hashmap_element;

/* A hashmap has some maximum size and current size,
 * as well as the data to hold. */
typedef struct _hashmap_map{
	int table_size;
	int size;
	hashmap_element *data;
} hashmap_map;

/*
 * Return an empty hashmap. Returns NULL if empty.
*/
extern map_t hashmap_new(void);

/*
 * Iteratively call f with argument (item, data) for
 * each element data in the hashmap. The function must
 * return a map status code. If it returns anything other
 * than MAP_OK the traversal is terminated. f must
 * not reenter any hashmap functions, or deadlock may arise.
 */
extern int hashmap_iterate(map_t in, PFany f, any_t item);

/*
 * This function is used for filtering the CSs only:
 *
 * Iterate over the hashmap and show the values from 
 * all the hash element that has the frequent value > freqthreshold
 *
 * */
extern int hashmap_iterate_threshold(map_t in, int freqthreshold);


/*
 * Collect the number of CSs for each support value ranging 
 * from 1 to maxfreqthreshold
 *
 * */

int hashmap_statistic_CSbysupport_cummulative(map_t in, int* ret, int maxfreqthreshold);


/*
 * Collect the number of CSs for each support value ranging 
 * from 1 to maxfreqthreshold
 *
 * */

int hashmap_statistic_CSbysupport(map_t in, int* ret, int maxfreqthreshold);

/*
 * This function is retrieve list of <num of CSs with the same size> 
 * e.g., <3,4,5> ==> 3 CSs has size 1,  
 * 
 * Note: size of ret >= maximum number of items (properties) in one CS
 * 
 * */

extern int hashmap_statistic_groupcs_by_size(map_t in, int* ret);

/*
 * Add an element to the hashmap. Return MAP_OK or MAP_OMEM.
 */
extern int hashmap_put(map_t in, int* key, int num,  any_t value);

/*
 * Add a pointer to the hashmap with some key
 * This function is ONLY used for the case of 
 * the predicate in RDF triple
 */
extern int hashmap_put_forP(map_t in, int* key, int num, any_t value, int support);

/*
 * Get an element from the hashmap. Return MAP_OK or MAP_MISSING.
 */
extern int hashmap_get(map_t in, int* key, int num, any_t *arg, char isUpdateFreq, int *retfreq);


/*
 * Get your pointer out of the hashmap with a key
 * This function is ONLY used for the case of 
 * the predicate in RDF triple
 */

extern int hashmap_get_forP(map_t in, int* key, any_t *arg);

/*
 * Remove an element from the hashmap. Return MAP_OK or MAP_MISSING.
 */
extern int hashmap_remove(map_t in, int* key, int num);

/*
 * Get any element. Return MAP_OK or MAP_MISSING.
 * remove - should the element be removed from the hashmap
 */
extern int hashmap_get_one(map_t in, any_t *arg, int remove);

/*
 * Free the hashmap
 */
extern void hashmap_free(map_t in);

/*
 * Get the current size of a hashmap
 */
extern int hashmap_length(map_t in);

#endif /*__HASHMAP_H__*/
