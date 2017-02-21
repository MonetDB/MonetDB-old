#include <sys/stat.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <assert.h>
#include <sys/mman.h>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MRSNAP_POS_OFFSET 268
#define MRSNAP_VEL_OFFSET 276  // +sum(npart)*12
#define MRSNAP_ID_OFFSET 284 // +sum(npart)*24

/* next assumes there are only positions, velocities and identifiers in a file */
#define MRSNAP_HASTHTABLE_OFFSET 308 // +sum(npart)*32

/*Default number of PHBins*/
#define MRS_PHBin_DEFAULT 5000
#define MRS_PARTICLE_DEFAULT 5000

/*Structures*/
typedef struct Header_ {
    char* fileName;
    int npartTotal; // = 0;
    int npart[6];
    double mass[6];
    double time, redshift;
    int flag_sfr, flag_feedback;
    int npart_total[6];
    int flag_cooling, num_files;
    double BoxSize, Omega0, OmegaLambda, HubbleParameter; // = -1;
    int flag_stellarage, flag_metals, hashtabsize;
    int headerOk;
    int first_cell, last_cell;
} Header;

typedef struct GadgetParticles_ {
    long numPart;
    int64_t *id;
    float *posX, *posY, *posZ;
    float *velX, *velY, *velZ;
} GadgetParticles;

typedef struct PHBin_ {
    int id, start, count;
} PHBin;

typedef struct PHBins_ {
    long numBins;
    int *id, *start, *count;
} PHBins;

/*Functions signature*/
extern int ReadGadgetFile(GadgetParticles* particles, char* filePath);

/* Read the  peano-hilbert hashtable from end of file*/
extern int GadgetPHBins(PHBins *bins, char *filePath);

extern void gadgetParticles_destroy(GadgetParticles* p);
extern void PHBins_destroy(PHBins* b);
extern void headerInit(Header *h, FILE *blob);
extern int PHBins_add(PHBins* b, int id, int start, int count);
extern int gadgetParticles_init(GadgetParticles *p);
extern int PHBins_init(PHBins* b);
