#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gadgetFile.h"

typedef  long long  peanokey;    

#ifdef DOUBLEPRECISION             
#define FLOAT double
#else
#define FLOAT float
#endif

typedef struct particle_data_
{
    FLOAT Pos[3];                 
    FLOAT Mass;                   
    FLOAT Vel[3];                 
    FLOAT GravAccel[3];           
#ifdef PMGRID
    FLOAT GravPM[3];              
#endif
#ifdef FORCETEST
    FLOAT GravAccelDirect[3];     
#endif
    FLOAT Potential;              
    FLOAT OldAcc;                 
#ifndef LONGIDS
    unsigned int ID;              
#else
    unsigned long long ID;        
#endif

    int Type;                     
    int Ti_endstep;               
    int Ti_begstep;               
    float GravCost;               
#ifdef PSEUDOSYMMETRIC
    float AphysOld;               
#endif
} particle_data;

typedef struct peano_hilbert_data
{
    peanokey key;
    int index;
}*mp;

extern void peano_hilbert_keys(peanokey **keys, GadgetParticles p, float boxSize, int bits);
extern int saveToFileKeys(char *fileOutPath, char *mode, peanokey *keys, long numPart);

extern void peano_hilbert_inverse_keys(float **x, float **y, float **z, PHBins b, float boxSize, int bits);
extern int saveToCSVFileCoord(char *fileOutPath, char *mode, float *x, float *y, float *z, float boxSize, int bins, int numBins);
extern peanokey peano_hilbert_key(float x, float y, float z, float boxSize, int bits);

extern int compare_key(const void *a, const void *b);
extern void reorder_particles(void);
extern void peano_hilbert_inverse_key(float *fx, float *fy, float *fz, peanokey phkey, float boxSize, int bits);
