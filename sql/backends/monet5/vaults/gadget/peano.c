#include "peano.h"

static int quadrants[24][2][2][2] = {
    /* rotx=0, roty=0-3 */
    {{{0, 7}, {1, 6}}, {{3, 4}, {2, 5}}},
    {{{7, 4}, {6, 5}}, {{0, 3}, {1, 2}}},
    {{{4, 3}, {5, 2}}, {{7, 0}, {6, 1}}},
    {{{3, 0}, {2, 1}}, {{4, 7}, {5, 6}}},
    /* rotx=1, roty=0-3 */
    {{{1, 0}, {6, 7}}, {{2, 3}, {5, 4}}},
    {{{0, 3}, {7, 4}}, {{1, 2}, {6, 5}}},
    {{{3, 2}, {4, 5}}, {{0, 1}, {7, 6}}},
    {{{2, 1}, {5, 6}}, {{3, 0}, {4, 7}}},
    /* rotx=2, roty=0-3 */
    {{{6, 1}, {7, 0}}, {{5, 2}, {4, 3}}},
    {{{1, 2}, {0, 3}}, {{6, 5}, {7, 4}}},
    {{{2, 5}, {3, 4}}, {{1, 6}, {0, 7}}},
    {{{5, 6}, {4, 7}}, {{2, 1}, {3, 0}}},
    /* rotx=3, roty=0-3 */
    {{{7, 6}, {0, 1}}, {{4, 5}, {3, 2}}},
    {{{6, 5}, {1, 2}}, {{7, 4}, {0, 3}}},
    {{{5, 4}, {2, 3}}, {{6, 7}, {1, 0}}},
    {{{4, 7}, {3, 0}}, {{5, 6}, {2, 1}}},
    /* rotx=4, roty=0-3 */
    {{{6, 7}, {5, 4}}, {{1, 0}, {2, 3}}},
    {{{7, 0}, {4, 3}}, {{6, 1}, {5, 2}}},
    {{{0, 1}, {3, 2}}, {{7, 6}, {4, 5}}},
    {{{1, 6}, {2, 5}}, {{0, 7}, {3, 4}}},
    /* rotx=5, roty=0-3 */
    {{{2, 3}, {1, 0}}, {{5, 4}, {6, 7}}},
    {{{3, 4}, {0, 7}}, {{2, 5}, {1, 6}}},
    {{{4, 5}, {7, 6}}, {{3, 2}, {0, 1}}},
    {{{5, 2}, {6, 1}}, {{4, 3}, {7, 0}}}
};

static int rotxmap_table[24] = { 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 0, 1, 2, 3, 17, 18, 19, 16, 23, 20, 21, 22
};

static int rotymap_table[24] = { 1, 2, 3, 0, 16, 17, 18, 19,
    11, 8, 9, 10, 22, 23, 20, 21, 14, 15, 12, 13, 4, 5, 6, 7
};

static int rotx_table[8] = { 3, 0, 0, 2, 2, 0, 0, 1 };
static int roty_table[8] = { 0, 1, 1, 2, 2, 3, 3, 0 };

static int sense_table[8] = { -1, -1, -1, +1, +1, -1, -1, -1 };

int flag_quadrants_inverse = 1;
char quadrants_inverse_x[24][8];
char quadrants_inverse_y[24][8];
char quadrants_inverse_z[24][8];

int *Id;
int N_gas;               
int NumPart;             
particle_data *P, *DomainPartBuf;  

int compare_key(const void *a, const void *b)
{
    if(((struct peano_hilbert_data *) a)->key < (((struct peano_hilbert_data *) b)->key))
        return -1;

    if(((struct peano_hilbert_data *) a)->key > (((struct peano_hilbert_data *) b)->key))
        return +1;

    return 0;
}

void reorder_particles(void)
{
    int i;
    particle_data Psave, Psource;
    int idsource, idsave, dest;

    for(i = N_gas; i < NumPart; i++)
    {
        if(Id[i] != i)
        {
            Psource = P[i];
            idsource = Id[i];

            dest = Id[i];

            do
            {
                Psave = P[dest];
                idsave = Id[dest];

                P[dest] = Psource;
                Id[dest] = idsource;

                if(dest == i)
                    break;

                Psource = Psave;
                idsource = idsave;

                dest = idsource;
            }
            while(1);
        }
    }
}

static peanokey
peano_hilbert_key_(int x, int y, int z, int bits)
{
    int i, quad, bitx, bity, bitz;
    int mask, rotation, rotx, roty, sense;
    peanokey key;


    mask = 1 << (bits - 1);
    key = 0;
    rotation = 0;
    sense = 1;


    for(i = 0; i < bits; i++, mask >>= 1)
    {
        bitx = (x & mask) ? 1 : 0;
        bity = (y & mask) ? 1 : 0;
        bitz = (z & mask) ? 1 : 0;

        quad = quadrants[rotation][bitx][bity][bitz];

        key <<= 3;
        key += (sense == 1) ? (quad) : (7 - quad);

        rotx = rotx_table[quad];
        roty = roty_table[quad];
        sense *= sense_table[quad];

        while(rotx > 0)
        {
            rotation = rotxmap_table[rotation];
            rotx--;
        }

        while(roty > 0)
        {
            rotation = rotymap_table[rotation];
            roty--;
        }
    }

    return key;
}

peanokey
peano_hilbert_key(float x, float y, float z, float boxSize, int bits)
{
    int ncells = (1 << bits); // number of cells per cordinate axis
    int  i = (int) floor((x/boxSize)*ncells);
    int  j = (int) floor((y/boxSize)*ncells);
    int  k = (int) floor((z/boxSize)*ncells);
    
    return peano_hilbert_key_(i, j, k, bits);
}

void 
peano_hilbert_keys(peanokey **keys, GadgetParticles p, float boxSize, int bits)
{
    int i = 0;
    *keys = (peanokey*) malloc(sizeof(peanokey)  * p.numPart);

    for (i = 0; i < p.numPart; i++) {
        (*keys)[i] = peano_hilbert_key(p.posX[i], p.posY[i], p.posZ[i], boxSize, bits);
    }
}

int saveToFileKeys(char *fileOutPath, char *mode, peanokey *keys, long numPart) {
    int i = 0, res = -1;
    FILE *fp = NULL;

    if ( !(fp = fopen(fileOutPath, mode))) {
        fprintf(stderr, "saveToFileKeys: failed to open file %s!!!\n", fileOutPath);
        goto out;
    }

    if (mode[1] == 'b')
        fwrite((peanokey*)keys, sizeof(peanokey), numPart, fp);
    else
        for (i = 0; i < numPart; i++)
            fprintf(fp, "%llu\n", keys[i]);

    fflush(fp);
    fclose(fp);
    res = 0;
out:
    return res;
}

static void
peano_hilbert_key_inverse(peanokey key, int bits, int *x, int *y, int *z)
{
    int i, keypart, bitx, bity, bitz, mask, quad, rotation, shift;
    char sense, rotx, roty;

    if(flag_quadrants_inverse)
    {
        flag_quadrants_inverse = 0;
        for(rotation = 0; rotation < 24; rotation++)
            for(bitx = 0; bitx < 2; bitx++)
                for(bity = 0; bity < 2; bity++)
                    for(bitz = 0; bitz < 2; bitz++)
                    {
                        quad = quadrants[rotation][bitx][bity][bitz];
                        quadrants_inverse_x[rotation][quad] = bitx;
                        quadrants_inverse_y[rotation][quad] = bity;
                        quadrants_inverse_z[rotation][quad] = bitz;
                    }
    }

    shift = 3 * (bits - 1);
    mask = 7 << shift;

    rotation = 0;
    sense = 1;

    *x = *y = *z = 0;

    for(i = 0; i < bits; i++, mask >>= 3, shift -= 3)
    {
        keypart = (key & mask) >> shift;

        quad = (sense == 1) ? (keypart) : (7 - keypart);

        *x = (*x << 1) + quadrants_inverse_x[rotation][quad];
        *y = (*y << 1) + quadrants_inverse_y[rotation][quad];
        *z = (*z << 1) + quadrants_inverse_z[rotation][quad];

        rotx = rotx_table[quad];
        roty = roty_table[quad];
        sense *= sense_table[quad];

        while(rotx > 0)
        {
            rotation = rotxmap_table[rotation];
            rotx--;
        }

        while(roty > 0)
        {
            rotation = rotymap_table[rotation];
            roty--;
        }
    }
}

static void 
peano_hilbert_inverse_keys_(int **x, int **y, int **z, PHBins b, int bits)
{
    int i = 0;

    *x = (int*) malloc(sizeof(int)  * b.numBins);
    *y = (int*) malloc(sizeof(int)  * b.numBins);
    *z = (int*) malloc(sizeof(int)  * b.numBins);

    for (i = 0; i < b.numBins; i++) {
        peano_hilbert_key_inverse(b.id[i], bits, &((*x)[i]), &((*y)[i]), &((*z)[i]));
    }
}

void 
peano_hilbert_inverse_keys(float **fx, float **fy, float **fz, PHBins b, float boxSize, int bits)
{
    int i = 0, *x = NULL, *y = NULL, *z = NULL;
    int ncells = (1 << bits); // number of cells per cordinate axis

     peano_hilbert_inverse_keys_(&x, &y, &z, b, bits);

    *fx = (float*) malloc(sizeof(float)  * b.numBins);
    *fy = (float*) malloc(sizeof(float)  * b.numBins);
    *fz = (float*) malloc(sizeof(float)  * b.numBins);

    for (i = 0; i < b.numBins; i++) {
        (*fx)[i] = (x[i]*boxSize)/ncells;
        (*fy)[i] = (y[i]*boxSize)/ncells;
        (*fz)[i] = (z[i]*boxSize)/ncells;
    }

}

void 
peano_hilbert_inverse_key(float *fx, float *fy, float *fz, peanokey phkey, float boxSize, int bits)
{
    int x, y, z;
    int ncells = (1 << bits); // number of cells per cordinate axis

    peano_hilbert_key_inverse(phkey, bits, &x, &y, &z);

    *fx = (x*boxSize)/ncells;
    *fy = (y*boxSize)/ncells;
    *fz = (z*boxSize)/ncells;
}

int
saveToCSVFileCoord(char *fileOutPath, char *mode, float *x, float *y, float *z, float boxSize, int bits, int numBins)
{
    int res = -1, i = 0;
    FILE *fp = NULL;
    int ncells = (1 << bits); // number of cells per cordinate axis

    /*Force to be ASCII file*/
    mode[1] = '\0';

    if ( !(fp = fopen(fileOutPath, mode))) {
        fprintf(stderr, "saveToFileKeys: failed to open file %s!!!\n", fileOutPath);
        goto out;
    }
    
    for (i = 0; i < numBins; i++) {
        fprintf(fp, "%f: %f %f %f\n", boxSize/ncells, x[i], y[i], z[i]);
    }
    
    fflush(fp);
    fclose(fp);
    res = 0;
out:
    return res;
}
