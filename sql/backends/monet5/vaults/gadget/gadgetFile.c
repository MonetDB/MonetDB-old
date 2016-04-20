#include "gadgetFile.h"

int
gadgetParticles_init(GadgetParticles *p)
{
    int res = -1;
    p->numPart = 0;
    p->id = NULL;
    p->posX = NULL;
    p->posY = NULL;
    p->posZ = NULL;
    p->velX = NULL;
    p->velY = NULL;
    p->velZ = NULL;

    if ( !(p->id = (int64_t*) malloc(sizeof(int64_t)*MRS_PARTICLE_DEFAULT)) ) {
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->posX = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->posY = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        free(p->posX);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->posZ = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        free(p->posX);
        free(p->posY);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->velX = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        free(p->posX);
        free(p->posY);
        free(p->posZ);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->velY = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        free(p->posX);
        free(p->posY);
        free(p->posZ);
        free(p->velX);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    if ( !(p->velZ = (float*) malloc(sizeof(float)*MRS_PARTICLE_DEFAULT)) ) {
        free(p->id);
        free(p->posX);
        free(p->posY);
        free(p->posZ);
        free(p->velX);
        free(p->velY);
        fprintf(stderr, "gadgetParticlesInit: malloc failed!!!\n");
        goto out;
    }

    res = 0;
out:
    return res;
}

void
gadgetParticles_destroy(GadgetParticles* p)
{
    if (p->id) {
        free (p->id);
        p->id = NULL;
    }
    if (p->posX) {
        free (p->posX);
        p->posX = NULL;
    }
    if (p->posY) {
        free (p->posY);
        p->posY = NULL;
    }
    if (p->posZ) {
        free (p->posZ);
        p->posZ = NULL;
    }
    if (p->velX) {
        free (p->velX);
        p->velX = NULL;
    }
    if (p->velY) {
        free (p->velY);
        p->velY = NULL;
    }
    if (p->velZ) {
        free (p->velZ);
        p->velZ = NULL;
    }

    p->numPart = 0;
}

static int
gadgetParticles_add(GadgetParticles* p, int64_t id, float pX, float pY, float pZ, float vX, float vY, float vZ)
{
    int res = -1;
    if (p->numPart && (p->numPart % MRS_PARTICLE_DEFAULT) == 0) {
        p->id = realloc(p->id, sizeof(int64_t) * (p->numPart*2));
        p->posX = realloc(p->posX, sizeof(float) * (p->numPart*2));
        p->posY = realloc(p->posY, sizeof(float) * (p->numPart*2));
        p->posZ = realloc(p->posZ, sizeof(float) * (p->numPart*2));
        p->velX = realloc(p->velX, sizeof(float) * (p->numPart*2));
        p->velY = realloc(p->velY, sizeof(float) * (p->numPart*2));
        p->velZ = realloc(p->velZ, sizeof(float) * (p->numPart*2));

        if ( !p->id || !p->posX || !p->posY || !p->posZ || !p->velX || !p->velY || !p->velZ ) {
            fprintf(stderr, "gadgetParticlesAdd: realloc failed!!!\n");
            goto out;
        }
    }

    p->id[p->numPart] = id;
    p->posX[p->numPart] = pX;
    p->posY[p->numPart] = pY;
    p->posZ[p->numPart] = pZ;
    p->velX[p->numPart] = vX;
    p->velY[p->numPart] = vY;
    p->velZ[p->numPart] = vZ;
    p->numPart++;

    res = 0;
out:
    return res;
}

int
PHBins_init(PHBins* b)
{
    int res = -1;
    b->id = b->start = b->count = NULL;

    if (!(b->id = (int*) malloc(sizeof(int)*MRS_PHBin_DEFAULT))) {
        fprintf(stderr, "PHBins_init: malloc failed!!!\n");
        goto out;
    }
    if (!(b->start = (int*) malloc(sizeof(int)*MRS_PHBin_DEFAULT))) {
        free(b->id);
        b->id = NULL;
        fprintf(stderr, "PHBins_init: malloc failed!!!\n");
        goto out;
    }
    if (!(b->count = (int*) malloc(sizeof(int)*MRS_PHBin_DEFAULT))) {
        free(b->id);
        b->id = NULL;
        free(b->start);
        b->start = NULL;
        fprintf(stderr, "PHBins_init: malloc failed!!!\n");
        goto out;
    }
    b->numBins = 0;

    res = 0;
out:
    return res;
}

void
PHBins_destroy(PHBins* b)
{
    if (b->id) {
        free(b->id);
        b->id = NULL;
    }
    if (b->start) {
        free(b->start);
        b->id = NULL;
    }
    if (b->count) {
        free(b->count);
        b->id = NULL;
    }

    b->numBins = 0;
}

int
PHBins_add(PHBins* b, int id, int start, int count)
{
    int res = -1;
    if (b->numBins && (b->numBins % MRS_PHBin_DEFAULT) == 0) {
        b->id = realloc(b->id, sizeof(int)*(b->numBins*2));
        b->start = realloc(b->start, sizeof(int)*(b->numBins*2));
        b->count = realloc(b->count, sizeof(int)*(b->numBins*2));

        if (!b->id || !b->start || !b->count) {
            fprintf(stderr, "PHBinList_add: realloc failed!!!\n");
            goto out;
        }
    }

    b->id[b->numBins] = id;
    b->start[b->numBins] = start;
    b->count[b->numBins] = count;
    b->numBins++;

    res = 0;
out:
    return res;
}




static void
headerInit_(Header* h, char* bytes)
{
    int k = 0, index = 0;
    int rec1 = *(int*) (&bytes[index]), rec2;
    index += 4;
    h->npartTotal = 0;

    for (k = 0; k < 6; k++)
    {
        h->npart[k] = *(int*) (&bytes[index]);
        index += 4;
        h->npartTotal += h->npart[k];
    }

    for (k = 0; k < 6; k++)
    {
        h->mass[k] = *(double*) (&bytes[index]);
        index += 8;
    }

    h->time = *(double*) (&bytes[index]);
    index += 8;
    h->redshift = *(double*) (&bytes[index]);
    index += 8;
    h->flag_sfr = *(int*) (&bytes[index]);
    index += 4;
    h->flag_feedback = *(int*) (&bytes[index]);
    index += 4;
    for (k = 0; k < 6; k++)
    {
        h->npart_total[k] = *(int*) (&bytes[index]);
        index += 4;
    }
    h->flag_cooling = *(int*) (&bytes[index]);
    index += 4;
    h->num_files = *(int*) (&bytes[index]);
    index += 4;
    h->BoxSize = *(double*) (&bytes[index]);
    index += 8;
    h->Omega0 = *(double*) (&bytes[index]);
    index += 8;
    h->OmegaLambda = *(double*) (&bytes[index]);
    index += 8;
    h->HubbleParameter = *(double*) (&bytes[index]);
    index += 8;
    h->flag_stellarage = *(int*) (&bytes[index]);
    index += 4;
    h->flag_metals = *(int*) (&bytes[index]);
    index += 4;
    h->hashtabsize = *(int*) (&bytes[index]);
    index += 88;
    rec2 = *(int*) (&bytes[index]);

    // int rec2 = rec1; // HACK
    if (rec1 != rec2)
        h->headerOk = 0;
    else
        h->headerOk = 1;
}


// The compiler complains if headerlength is defined as a variable
#define HEADERLENGTH 264

void
headerInit(Header *h, FILE *blob)
{
    long index = 0;
    int npart;
    char headerbytes[HEADERLENGTH];
    long retval = fread(headerbytes, HEADERLENGTH, 1, blob);
    int binbytes[2];
    (void) retval;
    assert(retval);

    headerInit_(h, headerbytes);
    assert(h->headerOk);
    npart = h->npartTotal;

    if (h->hashtabsize > 0) // has bins
    {
        index = 292 + npart * 32;
        fseek(blob, index, SEEK_SET);

        retval = fread(binbytes, 4, 2, blob);
        assert(retval);

        h->first_cell = *(int*) (&binbytes[0]);
        h->last_cell = *(int*) (&binbytes[1]);
        printf("Print first_cell %d and last_cell %d\n", h->first_cell, h->last_cell);
    }
    else
    {
        h->first_cell = -1;
        h->last_cell = -1;
    }
}

int ReadGadgetFile(GadgetParticles* particles, char* filePath)
{
    Header h;
    int res = -1, i = 0;
    FILE *stream = NULL;
    char *pos = NULL, *vel=NULL, *id=NULL;
    long index = MRSNAP_POS_OFFSET;
    int retval;
    (void) retval;

    stream = fopen(filePath, "rb");
    if (!stream) {
        fprintf(stderr, "ReadGadgetFile: failed to open file %s!!!\n", filePath);
        goto out;
    }

    headerInit(&h, stream);

    if (gadgetParticles_init(particles)) {
        fprintf(stderr, "ReadGadgetFile: gadgetParticles_init failed!!!\n");
        goto out;
    }
    if ( !(pos = (char *) malloc(sizeof(char) * 12 * h.npartTotal))) {
        fprintf(stderr, "ReadGadgetFile: malloc failed!!!\n");
        goto out;
    }
    if ( !(vel = (char *) malloc(sizeof(char) * 12 * h.npartTotal))) {
        fprintf(stderr, "ReadGadgetFile: malloc failed!!!\n");
        goto out;
    }
    if ( !(id = (char *) malloc(sizeof(char) * 8 * h.npartTotal))) {
        fprintf(stderr, "ReadGadgetFile: malloc failed!!!\n");
        goto out;
    }

    fseek(stream, index, SEEK_SET);
    retval = fread(pos, 12 * h.npartTotal, 1, stream);
    assert(retval);

    index = MRSNAP_VEL_OFFSET + h.npartTotal * 12; // skip over positions
    fseek(stream, index, SEEK_SET);
    retval = fread(vel, 12 * h.npartTotal, 1, stream);
    assert(retval);

    index = MRSNAP_ID_OFFSET + h.npartTotal * 24; // skip over positions and velocities
    fseek(stream, index, SEEK_SET);
    retval = fread(id, 8 * h.npartTotal, 1, stream);
    assert(retval);

    for (i = 0; i < h.npartTotal; i++)
    {
        int j = i * 12;
        gadgetParticles_add(particles, *(int64_t*)(&id[i*8]), *(float*)(&pos[j]), *(float*)(&pos[j+4]), *(float*)(&pos[j+8]), *(float*)(&vel[j]), *(float*)(&vel[j+4]), *(float*)(&vel[j+8]));
    }

    res = 0;
out:
    if (stream)
        fclose(stream);

    //Free structures
    if (pos)
        free(pos);
    if (vel)
        free(vel);
    if (id)
        free(id);

    return res;
}

/*Read the  peano-hilbert hashtable from end of file*/
int GadgetPHBins(PHBins *bins, char *filePath)
{
    Header h;
    int res = -1, i = 0;
    FILE *stream = NULL;
    stream = fopen(filePath, "rb");

    if (!stream) {
        fprintf(stderr, "GadgetPHBins: failed to open file %s!!!\n", filePath);
        goto out;
    }

    headerInit(&h, stream);

    //NOTE: convert ArrayList to a array of pointers to PHBin
    //ArrayList bins = new ArrayList();
    if (PHBins_init(bins)) {
        fprintf(stderr, "GadgetPHBins: PHBinList_init failed!!!\n");
        goto out;
    }

    if (h.hashtabsize > 0)
    {
        long nbins = h.last_cell - h.first_cell + 1;
        char* binbytes = NULL;
        PHBin *prev = NULL;
        long index, retval;
        (void) retval;

        binbytes = (char*) malloc(sizeof(char) * nbins * 4);
        if (!binbytes) {
            PHBins_destroy(bins);
            fprintf(stderr, "GadgetPHBins: Malloc of binbytes failed!!!\n");
            goto out;
        }

        index = 308 + h.npartTotal * 32;
        fseek(stream, index, SEEK_SET);

        retval = fread(binbytes, nbins * 4, 1, stream);
        assert(retval);

        for (i = 0; i < nbins; i++)
        {
            PHBin *bin = (PHBin *) malloc (sizeof(PHBin));
            bin->id = h.first_cell + i;
            bin->start = *(int*) (&binbytes[i * 4]);
            if (prev != NULL)
            {
                prev->count = bin->start - prev->start;
                PHBins_add(bins, prev->id, prev->start, prev->count);
            }
            if (prev)
                free(prev);
            prev = bin;
        }
        prev->count = h.npartTotal - prev->start;
        PHBins_add(bins, prev->id, prev->start, prev->count);
        if (prev)
            free(prev);
        if (binbytes)
            free(binbytes);
    }

    res = 0;
out:

    if (stream)
        fclose(stream);

    return res;
}
