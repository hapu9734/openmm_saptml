/**
 * Field at every site due to the permanent (geometry-dependent) charges of the other molecules.
 * Forked from OpenMM's AMOEBA multipoleFixedField kernel; charges only, TTM4-F damping.
 */
#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real4 posq;
    real3 field;
    float damp;
    int water, type;
} AtomData;

inline DEVICE AtomData loadAtomData(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
    AtomData data;
    data.posq = posq[atom];
    data.damp = damping[atom];
    data.water = atomInfo[atom].y;
    data.type = atomInfo[atom].x;
    data.field = make_real3(0);
    return data;
}

#ifdef USE_EWALD
DEVICE void computeOneInteraction(AtomData* atom1, LOCAL_ARG AtomData* atom2, real3 deltaR, int pc, real3* fields) {
    real r2 = dot(deltaR, deltaR);
    if (r2 <= CUTOFF_SQUARED) {
        real r = SQRT(r2);
        real ralpha = EWALD_ALPHA*r;
        real exp2a = EXP(-(ralpha*ralpha));
#ifdef USE_DOUBLE_PRECISION
        const real erfcAlphaR = erfc(ralpha);
#else
        const real t = RECIP(1.0f+0.3275911f*ralpha);
        const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
#endif
        real bn0 = erfcAlphaR/r;
        real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
        real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
        alsq2n *= alsq2;
        real bn1 = (bn0+alsq2n*exp2a)/r2;
        real s3, s5, s7;
        if (pc == 1)
            s3 = 0;
        else
            tholeScales357(r, atom1->damp*atom2->damp, THOLE_CD, &s3, &s5, &s7);
        real rr3 = (1-s3)/(r2*r);
        // field at i due to charge at k, and vice versa, with the damped bare part removed
        fields[0] = -deltaR*(atom2->posq.w*(bn1-rr3));
        fields[1] = deltaR*(atom1->posq.w*(bn1-rr3));
    }
    else {
        fields[0] = make_real3(0);
        fields[1] = make_real3(0);
    }
}
#else
DEVICE void computeOneInteraction(AtomData* atom1, LOCAL_ARG AtomData* atom2, real3 deltaR, int pc, real3* fields) {
    if (pc == 1) {
        fields[0] = make_real3(0);
        fields[1] = make_real3(0);
        return;
    }
    real r2 = dot(deltaR, deltaR);
    real r = SQRT(r2);
    real s3, s5, s7;
    tholeScales357(r, atom1->damp*atom2->damp, THOLE_CD, &s3, &s5, &s7);
    real rr3 = s3/(r2*r);
    fields[0] = -deltaR*(atom2->posq.w*rr3);
    fields[1] = deltaR*(atom1->posq.w*rr3);
}
#endif

KERNEL void computeFixedField(
        GLOBAL mm_ulong* RESTRICT fieldBuffers, GLOBAL mm_ulong* RESTRICT fieldPolarBuffers, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int2* RESTRICT exclusionTiles, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const unsigned int* RESTRICT interactingAtoms,
#endif
        GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
    const unsigned int totalWarps = (GLOBAL_SIZE)/TILE_SIZE;
    const unsigned int warp = (GLOBAL_ID)/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
    LOCAL AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions (intra-molecular pairs).

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        unsigned int atom1 = x*TILE_SIZE + tgx;
        AtomData data = loadAtomData(atom1, posq, damping, atomInfo);
        if (x == y) {
            localData[LOCAL_ID].posq = data.posq;
            localData[LOCAL_ID].damp = data.damp;
            localData[LOCAL_ID].water = data.water;
            localData[LOCAL_ID].type = data.type;
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+j].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    computeOneInteraction(&data, &localData[tbx+j], delta, pairClass(data.water, localData[tbx+j].water), fields);
                    data.field += fields[0];
                }
            }
            SYNC_WARPS;
        }
        else {
            unsigned int j = y*TILE_SIZE + tgx;
            localData[LOCAL_ID] = loadAtomData(j, posq, damping, atomInfo);
            unsigned int tj = tgx;
            SYNC_WARPS;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+tj].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    computeOneInteraction(&data, &localData[tbx+tj], delta, pairClass(data.water, localData[tbx+tj].water), fields);
                    data.field += fields[0];
                    localData[tbx+tj].field += fields[1];
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;
        }
        unsigned int offset = x*TILE_SIZE + tgx;
        ATOMIC_ADD(&fieldBuffers[offset], (mm_ulong) realToFixedPoint(data.field.x));
        ATOMIC_ADD(&fieldBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.y));
        ATOMIC_ADD(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.z));
        ATOMIC_ADD(&fieldPolarBuffers[offset], (mm_ulong) realToFixedPoint(data.field.x));
        ATOMIC_ADD(&fieldPolarBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.y));
        ATOMIC_ADD(&fieldPolarBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.z));
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            ATOMIC_ADD(&fieldBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
            ATOMIC_ADD(&fieldBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
            ATOMIC_ADD(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
            ATOMIC_ADD(&fieldPolarBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
            ATOMIC_ADD(&fieldPolarBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
            ATOMIC_ADD(&fieldPolarBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
        }
    }

    // Second loop: tiles without exclusions.

#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (numTiles > maxTiles ? startTileIndex+warp*(mm_long)numTileIndices/totalWarps : warp*(mm_long)numTiles/totalWarps);
    int end = (int) (numTiles > maxTiles ? startTileIndex+(warp+1)*(mm_long)numTileIndices/totalWarps : (warp+1)*(mm_long)numTiles/totalWarps);
#else
    const unsigned int numTiles = numTileIndices;
    int pos = (int) (startTileIndex+warp*(mm_long)numTiles/totalWarps);
    int end = (int) (startTileIndex+(warp+1)*(mm_long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    LOCAL int atomIndices[THREAD_BLOCK_SIZE];
    LOCAL volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[LOCAL_ID] = -1;

    while (pos < end) {
        bool includeTile = true;
        int x, y;
#ifdef USE_CUTOFF
        x = tiles[pos];
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) {
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }
        SYNC_WARPS;
        while (skipTiles[tbx+TILE_SIZE-1] < pos) {
            SYNC_WARPS;
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                int2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[LOCAL_ID] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[LOCAL_ID] = end;
            skipBase += TILE_SIZE;
            currentSkipIndex = tbx;
            SYNC_WARPS;
        }
        while (skipTiles[currentSkipIndex] < pos)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;
            AtomData data = loadAtomData(atom1, posq, damping, atomInfo);
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[LOCAL_ID] = j;
            localData[LOCAL_ID] = loadAtomData(j, posq, damping, atomInfo);
            SYNC_WARPS;
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+tj].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    computeOneInteraction(&data, &localData[tbx+tj], delta, 0, fields);
                    data.field += fields[0];
                    localData[tbx+tj].field += fields[1];
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
            unsigned int offset = x*TILE_SIZE + tgx;
            ATOMIC_ADD(&fieldBuffers[offset], (mm_ulong) realToFixedPoint(data.field.x));
            ATOMIC_ADD(&fieldBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.y));
            ATOMIC_ADD(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.z));
            ATOMIC_ADD(&fieldPolarBuffers[offset], (mm_ulong) realToFixedPoint(data.field.x));
            ATOMIC_ADD(&fieldPolarBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.y));
            ATOMIC_ADD(&fieldPolarBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.z));
#ifdef USE_CUTOFF
            offset = atomIndices[LOCAL_ID];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            ATOMIC_ADD(&fieldBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
            ATOMIC_ADD(&fieldBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
            ATOMIC_ADD(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
            ATOMIC_ADD(&fieldPolarBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
            ATOMIC_ADD(&fieldPolarBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
            ATOMIC_ADD(&fieldPolarBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
        }
        pos++;
    }
}
