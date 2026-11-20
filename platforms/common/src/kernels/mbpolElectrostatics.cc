/**
 * Real-space (or cluster) energy, force and site potential for MB-pol charges + induced dipoles.
 * Forked from OpenMM's AMOEBA electrostatics tile kernel; the pair math follows the MB-pol
 * reference implementation (calculatePmeDirectElectrostaticPairIxn / calculateElectrostaticPairIxn).
 */
#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos;
    real q;
    real3 force, inducedDipole, inducedDipolePolar;
    real potential;
    float damp;
    int type, water;
} AtomData;

inline DEVICE AtomData loadAtomData(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const real* RESTRICT inducedDipolePolar, GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
    AtomData data;
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.q = atomPosq.w;
    data.inducedDipole = make_real3(inducedDipole[3*atom], inducedDipole[3*atom+1], inducedDipole[3*atom+2]);
    data.inducedDipolePolar = make_real3(inducedDipolePolar[3*atom], inducedDipolePolar[3*atom+1], inducedDipolePolar[3*atom+2]);
    data.damp = damping[atom];
    int2 info = atomInfo[atom];
    data.type = info.x;
    data.water = info.y;
    data.force = make_real3(0);
    data.potential = 0;
    return data;
}

/**
 * Interaction of atom1 (i) with atom2 (k).  forceFactor is 1 when the pair is visited once (force and
 * potential are accumulated on both atoms) and 0.5 when the pair is visited twice in a diagonal tile
 * (only atom1 is updated, energy is halved).  Forces and potentials are unscaled (multiply by
 * ENERGY_SCALE_FACTOR afterwards).
 */
DEVICE void computeOneInteraction(AtomData* atom1, LOCAL_ARG AtomData* atom2, int pc, float forceFactor,
                                  mixed* energy, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real3 delta = atom2->pos - atom1->pos;
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    real r2 = dot(delta, delta);
#ifdef USE_CUTOFF
    if (r2 > CUTOFF_SQUARED)
        return;
#endif
    real r = SQRT(r2);
    real rr1 = RECIP(r);
    real rr3 = rr1/r2;
    real rr5 = 3*rr3/r2;
    real rr7 = 5*rr5/r2;
    real ci = atom1->q;
    real ck = atom2->q;
    real3 ui = atom1->inducedDipole;
    real3 upi = atom1->inducedDipolePolar;
    real3 uk = atom2->inducedDipole;
    real3 upk = atom2->inducedDipolePolar;
    real sci3 = dot(ui, delta);
    real sci4 = dot(uk, delta);
    real scip3 = dot(upi, delta);
    real scip4 = dot(upk, delta);
    real scip2 = dot(ui, upk) + dot(upi, uk);
    real gl0 = ci*ck;
    real gli1 = ck*sci3 - ci*sci4;
    real glip1 = ck*scip3 - ci*scip4;
    real damp = atom1->damp*atom2->damp;
    real s1CC, s3CC, s3CD, s5CD, s7CD, s5DD, s7DD, s3DD;
    tholeScales13(r, damp, THOLE_CC, THOLE_CC_QUARTER, &s1CC, &s3CC);
    classScales(pc, r, damp, atom1->type, atom2->type, &s3CD, &s5CD, &s7CD, &s3DD, &s5DD, &s7DD);
    bool sameWater = (pc == 1);     // no permanent charge-charge / charge-dipole inside the molecule
    real3 ftm2, ftm2i;
#ifdef USE_EWALD
    real ralpha = EWALD_ALPHA*r;
    real exp2a = EXP(-(ralpha*ralpha));
#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(ralpha);
#else
    const real t = RECIP(1.0f+0.3275911f*ralpha);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
#endif
    real bn0 = erfcAlphaR*rr1;
    real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
    real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
    alsq2n *= alsq2;
    real bn1 = (bn0+alsq2n*exp2a)/r2;
    alsq2n *= alsq2;
    real bn2 = (3*bn1+alsq2n*exp2a)/r2;
    alsq2n *= alsq2;
    real bn3 = (5*bn2+alsq2n*exp2a)/r2;
    if (sameWater) {
        // no permanent charge-charge or charge-induced dipole interaction inside a molecule:
        // remove the full real-space part (the reciprocal part cancels it)
        s1CC = 0;
        s3CC = 0;
        s3CD = 0;
        s5CD = 0;
    }
    real e = gl0*(bn0 - rr1*(1-s1CC));
    real ei = 0.5f*gli1*(bn1 - rr3*(1-s3CD));
    *energy += forceFactor*(e + ei);
    real potFactorC = bn0 - rr1*(1-s1CC);
    real potFactorD = bn1 - rr3*(1-s3CD);
    atom1->potential += ck*potFactorC - sci4*potFactorD;
    if (forceFactor == 1)
        atom2->potential += ci*potFactorC + sci3*potFactorD;
    real gf1 = bn1*gl0;
    real gfr1 = (1-s3CC)*rr3*gl0;
    real gfi1 = 0.5f*(bn2*(gli1 + glip1 + scip2) - bn3*(sci3*scip4 + scip3*sci4));
    real gfi2 = -ck*bn1;
    real gfi3 = ci*bn1;
    real gfri1 = 0.5f*(rr5*(gli1*(1-s5CD) + glip1*(1-s5CD) + scip2*(1-s5DD)) - rr7*(sci3*scip4 + scip3*sci4)*(1-s7DD));
    ftm2 = delta*(gf1 - gfr1);
    ftm2i = delta*gfi1 + 0.5f*((ui+upi)*gfi2 + (uk+upk)*gfi3 + (upi*sci4 + ui*scip4)*bn2 + (upk*sci3 + uk*scip3)*bn2);
    real3 ftm2ri = delta*gfri1 + (upi*sci4 + ui*scip4 + upk*sci3 + uk*scip3)*(0.5f*rr5*(1-s5DD))
                 + (uk+upk)*(ci*0.5f*rr3*(1-s3CD)) - (ui+upi)*(ck*0.5f*rr3*(1-s3CD));
    ftm2i -= ftm2ri;
#else
    if (sameWater) {
        gl0 = 0;
        gli1 = 0;
        glip1 = 0;
    }
    else {
        atom1->potential += ck*rr1*s1CC - sci4*rr3*s3CD;
        if (forceFactor == 1)
            atom2->potential += ci*rr1*s1CC + sci3*rr3*s3CD;
    }
    *energy += forceFactor*(rr1*gl0*s1CC + 0.5f*rr3*gli1*s3CD);
    real gf0 = rr3*gl0*s3CC;
    real gfi0 = 0.5f*rr5*(gli1 + glip1)*s5CD + 0.5f*rr5*scip2*s5DD - 0.5f*rr7*(sci3*scip4 + scip3*sci4)*s7DD;
    ftm2 = delta*gf0;
    ftm2i = delta*gfi0 + (upi*sci4 + ui*scip4 + upk*sci3 + uk*scip3)*(0.5f*rr5*s5DD);
    if (!sameWater)
        ftm2i += (uk+upk)*(ci*0.5f*rr3*s3CD) - (ui+upi)*(ck*0.5f*rr3*s3CD);
#endif
    // force on atom1 is -(ftm2+ftm2i); on atom2 +(ftm2+ftm2i)
    real3 f = ftm2 + ftm2i;
    atom1->force -= f;
    if (forceFactor == 1)
        atom2->force += f;
}

KERNEL void computeElectrostatics(
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT potentialBuffers, GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real4* RESTRICT posq, GLOBAL const int2* RESTRICT exclusionTiles, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const unsigned int* RESTRICT interactingAtoms,
#endif
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
#ifndef USE_CUTOFF
    real4 periodicBoxSize = make_real4(0, 0, 0, 0), invPeriodicBoxSize = make_real4(0, 0, 0, 0), periodicBoxVecX = make_real4(0, 0, 0, 0), periodicBoxVecY = make_real4(0, 0, 0, 0), periodicBoxVecZ = make_real4(0, 0, 0, 0);
#endif
    const unsigned int totalWarps = (GLOBAL_SIZE)/TILE_SIZE;
    const unsigned int warp = (GLOBAL_ID)/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
    mixed energy = 0;
    LOCAL AtomData localData[THREAD_BLOCK_SIZE];

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        unsigned int atom1 = x*TILE_SIZE + tgx;
        AtomData data = loadAtomData(atom1, posq, inducedDipole, inducedDipolePolar, damping, atomInfo);
        if (x == y) {
            localData[LOCAL_ID].pos = data.pos;
            localData[LOCAL_ID].q = data.q;
            localData[LOCAL_ID].inducedDipole = data.inducedDipole;
            localData[LOCAL_ID].inducedDipolePolar = data.inducedDipolePolar;
            localData[LOCAL_ID].damp = data.damp;
            localData[LOCAL_ID].type = data.type;
            localData[LOCAL_ID].water = data.water;
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+j], pairClass(data.water, localData[tbx+j].water), 0.5f, &energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
            }
#ifdef USE_EWALD
            if (atom1 < NUM_ATOMS)
                energy -= (EWALD_ALPHA/SQRT_PI)*data.q*data.q;   // Ewald self energy of the charge
#endif
            data.force *= ENERGY_SCALE_FACTOR;
            ATOMIC_ADD(&forceBuffers[atom1], (mm_ulong) realToFixedPoint(data.force.x));
            ATOMIC_ADD(&forceBuffers[atom1+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.y));
            ATOMIC_ADD(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.z));
            ATOMIC_ADD(&potentialBuffers[atom1], (mm_ulong) realToFixedPoint(data.potential));
            SYNC_WARPS;
        }
        else {
            unsigned int j = y*TILE_SIZE + tgx;
            localData[LOCAL_ID] = loadAtomData(j, posq, inducedDipole, inducedDipolePolar, damping, atomInfo);
            unsigned int tj = tgx;
            SYNC_WARPS;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+tj], pairClass(data.water, localData[tbx+tj].water), 1, &energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;
            data.force *= ENERGY_SCALE_FACTOR;
            localData[LOCAL_ID].force *= ENERGY_SCALE_FACTOR;
            unsigned int offset = x*TILE_SIZE + tgx;
            ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(data.force.x));
            ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.y));
            ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.z));
            ATOMIC_ADD(&potentialBuffers[offset], (mm_ulong) realToFixedPoint(data.potential));
            offset = y*TILE_SIZE + tgx;
            ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.x));
            ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.y));
            ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.z));
            ATOMIC_ADD(&potentialBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].potential));
            SYNC_WARPS;
        }
    }

#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return;
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
            AtomData data = loadAtomData(atom1, posq, inducedDipole, inducedDipolePolar, damping, atomInfo);
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[LOCAL_ID] = j;
            localData[LOCAL_ID] = loadAtomData(j, posq, inducedDipole, inducedDipolePolar, damping, atomInfo);
            SYNC_WARPS;
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+tj], 0, 1, &energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
            data.force *= ENERGY_SCALE_FACTOR;
            localData[LOCAL_ID].force *= ENERGY_SCALE_FACTOR;
            unsigned int offset = x*TILE_SIZE + tgx;
            ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(data.force.x));
            ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.y));
            ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.force.z));
            ATOMIC_ADD(&potentialBuffers[offset], (mm_ulong) realToFixedPoint(data.potential));
#ifdef USE_CUTOFF
            offset = atomIndices[LOCAL_ID];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            ATOMIC_ADD(&forceBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.x));
            ATOMIC_ADD(&forceBuffers[offset+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.y));
            ATOMIC_ADD(&forceBuffers[offset+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.z));
            ATOMIC_ADD(&potentialBuffers[offset], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].potential));
        }
        pos++;
    }
    energyBuffer[GLOBAL_ID] += energy*ENERGY_SCALE_FACTOR;
}
