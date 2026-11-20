/**
 * Field at every site due to the induced dipoles (mutual induction), plus the DIIS solver kernels.
 * Forked from OpenMM's AMOEBA multipoleInducedField kernel; TTM4-F damping with separate
 * inter- and intra-molecular Thole parameters.
 */
#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos;
    real3 field, fieldPolar, inducedDipole, inducedDipolePolar;
    float damp;
    int type, water;
} AtomData;

inline DEVICE AtomData loadAtomData(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const real* RESTRICT inducedDipolePolar, GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
    AtomData data;
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.inducedDipole = make_real3(inducedDipole[3*atom], inducedDipole[3*atom+1], inducedDipole[3*atom+2]);
    data.inducedDipolePolar = make_real3(inducedDipolePolar[3*atom], inducedDipolePolar[3*atom+1], inducedDipolePolar[3*atom+2]);
    data.damp = damping[atom];
    int2 info = atomInfo[atom];
    data.type = info.x;
    data.water = info.y;
    data.field = make_real3(0);
    data.fieldPolar = make_real3(0);
    return data;
}

inline DEVICE void saveAtomData(int index, AtomData data, GLOBAL mm_ulong* RESTRICT field, GLOBAL mm_ulong* RESTRICT fieldPolar) {
    ATOMIC_ADD(&field[index], (mm_ulong) realToFixedPoint(data.field.x));
    ATOMIC_ADD(&field[index+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.y));
    ATOMIC_ADD(&field[index+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.field.z));
    ATOMIC_ADD(&fieldPolar[index], (mm_ulong) realToFixedPoint(data.fieldPolar.x));
    ATOMIC_ADD(&fieldPolar[index+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.fieldPolar.y));
    ATOMIC_ADD(&fieldPolar[index+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data.fieldPolar.z));
}

#ifdef USE_EWALD
DEVICE void computeOneInteraction(AtomData* atom1, LOCAL_ARG AtomData* atom2, real3 deltaR, bool isSelfInteraction, int pc) {
    if (isSelfInteraction)
        return;
    real scale1, scale2;
    real r2 = dot(deltaR, deltaR);
    if (r2 < CUTOFF_SQUARED) {
        real rI = RSQRT(r2);
        real r = RECIP(rI);
        real rI2 = rI*rI;
        real ralpha = EWALD_ALPHA*r;
        real exp2a = EXP(-(ralpha*ralpha));
#ifdef USE_DOUBLE_PRECISION
        const real erfcAlphaR = erfc(ralpha);
#else
        const real t = RECIP(1.0f+0.3275911f*ralpha);
        const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
#endif
        real bn0 = erfcAlphaR*rI;
        real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
        real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
        alsq2n *= alsq2;
        real bn1 = (bn0+alsq2n*exp2a)*rI2;
        alsq2n *= alsq2;
        real bn2 = (3*bn1+alsq2n*exp2a)*rI2;
        real s3, s5, s7, t3, t5, t7;
        classScales(pc, r, atom1->damp*atom2->damp, atom1->type, atom2->type, &t3, &t5, &t7, &s3, &s5, &s7);
        real r3 = r*r2;
        real r5 = r3*r2;
        real rr3 = (1-s3)/r3;
        real rr5 = 3*(1-s5)/r5;
        scale1 = rr3 - bn1;
        scale2 = bn2 - rr5;
    }
    else {
        scale1 = 0;
        scale2 = 0;
    }
    real dDotDelta = scale2*dot(deltaR, atom2->inducedDipole);
    atom1->field += scale1*atom2->inducedDipole + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom2->inducedDipolePolar);
    atom1->fieldPolar += scale1*atom2->inducedDipolePolar + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom1->inducedDipole);
    atom2->field += scale1*atom1->inducedDipole + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom1->inducedDipolePolar);
    atom2->fieldPolar += scale1*atom1->inducedDipolePolar + dDotDelta*deltaR;
}
#else
DEVICE void computeOneInteraction(AtomData* atom1, LOCAL_ARG AtomData* atom2, real3 deltaR, bool isSelfInteraction, int pc) {
    if (isSelfInteraction)
        return;
    real r2 = dot(deltaR, deltaR);
    real rI = RSQRT(r2);
    real r = RECIP(rI);
    real r2I = rI*rI;
    real s3, s5, s7, t3, t5, t7;
    classScales(pc, r, atom1->damp*atom2->damp, atom1->type, atom2->type, &t3, &t5, &t7, &s3, &s5, &s7);
    real rr3 = -s3*rI*r2I;
    real rr5 = 3*s5*rI*r2I*r2I;
    real dDotDelta = rr5*dot(deltaR, atom2->inducedDipole);
    atom1->field += rr3*atom2->inducedDipole + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom2->inducedDipolePolar);
    atom1->fieldPolar += rr3*atom2->inducedDipolePolar + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom1->inducedDipole);
    atom2->field += rr3*atom1->inducedDipole + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom1->inducedDipolePolar);
    atom2->fieldPolar += rr3*atom1->inducedDipolePolar + dDotDelta*deltaR;
}
#endif

KERNEL void computeInducedField(
        GLOBAL mm_ulong* RESTRICT field, GLOBAL mm_ulong* RESTRICT fieldPolar, GLOBAL const real4* RESTRICT posq, GLOBAL const int2* RESTRICT exclusionTiles,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter, GLOBAL const unsigned int* RESTRICT interactingAtoms,
#endif
        GLOBAL const float* RESTRICT damping, GLOBAL const int2* RESTRICT atomInfo) {
    const unsigned int totalWarps = (GLOBAL_SIZE)/TILE_SIZE;
    const unsigned int warp = (GLOBAL_ID)/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
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
            localData[LOCAL_ID].inducedDipole = data.inducedDipole;
            localData[LOCAL_ID].inducedDipolePolar = data.inducedDipolePolar;
            localData[LOCAL_ID].damp = data.damp;
            localData[LOCAL_ID].type = data.type;
            localData[LOCAL_ID].water = data.water;
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+j].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+j], delta, atom1 == atom2, pairClass(data.water, localData[tbx+j].water));
            }
            SYNC_WARPS;
        }
        else {
            localData[LOCAL_ID] = loadAtomData(y*TILE_SIZE+tgx, posq, inducedDipole, inducedDipolePolar, damping, atomInfo);
            unsigned int tj = tgx;
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+tj], delta, false, pairClass(data.water, localData[tbx+tj].water));
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;
        }
        unsigned int offset = x*TILE_SIZE + tgx;
        saveAtomData(offset, data, field, fieldPolar);
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            saveAtomData(offset, localData[LOCAL_ID], field, fieldPolar);
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
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(&data, &localData[tbx+tj], delta, false, 0);
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
            unsigned int offset = x*TILE_SIZE + tgx;
            saveAtomData(offset, data, field, fieldPolar);
#ifdef USE_CUTOFF
            offset = atomIndices[LOCAL_ID];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            saveAtomData(offset, localData[LOCAL_ID], field, fieldPolar);
        }
        pos++;
    }
}

KERNEL void recordInducedDipolesForDIIS(GLOBAL const mm_long* RESTRICT fixedField, GLOBAL const mm_long* RESTRICT fixedFieldPolar,
        GLOBAL const float* RESTRICT polarizability, GLOBAL float2* RESTRICT errors, GLOBAL real* RESTRICT prevErrors, GLOBAL real* RESTRICT matrix,
        GLOBAL const mm_long* RESTRICT inducedField, GLOBAL const mm_long* RESTRICT inducedFieldPolar,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        GLOBAL real* RESTRICT prevDipoles, GLOBAL real* RESTRICT prevDipolesPolar, int iteration) {
    LOCAL real2 buffer[64];
    const real fieldScale = 1/(real) 0x100000000;
    real sumErrors = 0;
    real sumPolarErrors = 0;
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real scale = polarizability[atom];
        if (iteration >= MAX_PREV_DIIS_DIPOLES) {
            for (int i = 1; i < MAX_PREV_DIIS_DIPOLES; i++) {
                int index1 = atom+(i-1)*NUM_ATOMS;
                int index2 = atom+i*NUM_ATOMS;
                for (int j = 0; j < 3; j++) {
                    prevDipoles[3*index1+j] = prevDipoles[3*index2+j];
                    prevDipolesPolar[3*index1+j] = prevDipolesPolar[3*index2+j];
                    prevErrors[3*index1+j] = prevErrors[3*index2+j];
                }
            }
        }
        real3 oldDipole = make_real3(inducedDipole[3*atom], inducedDipole[3*atom+1], inducedDipole[3*atom+2]);
        real3 oldDipolePolar = make_real3(inducedDipolePolar[3*atom], inducedDipolePolar[3*atom+1], inducedDipolePolar[3*atom+2]);
        real3 fixed = make_real3(fixedField[atom], fixedField[atom+PADDED_NUM_ATOMS], fixedField[atom+2*PADDED_NUM_ATOMS])*fieldScale;
        real3 fixedPolar = make_real3(fixedFieldPolar[atom], fixedFieldPolar[atom+PADDED_NUM_ATOMS], fixedFieldPolar[atom+2*PADDED_NUM_ATOMS])*fieldScale;
        real3 induced = make_real3(inducedField[atom], inducedField[atom+PADDED_NUM_ATOMS], inducedField[atom+2*PADDED_NUM_ATOMS])*fieldScale;
        real3 inducedPolar = make_real3(inducedFieldPolar[atom], inducedFieldPolar[atom+PADDED_NUM_ATOMS], inducedFieldPolar[atom+2*PADDED_NUM_ATOMS])*fieldScale;
        real3 newDipole = scale*(fixed+induced);
        real3 newDipolePolar = scale*(fixedPolar+inducedPolar);
        int storePrevIndex = atom+min(iteration, MAX_PREV_DIIS_DIPOLES-1)*NUM_ATOMS;
        prevDipoles[3*storePrevIndex] = newDipole.x;
        prevDipoles[3*storePrevIndex+1] = newDipole.y;
        prevDipoles[3*storePrevIndex+2] = newDipole.z;
        prevDipolesPolar[3*storePrevIndex] = newDipolePolar.x;
        prevDipolesPolar[3*storePrevIndex+1] = newDipolePolar.y;
        prevDipolesPolar[3*storePrevIndex+2] = newDipolePolar.z;
        prevErrors[3*storePrevIndex] = newDipole.x-oldDipole.x;
        prevErrors[3*storePrevIndex+1] = newDipole.y-oldDipole.y;
        prevErrors[3*storePrevIndex+2] = newDipole.z-oldDipole.z;
        real3 errors = (newDipole-oldDipole)*(newDipole-oldDipole);
        real3 errorsPolar = (newDipolePolar-oldDipolePolar)*(newDipolePolar-oldDipolePolar);
        sumErrors += errors.x + errors.y + errors.z;
        sumPolarErrors += errorsPolar.x + errorsPolar.y + errorsPolar.z;
    }
    buffer[LOCAL_ID] = make_real2(sumErrors, sumPolarErrors);
    SYNC_THREADS;
    for (int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
        if (LOCAL_ID+offset < LOCAL_SIZE && (LOCAL_ID&(2*offset-1)) == 0) {
            buffer[LOCAL_ID].x += buffer[LOCAL_ID+offset].x;
            buffer[LOCAL_ID].y += buffer[LOCAL_ID+offset].y;
        }
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        errors[GROUP_ID] = make_float2((float) buffer[0].x, (float) buffer[0].y);
    if (iteration >= MAX_PREV_DIIS_DIPOLES && GROUP_ID == 0) {
        for (int i = 0; i < MAX_PREV_DIIS_DIPOLES-1; i++) {
            if (LOCAL_ID < MAX_PREV_DIIS_DIPOLES-1)
                matrix[LOCAL_ID+i*MAX_PREV_DIIS_DIPOLES] = matrix[(LOCAL_ID+1)+(i+1)*MAX_PREV_DIIS_DIPOLES];
            SYNC_THREADS;
        }
    }
}

KERNEL void computeDIISMatrix(GLOBAL real* RESTRICT prevErrors, int iteration, GLOBAL real* RESTRICT matrix) {
    LOCAL real sumBuffer[512];
    int j = min(iteration, MAX_PREV_DIIS_DIPOLES-1);
    for (int i = GROUP_ID; i <= j; i += NUM_GROUPS) {
        real sum = 0;
        for (int index = LOCAL_ID; index < 3*NUM_ATOMS; index += LOCAL_SIZE)
            sum += prevErrors[index+i*3*NUM_ATOMS]*prevErrors[index+j*3*NUM_ATOMS];
        sumBuffer[LOCAL_ID] = sum;
        SYNC_THREADS;
        for (int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
            if (LOCAL_ID+offset < LOCAL_SIZE && (LOCAL_ID&(2*offset-1)) == 0)
                sumBuffer[LOCAL_ID] += sumBuffer[LOCAL_ID+offset];
            SYNC_THREADS;
        }
        if (LOCAL_ID == 0) {
            matrix[i+MAX_PREV_DIIS_DIPOLES*j] = sumBuffer[0];
            if (i != j)
                matrix[j+MAX_PREV_DIIS_DIPOLES*i] = sumBuffer[0];
        }
    }
}

KERNEL void solveDIISMatrix(int iteration, GLOBAL const real* RESTRICT matrix, GLOBAL float* RESTRICT coefficients) {
    LOCAL real b[MAX_PREV_DIIS_DIPOLES+1][MAX_PREV_DIIS_DIPOLES+1];
    LOCAL real piv[MAX_PREV_DIIS_DIPOLES+1];
    LOCAL real x[MAX_PREV_DIIS_DIPOLES+1];
    if (iteration == 0) {
        if (LOCAL_ID == 0)
            coefficients[0] = 1;
        return;
    }
    int numPrev = min(iteration+1, MAX_PREV_DIIS_DIPOLES);
    int rank = numPrev+1;
    for (int index = LOCAL_ID; index < numPrev*numPrev; index += LOCAL_SIZE) {
        int i = index/numPrev;
        int j = index-i*numPrev;
        b[i+1][j+1] = matrix[i*MAX_PREV_DIIS_DIPOLES+j];
    }
    for (int i = LOCAL_ID; i < rank; i += LOCAL_SIZE) {
        b[i][0] = -1;
        piv[i] = i;
    }
    SYNC_THREADS;
    if (LOCAL_ID == 0) {
        real mean = 0;
        for (int i = 0; i < numPrev; i++)
            for (int j = 0; j < numPrev; j++)
                mean += fabs(b[i+1][j+1]);
        mean /= numPrev*numPrev;
        b[0][0] = 0;
        for (int i = 1; i < rank; i++)
            b[0][i] = -mean;
        int pivsign = 1;
        for (int j = 0; j < rank; j++) {
            for (int i = 0; i < rank; i++) {
                int kmax = min(i, j);
                real s = 0;
                for (int k = 0; k < kmax; k++)
                    s += b[i][k] * b[k][j];
                b[i][j] -= s;
            }
            int p = j;
            for (int i = j+1; i < rank; i++)
                if (fabs(b[i][j]) > fabs(b[p][j]))
                    p = i;
            if (p != j) {
                int k = 0;
                for (k = 0; k < rank; k++) {
                    real t = b[p][k];
                    b[p][k] = b[j][k];
                    b[j][k] = t;
                }
                k = piv[p];
                piv[p] = piv[j];
                piv[j] = k;
                pivsign = -pivsign;
            }
            if ((j < rank) && (b[j][j] != 0))
                for (int i = j+1; i < rank; i++)
                    b[i][j] /= b[j][j];
        }
        for (int i = 0; i < rank; i++)
            if (b[i][i] == 0) {
                for (int j = 0; j < rank-1; j++)
                    coefficients[j] = 0;
                coefficients[rank-1] = 1;
                return;
            }
        for (int i = 0; i < rank; i++)
            x[i] = (piv[i] == 0 ? -1 : 0);
        for (int k = 0; k < rank; k++)
            for (int i = k+1; i < rank; i++)
                x[i] -= x[k] * b[i][k];
        for (int k = rank-1; k >= 0; k--) {
            x[k] /= b[k][k];
            for (int i = 0; i < k; i++)
                x[i] -= x[k] * b[i][k];
        }
        real lastCoeff = 1;
        for (int i = 0; i < rank-1; i++) {
            real c = x[i+1]*mean;
            coefficients[i] = c;
            lastCoeff -= c;
        }
        coefficients[rank-1] = lastCoeff;
    }
}

KERNEL void updateInducedFieldByDIIS(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT inducedDipolePolar,
        GLOBAL const real* RESTRICT prevDipoles, GLOBAL const real* RESTRICT prevDipolesPolar, GLOBAL const float* RESTRICT coefficients, int numPrev) {
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        real sum = 0;
        real sumPolar = 0;
        for (int i = 0; i < numPrev; i++) {
            sum += coefficients[i]*prevDipoles[i*3*NUM_ATOMS+index];
            sumPolar += coefficients[i]*prevDipolesPolar[i*3*NUM_ATOMS+index];
        }
        inducedDipole[index] = sum;
        inducedDipolePolar[index] = sumPolar;
    }
}

/**
 * Successive over-relaxation step (solver without DIIS): mu <- mu + omega (mu_Jacobi - mu), where mu_Jacobi =
 * alpha (E_fixed + E_induced) was recorded in slot 0 of the history by recordInducedDipolesForDIIS.
 */
KERNEL void updateInducedDipolesBySOR(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT inducedDipolePolar,
        GLOBAL const real* RESTRICT prevDipoles, GLOBAL const real* RESTRICT prevDipolesPolar, float omega) {
    for (int index = GLOBAL_ID; index < 3*NUM_ATOMS; index += GLOBAL_SIZE) {
        inducedDipole[index] += omega*(prevDipoles[index]-inducedDipole[index]);
        inducedDipolePolar[index] += omega*(prevDipolesPolar[index]-inducedDipolePolar[index]);
    }
}
