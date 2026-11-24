// WaterTrimerPIPForce: O-O neighbour list, trimer list and stream evaluation of the three-body polynomial (CUDA warp
// intrinsics).  diagnosticStatus: [0] neighbour overflow, [1] non-finite or unsafe value, [2] STREAM_CAPACITY exceeded.
inline DEVICE real3 wtpipDelta(real3 from, real3 to,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 delta = make_real4(to.x-from.x, to.y-from.y, to.z-from.z, 0);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    return make_real3(delta.x, delta.y, delta.z);
}

inline DEVICE void wtpipStoreForce(int atom, real3 force,
        GLOBAL mm_ulong* RESTRICT forceBuffers) {
    ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(force.x));
    ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
    ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
}

inline DEVICE void wtpipPairSwitch(real r, real* value, real* derivative) {
    if (r <= (real) SWITCH_ON) {
        *value = (real) 1;
        *derivative = (real) 0;
    }
    else if (r >= (real) SWITCH_CUT) {
        *value = (real) 0;
        *derivative = (real) 0;
    }
    else {
        real argument = (real) M_PI*(r-(real) SWITCH_ON)/((real) SWITCH_CUT-(real) SWITCH_ON);
        *value = (real) 0.5*((real) 1+cos(argument));
        *derivative = -(real) 0.5*(real) M_PI*sin(argument)/((real) SWITCH_CUT-(real) SWITCH_ON);
    }
}

inline DEVICE void wtpipSmoothRise(real r, real zero, real full,
        real* value, real* derivative) {
    if (r <= zero) {
        *value = (real) 0;
        *derivative = (real) 0;
    }
    else if (r >= full) {
        *value = (real) 1;
        *derivative = (real) 0;
    }
    else {
        real t = (r-zero)/(full-zero);
        *value = t*t*t*((real) 10+t*((real) -15+(real) 6*t));
        *derivative = (real) 30*t*t*(t-(real) 1)*(t-(real) 1)/(full-zero);
    }
}

inline DEVICE void wtpipSmoothWindow(real r, real lowerZero, real lowerFull,
        real upperFull, real upperZero, real* value, real* derivative) {
    if (r <= lowerZero || r >= upperZero) {
        *value = (real) 0;
        *derivative = (real) 0;
    }
    else if (r < lowerFull) {
        wtpipSmoothRise(r, lowerZero, lowerFull, value, derivative);
    }
    else if (r <= upperFull) {
        *value = (real) 1;
        *derivative = (real) 0;
    }
    else {
        real rise;
        wtpipSmoothRise(r, upperFull, upperZero, &rise, derivative);
        *value = (real) 1-rise;
        *derivative = -*derivative;
    }
}

KERNEL void invertAtomOrder(GLOBAL const int* RESTRICT atomIndex,
                            GLOBAL int* RESTRICT originalToCurrent) {
    for (int current = GLOBAL_ID; current < NUM_ATOMS; current += GLOBAL_SIZE)
        originalToCurrent[atomIndex[current]] = current;
}

// O neighbours inside the cutoff in water index order, one thread per water; the O positions are staged through shared
// memory in tiles.  Launched with NEIGHBOR_BLOCK threads per block, all threads of a block take part in the tile loads.
#define NEIGHBOR_BLOCK 128
#define NEIGHBOR_TILE 1024
KERNEL void findOxygenNeighbors(
        GLOBAL const int* RESTRICT originalToCurrent,
        GLOBAL const int4* RESTRICT waterAtoms,
        GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL int* RESTRICT neighborCounts,
        GLOBAL int* RESTRICT neighbors,
        GLOBAL int* RESTRICT diagnosticStatus,
        GLOBAL real4* RESTRICT oxygenPos) {
    LOCAL real3 tileO[NEIGHBOR_TILE];
    const int water = GLOBAL_ID;
    bool active = (water < NUM_WATERS);
    real3 posO = make_real3(0, 0, 0);
    if (active) {
        int4 original = waterAtoms[water];
        int registered[3] = {originalToCurrent[original.x], originalToCurrent[original.y], originalToCurrent[original.z]};
        bool coordinatesValid = true;
        for (int site = 0; site < 3; site++) {
            real4 position = posq[registered[site]];
            if (!isfinite(position.x) || !isfinite(position.y) || !isfinite(position.z))
                coordinatesValid = false;
        }
        if (!coordinatesValid) {
            ATOMIC_ADD(&diagnosticStatus[1], 1);
            neighborCounts[water] = 0;
            active = false;
        }
        else {
            posO = trimTo3(posq[registered[0]]);
            oxygenPos[water] = make_real4(posO.x, posO.y, posO.z, 0);
        }
    }
    int count = 0;
    bool overflowRecorded = false;
    for (int tileStart = 0; tileStart < NUM_WATERS; tileStart += NEIGHBOR_TILE) {
        int tileSize = min(NEIGHBOR_TILE, NUM_WATERS-tileStart);
        SYNC_THREADS;
        for (int i = LOCAL_ID; i < tileSize; i += LOCAL_SIZE)
            tileO[i] = trimTo3(posq[originalToCurrent[waterAtoms[tileStart+i].x]]);
        SYNC_THREADS;
        if (!active)
            continue;
        for (int i = 0; i < tileSize; i++) {
            int other = tileStart+i;
            if (other == water)
                continue;
            real3 delta = wtpipDelta(posO, tileO[i], periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
            real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            if (!isfinite(r2)) {
                ATOMIC_ADD(&diagnosticStatus[1], 1);
                continue;
            }
            if (r2 < (real) CUTOFF_SQUARED) {
                if (count < MAX_NEIGHBORS)
                    neighbors[water*MAX_NEIGHBORS+count] = other;
                else if (!overflowRecorded) {
                    ATOMIC_ADD(&diagnosticStatus[0], 1);
                    overflowRecorded = true;
                }
                count++;
            }
        }
    }
    if (active)
        neighborCounts[water] = (count < MAX_NEIGHBORS ? count : MAX_NEIGHBORS);
}

KERNEL void buildWaterTrimers(
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT neighborCounts,
        GLOBAL const int* RESTRICT neighbors,
        GLOBAL int* RESTRICT trimerCount,
        GLOBAL int4* RESTRICT trimers,
        GLOBAL const real4* RESTRICT oxygenPos) {
    // one thread per (centre, first neighbour J); O positions cached by findOxygenNeighbors
    for (int item = GLOBAL_ID; item < NUM_WATERS*MAX_NEIGHBORS; item += GLOBAL_SIZE) {
        int center = item/MAX_NEIGHBORS, a = item-center*MAX_NEIGHBORS;
        int count = neighborCounts[center];
        if (a >= count-1)
            continue;
        int waterJ = neighbors[center*MAX_NEIGHBORS+a];
        real3 posJ = trimTo3(oxygenPos[waterJ]);
        for (int b = a+1; b < count; b++) {
            int waterK = neighbors[center*MAX_NEIGHBORS+b];
            real3 deltaJK = wtpipDelta(posJ, trimTo3(oxygenPos[waterK]),
                    periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
            real r2JK = deltaJK.x*deltaJK.x + deltaJK.y*deltaJK.y + deltaJK.z*deltaJK.z;
            bool openChain = (r2JK >= (real) CUTOFF_SQUARED);
            bool uniqueClosedTriangle = (center < waterJ && center < waterK);
            if (openChain || uniqueClosedTriangle) {
                int index = ATOMIC_ADD(trimerCount, 1);
                if (index < MAX_TRIMERS)
                    trimers[index] = make_int4(center, waterJ, waterK, 0);
            }
        }
    }
}

// Stream evaluation (tables built by PIPStream.h, which documents the entry bit layout).  The polynomial is
// P = 1/2 y^T A y with y = (1, x~_i, x~_i x~_j) (x~ = x - centre, NUM_VARIABLES is the dummy variable of value 1) and
// A symmetric; each row of A is stored as groups of entries that share the first factor x~_i of their column,
//     z_row = (A y)_row = sum_i x~_i sum_j A_row,(i,j) x~_j .
// All lanes of a warp walk the same entry stream for 32 different trimers (lane = trimer), so control flow and
// coefficient loads are uniform and every lane only touches its own column of the shared arrays.  At the end of a row
// y_row = x~_p x~_q gives P += 1/2 y z and dP/dx~_p += z x~_q, dP/dx~_q += z x~_p.  The stream is cut into
// NUM_SEGMENTS row-aligned segments.  clusterVars: [batch of 32 trimers][variable, last = dummy 1][lane].
#define WTPIP_SV (NUM_VARIABLES+1)
#define WTPIP_END_GROUP (1u << 24)
#define WTPIP_END_ROW (1u << 25)
typedef STREAM_REAL sreal;

// imaged sites of one trimer (O, H1, H2 of A, B, C, all relative to O of A) and their atom indices
inline DEVICE void wtpipTrimerSites(int4 waterSet, GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent, GLOBAL const int4* RESTRICT waterAtoms,
        int* atoms, real3* sites) {
    int waters[3] = {waterSet.x, waterSet.y, waterSet.z};
    for (int w = 0; w < 3; w++) {
        int4 original = waterAtoms[waters[w]];
        atoms[3*w] = originalToCurrent[original.x];
        atoms[3*w+1] = originalToCurrent[original.y];
        atoms[3*w+2] = originalToCurrent[original.z];
    }
    real3 origin = trimTo3(posq[atoms[0]]);
    for (int w = 0; w < 3; w++) {
        real3 rawO = trimTo3(posq[atoms[3*w]]);
        real3 imagedO = origin;
        if (w != 0)
            imagedO = origin + wtpipDelta(origin, rawO,
                    periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        sites[3*w] = imagedO;
        sites[3*w+1] = imagedO + wtpipDelta(rawO, trimTo3(posq[atoms[3*w+1]]),
                periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        sites[3*w+2] = imagedO + wtpipDelta(rawO, trimTo3(posq[atoms[3*w+2]]),
                periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
    }
}

// one warp per trimer, lane = variable (lanes 0-3 also take variables 32-35): distances, variables
// x = exp(-k (10 r - d0)) in double, centred x - mu for the stream; r and x are kept for wtpipFinish (r < 0 = unusable)
KERNEL void wtpipPrepare(GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent,
        GLOBAL const int4* RESTRICT waterAtoms,
        GLOBAL const int* RESTRICT trimerCount,
        GLOBAL const int4* RESTRICT trimers,
        GLOBAL const int2* RESTRICT variablePairs,
        GLOBAL const double* RESTRICT variableK,
        GLOBAL const double* RESTRICT variableD0,
        GLOBAL const double* RESTRICT variableCenter,
        GLOBAL sreal* RESTRICT clusterVars,
        GLOBAL float* RESTRICT clusterR,
        GLOBAL float* RESTRICT clusterX,
        GLOBAL int* RESTRICT diagnosticStatus) {
    const int lane = LOCAL_ID%32;
    int count = *trimerCount;
    if (count > STREAM_CAPACITY) {
        if (GLOBAL_ID == 0)
            ATOMIC_ADD(&diagnosticStatus[2], 1);
        count = STREAM_CAPACITY;
    }
    int padded = (count+31)/32*32;
    for (int t = GLOBAL_ID/32; t < padded; t += GLOBAL_SIZE/32) {
        GLOBAL sreal* out = clusterVars + (size_t) (t/32)*WTPIP_SV*32 + (t%32);
        if (t >= count) {
            for (int v = lane; v < WTPIP_SV; v += 32)
                out[v*32] = (v < NUM_VARIABLES) ? (sreal) 0 : (sreal) 1;
            continue;
        }
        int atoms[9];
        real3 sites[9];
        wtpipTrimerSites(trimers[t], posq, periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, originalToCurrent, waterAtoms, atoms, sites);
        for (int v = lane; v < NUM_VARIABLES; v += 32) {
            int2 pair = variablePairs[v];
            double dx = (double) sites[pair.y].x-(double) sites[pair.x].x;
            double dy = (double) sites[pair.y].y-(double) sites[pair.x].y;
            double dz = (double) sites[pair.y].z-(double) sites[pair.x].z;
            double r = sqrt(dx*dx+dy*dy+dz*dz);
            double exponent = -variableK[v]*(10.0*r-variableD0[v]);
            bool ok = isfinite(r) && r > 1.0e-8 && isfinite(exponent) && exponent <= 80.0;
            double x = ok ? exp(exponent) : 0.0;
            ok = ok && isfinite(x);
            clusterR[(size_t) t*NUM_VARIABLES+v] = ok ? (float) r : -1.0f;
            clusterX[(size_t) t*NUM_VARIABLES+v] = ok ? (float) x : 0.0f;
            out[v*32] = ok ? (sreal) (x-variableCenter[v]) : (sreal) 0;
        }
        if (lane == 0)
            out[NUM_VARIABLES*32] = (sreal) 1;
    }
}

KERNEL void wtpipStream(GLOBAL const int* RESTRICT trimerCount,
        GLOBAL const sreal* RESTRICT clusterVars,
        GLOBAL const unsigned int* RESTRICT streamInfo,
        GLOBAL const sreal* RESTRICT streamCoef,
        GLOBAL const int4* RESTRICT segments,
        GLOBAL sreal* RESTRICT partialP,
        GLOBAL sreal* RESTRICT partialG,
        int includeForces) {
    // a block of STREAM_WARPS warps works on one segment for STREAM_WARPS batches of 32 clusters; the segment is
    // staged through shared memory in chunks of STREAM_CHUNK entries and walked 8 entries at a time, so that the
    // entry loads and the variable loads of 8 entries are in flight together (the sums stay sequential)
    LOCAL sreal xs[STREAM_WARPS][WTPIP_SV][32];
    LOCAL unsigned int sInfo[STREAM_CHUNK];
    LOCAL sreal sCoef[STREAM_CHUNK];
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    int count = min(*trimerCount, STREAM_CAPACITY);
    int numBatches = (count+31)/32;
    int numGroups = (numBatches+STREAM_WARPS-1)/STREAM_WARPS;
    int numItems = numGroups*NUM_SEGMENTS;
    for (int item = GROUP_ID; item < numItems; item += NUM_GROUPS) {
        int seg = item/numGroups;
        int batch = (item-seg*numGroups)*STREAM_WARPS+warp;
        bool active = (batch < numBatches);
        sreal gl[NUM_VARIABLES];
        for (int v = 0; v < NUM_VARIABLES; v++)
            gl[v] = (sreal) 0;
        if (active) {
            GLOBAL const sreal* in = clusterVars + (size_t) batch*WTPIP_SV*32 + lane;
            for (int v = 0; v < WTPIP_SV; v++)
                xs[warp][v][lane] = in[v*32];
        }
        int4 sg = segments[seg];
        unsigned int gi = (unsigned int) sg.z;
        sreal accIn = 0, accOut = 0, energy = 0;
        for (int base = sg.x; base < sg.y; base += STREAM_CHUNK) {
            int n = min(STREAM_CHUNK, sg.y-base);
            SYNC_THREADS;
            for (int i = LOCAL_ID; i < n; i += LOCAL_SIZE) {
                sInfo[i] = streamInfo[base+i];
                sCoef[i] = streamCoef[base+i];
            }
            SYNC_THREADS;
            if (!active)
                continue;
            int k = 0;
            for (; k+8 <= n; k += 8) {
                unsigned int inf[8];
                sreal cf[8], xv[8];
#pragma unroll
                for (int u = 0; u < 8; u++) {
                    inf[u] = sInfo[k+u];
                    cf[u] = sCoef[k+u];
                }
#pragma unroll
                for (int u = 0; u < 8; u++)
                    xv[u] = xs[warp][inf[u] & 63u][lane];
#pragma unroll
                for (int u = 0; u < 8; u++) {
                    accIn += cf[u]*xv[u];
                    if (inf[u] & WTPIP_END_GROUP) {
                        accOut += xs[warp][gi][lane]*accIn;
                        accIn = 0;
                        gi = (inf[u] >> 6) & 63u;
                        if (inf[u] & WTPIP_END_ROW) {
                            unsigned int p = (inf[u] >> 12) & 63u, q = (inf[u] >> 18) & 63u;
                            sreal xp = xs[warp][p][lane], xq = xs[warp][q][lane];
                            energy += xp*xq*accOut;
                            if (p < NUM_VARIABLES)
                                gl[p] += accOut*xq;
                            if (q < NUM_VARIABLES)
                                gl[q] += accOut*xp;
                            accOut = 0;
                        }
                    }
                }
            }
            for (; k < n; k++) {
                unsigned int info = sInfo[k];
                accIn += sCoef[k]*xs[warp][info & 63u][lane];
                if (info & WTPIP_END_GROUP) {
                    accOut += xs[warp][gi][lane]*accIn;
                    accIn = 0;
                    gi = (info >> 6) & 63u;
                    if (info & WTPIP_END_ROW) {
                        unsigned int p = (info >> 12) & 63u, q = (info >> 18) & 63u;
                        sreal xp = xs[warp][p][lane], xq = xs[warp][q][lane];
                        energy += xp*xq*accOut;
                        if (p < NUM_VARIABLES)
                            gl[p] += accOut*xq;
                        if (q < NUM_VARIABLES)
                            gl[q] += accOut*xp;
                        accOut = 0;
                    }
                }
            }
        }
        if (active) {
            partialP[(size_t) seg*STREAM_CAPACITY_PADDED + batch*32 + lane] = (sreal) 0.5*energy;
            if (includeForces) {
                GLOBAL sreal* g = partialG + ((size_t) seg*STREAM_BATCHES + batch)*NUM_VARIABLES*32 + lane;
                for (int v = 0; v < NUM_VARIABLES; v++)
                    g[v*32] = gl[v];
            }
        }
    }
}

// one warp per trimer, lane = variable: switch and guard products by warp reductions, dE/dr per variable,
// site forces summed in shared memory, lanes 0-8 store the atom forces
KERNEL void wtpipFinish(
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent,
        GLOBAL const int4* RESTRICT waterAtoms,
        GLOBAL const int* RESTRICT trimerCount,
        GLOBAL const int4* RESTRICT trimers,
        GLOBAL const int2* RESTRICT variablePairs,
        GLOBAL const double* RESTRICT variableK,
        GLOBAL const float* RESTRICT clusterR,
        GLOBAL const float* RESTRICT clusterX,
        GLOBAL const sreal* RESTRICT partialP,
        GLOBAL const sreal* RESTRICT partialG,
        GLOBAL int* RESTRICT diagnosticStatus,
        int includeForces,
        int includeEnergy) {
    LOCAL float sf[FINISH_WARPS][9][3];
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    const unsigned int all = 0xffffffffu;
    mixed threadEnergy = 0;
    int count = min(*trimerCount, STREAM_CAPACITY);
    for (int t = GLOBAL_ID/32; t < count; t += GLOBAL_SIZE/32) {
        float r[2] = {1.0f, 1.0f}, x[2] = {0.0f, 0.0f};
        bool ok = true;
        for (int j = 0; j < 2; j++) {
            int v = lane+32*j;
            if (v < NUM_VARIABLES) {
                r[j] = clusterR[(size_t) t*NUM_VARIABLES+v];
                x[j] = clusterX[(size_t) t*NUM_VARIABLES+v];
                if (!(r[j] > 0.0f))
                    ok = false;
            }
        }
        if (__any_sync(all, !ok)) {
            if (lane == 0)
                ATOMIC_ADD(&diagnosticStatus[1], 1);
            continue;
        }
        // pair switches of the O-O variables 33-35 (second variable of lanes 1-3)
        real sw = 1, dsw = 0;
        if (lane >= 1 && lane <= 3)
            wtpipPairSwitch((real) r[1], &sw, &dsw);
        real sAB = __shfl_sync(all, sw, 1), sAC = __shfl_sync(all, sw, 2), sBC = __shfl_sync(all, sw, 3);
        real dsAB = __shfl_sync(all, dsw, 1), dsAC = __shfl_sync(all, dsw, 2), dsBC = __shfl_sync(all, dsw, 3);
        real switchValue = sAB*sAC + sAB*sBC + sAC*sBC;
        if (switchValue == (real) 0)
            continue;
        real g[2] = {1, 1}, gd[2] = {0, 0};
        for (int j = 0; j < 2; j++) {
            int v = lane+32*j;
            if (v >= NUM_VARIABLES)
                continue;
            if (v >= 33)
                wtpipSmoothRise((real) r[j], (real) OO_SHORT_ZERO, (real) OO_SHORT_FULL, &g[j], &gd[j]);
            else if (v < 3)
                wtpipSmoothWindow((real) r[j], (real) HH_LOWER_ZERO, (real) HH_LOWER_FULL,
                        (real) HH_UPPER_FULL, (real) HH_UPPER_ZERO, &g[j], &gd[j]);
            else if (v < 9)
                wtpipSmoothWindow((real) r[j], (real) OH_LOWER_ZERO, (real) OH_LOWER_FULL,
                        (real) OH_UPPER_FULL, (real) OH_UPPER_ZERO, &g[j], &gd[j]);
            else if (v >= 21 && (real) OH_INTER_FULL > (real) 0)
                wtpipSmoothRise((real) r[j], (real) OH_INTER_ZERO, (real) OH_INTER_FULL, &g[j], &gd[j]);
            else if (v < 21 && (real) HH_INTER_FULL > (real) 0)
                wtpipSmoothRise((real) r[j], (real) HH_INTER_ZERO, (real) HH_INTER_FULL, &g[j], &gd[j]);
        }
        real guardValue = g[0]*g[1];
        for (int m = 16; m > 0; m >>= 1)
            guardValue *= __shfl_xor_sync(all, guardValue, m);
        if (guardValue == (real) 0)
            continue;
        const int batch = t/32, tl = t%32;
        mixed polynomial = 0;
        if (lane == 0)
            for (int seg = 0; seg < NUM_SEGMENTS; seg++)
                polynomial += (mixed) partialP[(size_t) seg*STREAM_CAPACITY_PADDED + t];
        polynomial = __shfl_sync(all, polynomial, 0);
        mixed envelope = (mixed) (guardValue*switchValue);
        mixed energy = envelope*polynomial;
        if (!isfinite(polynomial) || !isfinite(energy))
            ok = false;
        mixed dEdr[2] = {0, 0};
        if (includeForces)
            for (int j = 0; j < 2; j++) {
                int v = lane+32*j;
                if (v >= NUM_VARIABLES)
                    continue;
                mixed dPdx = 0;
                for (int seg = 0; seg < NUM_SEGMENTS; seg++)
                    dPdx += (mixed) partialG[((size_t) seg*STREAM_BATCHES + batch)*NUM_VARIABLES*32 + v*32 + tl];
                real dSwitch = 0;
                if (v == 33)
                    dSwitch = dsAB*(sAC+sBC);
                else if (v == 34)
                    dSwitch = dsAC*(sAB+sBC);
                else if (v == 35)
                    dSwitch = dsBC*(sAB+sAC);
                real dGuard = (gd[j] == (real) 0) ? (real) 0 : gd[j]*(guardValue/g[j]);   // guardValue != 0: no zero factor
                dEdr[j] = envelope*dPdx*(mixed) (-10*variableK[v]*x[j])+
                        polynomial*(mixed) (guardValue*dSwitch+switchValue*dGuard);
                if (!isfinite(dEdr[j]))
                    ok = false;
            }
        if (__any_sync(all, !ok)) {
            if (lane == 0)
                ATOMIC_ADD(&diagnosticStatus[1], 1);
            continue;
        }
        if (includeEnergy && lane == 0) {
            mixed accumulated = threadEnergy+energy;
            if (isfinite(accumulated))
                threadEnergy = accumulated;
        }
        if (!includeForces)
            continue;
        int atoms[9];
        real3 sites[9];
        wtpipTrimerSites(trimers[t], posq, periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, originalToCurrent, waterAtoms, atoms, sites);
        if (lane < 27)
            sf[warp][lane/3][lane%3] = 0.0f;
        SYNC_WARPS;
        for (int j = 0; j < 2; j++) {
            int v = lane+32*j;
            if (v >= NUM_VARIABLES)
                continue;
            int2 pair = variablePairs[v];
            real3 delta = sites[pair.y]-sites[pair.x];
            real scale = (real) (dEdr[j]/(mixed) r[j]);
            real3 force = delta*scale;
            if (!isfinite(force.x) || !isfinite(force.y) || !isfinite(force.z) ||
                    fabs(force.x) > (real) MAX_SAFE_FORCE || fabs(force.y) > (real) MAX_SAFE_FORCE ||
                    fabs(force.z) > (real) MAX_SAFE_FORCE)
                ok = false;
            atomicAdd(&sf[warp][pair.x][0], (float) force.x);
            atomicAdd(&sf[warp][pair.x][1], (float) force.y);
            atomicAdd(&sf[warp][pair.x][2], (float) force.z);
            atomicAdd(&sf[warp][pair.y][0], (float) -force.x);
            atomicAdd(&sf[warp][pair.y][1], (float) -force.y);
            atomicAdd(&sf[warp][pair.y][2], (float) -force.z);
        }
        SYNC_WARPS;
        real3 f = make_real3(0, 0, 0);
        if (lane < 9) {
            f = make_real3(sf[warp][lane][0], sf[warp][lane][1], sf[warp][lane][2]);
            if (!isfinite(f.x) || !isfinite(f.y) || !isfinite(f.z) || fabs(f.x) > (real) MAX_SAFE_FORCE ||
                    fabs(f.y) > (real) MAX_SAFE_FORCE || fabs(f.z) > (real) MAX_SAFE_FORCE)
                ok = false;
        }
        if (__any_sync(all, !ok)) {
            if (lane == 0)
                ATOMIC_ADD(&diagnosticStatus[1], 1);
        }
        else if (lane < 9)
            wtpipStoreForce(atoms[lane], f, forceBuffers);
        SYNC_WARPS;
    }
    if (includeEnergy) {
        mixed accumulatedEnergy = energyBuffer[GLOBAL_ID]+threadEnergy;
        if (isfinite(accumulatedEnergy))
            energyBuffer[GLOBAL_ID] = accumulatedEnergy;
        else
            ATOMIC_ADD(&diagnosticStatus[1], 1);
    }
}
