inline DEVICE real3 wdpipDelta(real3 from, real3 to,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 delta = make_real4(to.x-from.x, to.y-from.y, to.z-from.z, 0);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
    return make_real3(delta.x, delta.y, delta.z);
}

inline DEVICE real3 wdpipCross(real3 a, real3 b) {
    return make_real3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline DEVICE real wdpipNorm(real3 value) {
    return sqrt(value.x*value.x + value.y*value.y + value.z*value.z);
}

inline DEVICE void wdpipStoreForce(int atom, real3 force,
        GLOBAL mm_ulong* RESTRICT forceBuffers) {
    ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) realToFixedPoint(force.x));
    ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.y));
    ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force.z));
}

inline DEVICE int wdpipInt4Component(int4 value, int index) {
    if (index == 0)
        return value.x;
    if (index == 1)
        return value.y;
    if (index == 2)
        return value.z;
    return value.w;
}

inline DEVICE void wdpipPairSwitch(real r, real* value, real* derivative) {
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

inline DEVICE void wdpipSmoothRise(real r, real zero, real full,
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

inline DEVICE void wdpipSmoothWindow(real r, real lowerZero, real lowerFull,
        real upperFull, real upperZero, real* value, real* derivative) {
    if (r <= lowerZero || r >= upperZero) {
        *value = (real) 0;
        *derivative = (real) 0;
    }
    else if (r < lowerFull) {
        wdpipSmoothRise(r, lowerZero, lowerFull, value, derivative);
    }
    else if (r <= upperFull) {
        *value = (real) 1;
        *derivative = (real) 0;
    }
    else {
        real rise;
        wdpipSmoothRise(r, upperFull, upperZero, &rise, derivative);
        *value = (real) 1-rise;
        *derivative = -*derivative;
    }
}

KERNEL void invertAtomOrder(GLOBAL const int* RESTRICT atomIndex,
                            GLOBAL int* RESTRICT originalToCurrent) {
    for (int current = GLOBAL_ID; current < NUM_ATOMS; current += GLOBAL_SIZE)
        originalToCurrent[atomIndex[current]] = current;
}

// O neighbours inside the cutoff, one thread per water; the O positions are staged through shared memory in tiles
// (neighbour order = water index order, as before).  Launched with NEIGHBOR_BLOCK threads per block, all threads of
// a block take part in the tile loads.
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
        GLOBAL int* RESTRICT diagnosticStatus) {
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
        else
            posO = trimTo3(posq[registered[0]]);
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
            real3 delta = wdpipDelta(posO, tileO[i], periodicBoxSize, invPeriodicBoxSize,
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

KERNEL void buildWaterDimers(
        GLOBAL const int* RESTRICT neighborCounts,
        GLOBAL const int* RESTRICT neighbors,
        GLOBAL int* RESTRICT dimerCount,
        GLOBAL int2* RESTRICT dimers) {
    for (int center = GLOBAL_ID; center < NUM_WATERS; center += GLOBAL_SIZE) {
        int count = neighborCounts[center];
        for (int a = 0; a < count; a++) {
            int other = neighbors[center*MAX_NEIGHBORS+a];
            if (other <= center)
                continue;                       // each unordered pair once
            int index = ATOMIC_ADD(dimerCount, 1);
            if (index < MAX_DIMERS)
                dimers[index] = make_int2(center, other);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Stream mode (see waterTrimerPIP.cc): P = 1/2 y^T A y walked as a uniform entry stream, lane = dimer.
//   wdpipPrepare: one thread per dimer -> variables, batch-major / lane-minor
//   wdpipStream:  one warp per (segment, batch of 32 dimers) -> partial P and dP/dx
//   wdpipFinish:  one thread per dimer -> segment sums, envelope (switch x guards), forces, energy
// ------------------------------------------------------------------------------------------------
#define WDPIP_SV (NUM_VARIABLES+1)
#define WDPIP_END_GROUP (1u << 24)
#define WDPIP_END_ROW (1u << 25)
typedef STREAM_REAL sreal;

// imaged sites of one dimer (O, H1, H2, L1, L2 per water, relative to O of the first water), O-H vectors, atoms
inline DEVICE void wdpipDimerSites(int2 pair, GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent, GLOBAL const int4* RESTRICT waterAtoms,
        int* atoms, real3* sites, real3* ohVec) {
    int waterSet[2] = {pair.x, pair.y};
    real3 origin = trimTo3(posq[originalToCurrent[waterAtoms[waterSet[0]].x]]);
    for (int w = 0; w < 2; w++) {
        int4 original = waterAtoms[waterSet[w]];
        int registered[3] = {originalToCurrent[original.x], originalToCurrent[original.y],
                             originalToCurrent[original.z]};
        real3 rawO = trimTo3(posq[registered[0]]);
        real3 imagedO = origin;
        if (w != 0)
            imagedO = origin + wdpipDelta(origin, rawO,
                    periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        sites[5*w] = imagedO;
        atoms[3*w] = registered[0];
        for (int site = 1; site < 3; site++) {
            atoms[3*w+site] = registered[site];
            sites[5*w+site] = imagedO + wdpipDelta(rawO, trimTo3(posq[registered[site]]),
                    periodicBoxSize, invPeriodicBoxSize,
                    periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
        }
        ohVec[2*w] = sites[5*w+1]-imagedO;
        ohVec[2*w+1] = sites[5*w+2]-imagedO;
        real3 crossProduct = wdpipCross(ohVec[2*w], ohVec[2*w+1]);
        real3 mid = imagedO + (ohVec[2*w]+ohVec[2*w+1])*((real) (0.5*GAMMA_IN));
        sites[5*w+3] = mid + crossProduct*((real) GAMMA_OUT);
        sites[5*w+4] = mid - crossProduct*((real) GAMMA_OUT);
    }
}

// one warp per dimer, lane = variable: distances and variables in double (r and x kept for wdpipFinish, r < 0 = unusable)
KERNEL void wdpipPrepare(GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent,
        GLOBAL const int4* RESTRICT waterAtoms,
        GLOBAL const int* RESTRICT dimerCount,
        GLOBAL const int2* RESTRICT dimers,
        GLOBAL const int2* RESTRICT variablePairs,
        GLOBAL const double* RESTRICT variableK,
        GLOBAL const double* RESTRICT variableD0,
        GLOBAL const int* RESTRICT variableForm,
        GLOBAL sreal* RESTRICT clusterVars,
        GLOBAL float* RESTRICT clusterR,
        GLOBAL float* RESTRICT clusterX,
        GLOBAL int* RESTRICT diagnosticStatus) {
    const int lane = LOCAL_ID%32;
    int count = *dimerCount;
    if (count > STREAM_CAPACITY) {
        if (GLOBAL_ID == 0)
            ATOMIC_ADD(&diagnosticStatus[2], 1);
        count = STREAM_CAPACITY;
    }
    int padded = (count+31)/32*32;
    for (int t = GLOBAL_ID/32; t < padded; t += GLOBAL_SIZE/32) {
        GLOBAL sreal* out = clusterVars + (size_t) (t/32)*WDPIP_SV*32 + (t%32);
        if (t >= count) {
            for (int v = lane; v < WDPIP_SV; v += 32)
                out[v*32] = (v < NUM_VARIABLES) ? (sreal) 0 : (sreal) 1;
            continue;
        }
        int atoms[6];
        real3 sites[10], ohVec[4];
        wdpipDimerSites(dimers[t], posq, periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, originalToCurrent, waterAtoms, atoms, sites, ohVec);
        for (int v = lane; v < NUM_VARIABLES; v += 32) {
            int2 sp = variablePairs[v];
            double dx = (double) sites[sp.y].x-(double) sites[sp.x].x;
            double dy = (double) sites[sp.y].y-(double) sites[sp.x].y;
            double dz = (double) sites[sp.y].z-(double) sites[sp.x].z;
            double r = sqrt(dx*dx+dy*dy+dz*dz);
            double rAngstrom = 10.0*r;
            double exponent = -variableK[v]*(rAngstrom-variableD0[v]);
            bool ok = isfinite(r) && r > 1.0e-8 && isfinite(exponent) && exponent <= 80.0;
            double x = ok ? exp(exponent) : 0.0;
            if (ok && variableForm[v] == 1)
                x /= rAngstrom;
            ok = ok && isfinite(x);
            clusterR[(size_t) t*NUM_VARIABLES+v] = ok ? (float) r : -1.0f;
            clusterX[(size_t) t*NUM_VARIABLES+v] = ok ? (float) x : 0.0f;
            out[v*32] = ok ? (sreal) x : (sreal) 0;
        }
        if (lane == 0)
            out[NUM_VARIABLES*32] = (sreal) 1;
    }
}

KERNEL void wdpipStream(GLOBAL const int* RESTRICT dimerCount,
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
    LOCAL sreal xs[STREAM_WARPS][WDPIP_SV][32];
    LOCAL unsigned int sInfo[STREAM_CHUNK];
    LOCAL sreal sCoef[STREAM_CHUNK];
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    int count = min(*dimerCount, STREAM_CAPACITY);
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
            GLOBAL const sreal* in = clusterVars + (size_t) batch*WDPIP_SV*32 + lane;
            for (int v = 0; v < WDPIP_SV; v++)
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
                    if (inf[u] & WDPIP_END_GROUP) {
                        accOut += xs[warp][gi][lane]*accIn;
                        accIn = 0;
                        gi = (inf[u] >> 6) & 63u;
                        if (inf[u] & WDPIP_END_ROW) {
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
                if (info & WDPIP_END_GROUP) {
                    accOut += xs[warp][gi][lane]*accIn;
                    accIn = 0;
                    gi = (info >> 6) & 63u;
                    if (info & WDPIP_END_ROW) {
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

// one warp per dimer, lane = variable: switch (O-O, variable 14) and guard products by warp reductions, dE/dr per
// variable, site forces summed in shared memory, lone-pair forces redistributed onto O, H1, H2 (MB-pol monomer::grads)

KERNEL void wdpipFinish(
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mixed* RESTRICT energyBuffer,
        GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* RESTRICT originalToCurrent,
        GLOBAL const int4* RESTRICT waterAtoms,
        GLOBAL const int* RESTRICT dimerCount,
        GLOBAL const int2* RESTRICT dimers,
        GLOBAL const int2* RESTRICT variablePairs,
        GLOBAL const double* RESTRICT variableK,
        GLOBAL const int* RESTRICT variableForm,
        GLOBAL const float* RESTRICT clusterR,
        GLOBAL const float* RESTRICT clusterX,
        GLOBAL const sreal* RESTRICT partialP,
        GLOBAL const sreal* RESTRICT partialG,
        GLOBAL int* RESTRICT diagnosticStatus,
        int includeForces,
        int includeEnergy) {
    LOCAL float sf[FINISH_WARPS][10][3];
    const int lane = LOCAL_ID%32;
    const int warp = LOCAL_ID/32;
    const unsigned int all = 0xffffffffu;
    const int v = lane;
    const bool hasVar = (v < NUM_VARIABLES);
    mixed threadEnergy = 0;
    int count = min(*dimerCount, STREAM_CAPACITY);
    for (int t = GLOBAL_ID/32; t < count; t += GLOBAL_SIZE/32) {
        float r = 1.0f, x = 0.0f;
        bool ok = true;
        if (hasVar) {
            r = clusterR[(size_t) t*NUM_VARIABLES+v];
            x = clusterX[(size_t) t*NUM_VARIABLES+v];
            if (!(r > 0.0f))
                ok = false;
        }
        if (__any_sync(all, !ok)) {
            if (lane == 0)
                ATOMIC_ADD(&diagnosticStatus[1], 1);
            continue;
        }
        real sw = 1, dsw = 0;
        if (v == 14)
            wdpipPairSwitch((real) r, &sw, &dsw);
        real switchValue = __shfl_sync(all, sw, 14), dSwitch = __shfl_sync(all, dsw, 14);
        if (switchValue == (real) 0)
            continue;
        real g = 1, gd = 0;
        if (hasVar) {
            if (v == 14)
                wdpipSmoothRise((real) r, (real) OO_SHORT_ZERO, (real) OO_SHORT_FULL, &g, &gd);
            else if (v < 2)
                wdpipSmoothWindow((real) r, (real) HH_LOWER_ZERO, (real) HH_LOWER_FULL,
                        (real) HH_UPPER_FULL, (real) HH_UPPER_ZERO, &g, &gd);
            else if (v < 6)
                wdpipSmoothWindow((real) r, (real) OH_LOWER_ZERO, (real) OH_LOWER_FULL,
                        (real) OH_UPPER_FULL, (real) OH_UPPER_ZERO, &g, &gd);
            else if (v < 10 && (real) HH_INTER_FULL > (real) 0)
                wdpipSmoothRise((real) r, (real) HH_INTER_ZERO, (real) HH_INTER_FULL, &g, &gd);
            else if (v >= 10 && v < 14 && (real) OH_INTER_FULL > (real) 0)
                wdpipSmoothRise((real) r, (real) OH_INTER_ZERO, (real) OH_INTER_FULL, &g, &gd);
        }
        real guardValue = g;
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
        mixed dEdr = 0;
        if (includeForces && hasVar) {
            mixed dPdx = 0;
            for (int seg = 0; seg < NUM_SEGMENTS; seg++)
                dPdx += (mixed) partialG[((size_t) seg*STREAM_BATCHES + batch)*NUM_VARIABLES*32 + v*32 + tl];
            double rA = 10.0*(double) r;
            double dxdr = -variableK[v]*(double) x;
            if (variableForm[v] == 1)
                dxdr -= (double) x/rA;
            real dGuard = (gd == (real) 0) ? (real) 0 : gd*(guardValue/g);
            real dSw = (v == 14) ? dSwitch : (real) 0;
            dEdr = envelope*dPdx*(mixed) (10.0*dxdr)+polynomial*(mixed) (guardValue*dSw+switchValue*dGuard);
            if (!isfinite(dEdr))
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
        int atoms[6];
        real3 sites[10], ohVec[4];
        wdpipDimerSites(dimers[t], posq, periodicBoxSize, invPeriodicBoxSize,
                periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, originalToCurrent, waterAtoms, atoms, sites, ohVec);
        if (lane < 30)
            sf[warp][lane/3][lane%3] = 0.0f;
        SYNC_WARPS;
        if (hasVar) {
            int2 sp = variablePairs[v];
            real3 delta = sites[sp.y]-sites[sp.x];
            real scale = (real) (dEdr/(mixed) r);
            real3 force = delta*scale;
            if (!isfinite(force.x) || !isfinite(force.y) || !isfinite(force.z) ||
                    fabs(force.x) > (real) MAX_SAFE_FORCE || fabs(force.y) > (real) MAX_SAFE_FORCE ||
                    fabs(force.z) > (real) MAX_SAFE_FORCE)
                ok = false;
            atomicAdd(&sf[warp][sp.x][0], (float) force.x);
            atomicAdd(&sf[warp][sp.x][1], (float) force.y);
            atomicAdd(&sf[warp][sp.x][2], (float) force.z);
            atomicAdd(&sf[warp][sp.y][0], (float) -force.x);
            atomicAdd(&sf[warp][sp.y][1], (float) -force.y);
            atomicAdd(&sf[warp][sp.y][2], (float) -force.z);
        }
        SYNC_WARPS;
        // redistribute the extra-point forces onto O, H1 and H2 (lane w for water w)
        if (lane < 2) {
            int w = lane;
            real3 f1 = make_real3(sf[warp][5*w+3][0], sf[warp][5*w+3][1], sf[warp][5*w+3][2]);
            real3 f2 = make_real3(sf[warp][5*w+4][0], sf[warp][5*w+4][1], sf[warp][5*w+4][2]);
            real3 gm = f1-f2;
            real3 t1 = wdpipCross(ohVec[2*w+1], gm);
            real3 t2 = wdpipCross(ohVec[2*w], gm);
            real3 gsum = f1+f2;
            real3 inPlane = gsum*((real) (0.5*GAMMA_IN));
            real3 fh1 = inPlane + t1*((real) GAMMA_OUT);
            real3 fh2 = inPlane - t2*((real) GAMMA_OUT);
            real3 fo = gsum-(fh1+fh2);
            sf[warp][5*w][0] += (float) fo.x; sf[warp][5*w][1] += (float) fo.y; sf[warp][5*w][2] += (float) fo.z;
            sf[warp][5*w+1][0] += (float) fh1.x; sf[warp][5*w+1][1] += (float) fh1.y; sf[warp][5*w+1][2] += (float) fh1.z;
            sf[warp][5*w+2][0] += (float) fh2.x; sf[warp][5*w+2][1] += (float) fh2.y; sf[warp][5*w+2][2] += (float) fh2.z;
        }
        SYNC_WARPS;
        real3 f = make_real3(0, 0, 0);
        int site = 5*(lane/3)+lane%3;
        if (lane < 6) {
            f = make_real3(sf[warp][site][0], sf[warp][site][1], sf[warp][site][2]);
            if (!isfinite(f.x) || !isfinite(f.y) || !isfinite(f.z) || fabs(f.x) > (real) MAX_SAFE_FORCE ||
                    fabs(f.y) > (real) MAX_SAFE_FORCE || fabs(f.z) > (real) MAX_SAFE_FORCE)
                ok = false;
        }
        if (__any_sync(all, !ok)) {
            if (lane == 0)
                ATOMIC_ADD(&diagnosticStatus[1], 1);
        }
        else if (lane < 6)
            wdpipStoreForce(atoms[lane], f, forceBuffers);
        SYNC_WARPS;
    }
    if (includeEnergy) {
        mixed accumulated = energyBuffer[GLOBAL_ID]+threadEnergy;
        if (isfinite(accumulated))
            energyBuffer[GLOBAL_ID] = accumulated;
        else
            ATOMIC_ADD(&diagnosticStatus[1], 1);
    }
}
