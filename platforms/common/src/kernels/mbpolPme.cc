/**
 * Reciprocal-space PME for the MB-pol charges and induced dipoles, forked from OpenMM's AMOEBA multipolePme kernel:
 * spreading onto the grid, convolution, and the potential, field, force and energy at each site.
 */
#define ARRAY(x,y) array[(x)-1+((y)-1)*PME_ORDER]

/**
 * Calculate the spline coefficients for a single atom along a single axis.
 */
DEVICE void computeBSplinePoint(real4* thetai, real w, real* array) {
    // initialization to get to 2nd order recursion

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1 - w;

    // perform one pass to get to 3rd order recursion

    ARRAY(3,3) = 0.5f * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5f * ((1+w)*ARRAY(2,1)+(2-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5f * (1-w) * ARRAY(2,1);

    // compute standard B-spline recursion to desired order

    for (int i = 4; i <= PME_ORDER; i++)
    {
        int k = i - 1;
        real denom = RECIP((real) k);
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1-w) * ARRAY(k,1);
    }

    // get coefficients for the B-spline first derivative

    int k = PME_ORDER - 1;
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline second derivative

    k = PME_ORDER - 2;
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline third derivative

    k = PME_ORDER - 3;
    ARRAY(k,PME_ORDER-2) = ARRAY(k,PME_ORDER-3);
    for (int i = PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // thetai[i-1] = (value, 1st, 2nd, 3rd derivative)

    for (int i = 1; i <= PME_ORDER; i++)
        thetai[i-1] = make_real4(ARRAY(PME_ORDER,i), ARRAY(PME_ORDER-1,i), ARRAY(PME_ORDER-2,i), ARRAY(PME_ORDER-3,i));
}

KERNEL void gridSpreadFixedMultipoles(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    // We have shared memory to spare, and putting the workspace array there reduces the load on L2 cache.
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);
        real atomCharge = pos.w;

        // It's faster to compute the thetas here than load them from global memory.

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = remainder(GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_X);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = remainder(GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Y);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = remainder(GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Z);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Spread the charge from this atom onto each grid point.

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = igrid1+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real4 t = theta1[ix];

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = igrid2+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real4 u = theta2[iy];
                real term = atomCharge*t.x*u.x;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real4 v = theta3[iz];
                    real add = term*v.x;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add));
#else
                    ATOMIC_ADD(&pmeGrid[index].x, add);
#endif
                }
            }
        }
    }
}

KERNEL void gridSpreadInducedDipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const real* RESTRICT inducedDipolePolar,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    LOCAL real cartToFrac[3][3];
    if (LOCAL_ID == 0) {
        cartToFrac[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        cartToFrac[0][1] = GRID_SIZE_X*recipBoxVecY.x;
        cartToFrac[0][2] = GRID_SIZE_X*recipBoxVecZ.x;
        cartToFrac[1][0] = GRID_SIZE_Y*recipBoxVecX.y;
        cartToFrac[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        cartToFrac[1][2] = GRID_SIZE_Y*recipBoxVecZ.y;
        cartToFrac[2][0] = GRID_SIZE_Z*recipBoxVecX.z;
        cartToFrac[2][1] = GRID_SIZE_Z*recipBoxVecY.z;
        cartToFrac[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);
        real3 cinducedDipole = make_real3(inducedDipole[3*m], inducedDipole[3*m+1], inducedDipole[3*m+2]);
        real3 finducedDipole = make_real3(cinducedDipole.x*cartToFrac[0][0] + cinducedDipole.y*cartToFrac[0][1] + cinducedDipole.z*cartToFrac[0][2],
                                          cinducedDipole.x*cartToFrac[1][0] + cinducedDipole.y*cartToFrac[1][1] + cinducedDipole.z*cartToFrac[1][2],
                                          cinducedDipole.x*cartToFrac[2][0] + cinducedDipole.y*cartToFrac[2][1] + cinducedDipole.z*cartToFrac[2][2]);
        real3 cinducedDipolePolar = make_real3(inducedDipolePolar[3*m], inducedDipolePolar[3*m+1], inducedDipolePolar[3*m+2]);
        real3 finducedDipolePolar = make_real3(cinducedDipolePolar.x*cartToFrac[0][0] + cinducedDipolePolar.y*cartToFrac[0][1] + cinducedDipolePolar.z*cartToFrac[0][2],
                                               cinducedDipolePolar.x*cartToFrac[1][0] + cinducedDipolePolar.y*cartToFrac[1][1] + cinducedDipolePolar.z*cartToFrac[1][2],
                                               cinducedDipolePolar.x*cartToFrac[2][0] + cinducedDipolePolar.y*cartToFrac[2][1] + cinducedDipolePolar.z*cartToFrac[2][2]);

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = remainder(GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_X);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = remainder(GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Y);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = remainder(GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Z);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = igrid1+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real4 t = theta1[ix];

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = igrid2+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real4 u = theta2[iy];
                real term01 = finducedDipole.y*t.x*u.y + finducedDipole.x*t.y*u.x;
                real term11 = finducedDipole.z*t.x*u.x;
                real term02 = finducedDipolePolar.y*t.x*u.y + finducedDipolePolar.x*t.y*u.x;
                real term12 = finducedDipolePolar.z*t.x*u.x;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real4 v = theta3[iz];

                    real add1 = term01*v.x + term11*v.y;
                    real add2 = term02*v.x + term12*v.y;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add1));
                    ATOMIC_ADD(&pmeGrid[2*index+1], (mm_ulong) realToFixedPoint(add2));
#else
                    ATOMIC_ADD(&pmeGrid[index].x, add1);
                    ATOMIC_ADD(&pmeGrid[index].y, add2);
#endif
                }
            }
        }
    }
}

/**
 * In double precision, we have to use fixed point to accumulate the grid values, so convert them to floating point.
 */
KERNEL void finishSpreadCharge(GLOBAL const mm_long* RESTRICT pmeGridLong, GLOBAL real* RESTRICT pmeGrid) {
    const unsigned int gridSize = 2*GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE)
        pmeGrid[index] = scale*pmeGridLong[index];
}

KERNEL void reciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid, GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY, GLOBAL const real* RESTRICT pmeBsplineModuliZ, real4 periodicBoxSize,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real expFactor = M_PI*M_PI/(EWALD_ALPHA*EWALD_ALPHA);
    real scaleFactor = RECIP(M_PI*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        int kx = index/(GRID_SIZE_Y*GRID_SIZE_Z);
        int remainder = index-kx*GRID_SIZE_Y*GRID_SIZE_Z;
        int ky = remainder/GRID_SIZE_Z;
        int kz = remainder-ky*GRID_SIZE_Z;
        if (kx == 0 && ky == 0 && kz == 0) {
            pmeGrid[index] = make_real2(0, 0);
            continue;
        }
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real denom = m2*bx*by*bz;
        real eterm = scaleFactor*EXP(-expFactor*m2)/denom;
        pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
    }
}

KERNEL void computeFixedPotentialFromGrid(GLOBAL const real2* RESTRICT pmeGrid, GLOBAL real* RESTRICT phi, GLOBAL mm_long* RESTRICT fieldBuffers,
        GLOBAL mm_long* RESTRICT fieldPolarBuffers, GLOBAL const real4* RESTRICT posq, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = remainder(GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_X);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = remainder(GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Y);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = remainder(GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Z);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        real tuv000 = 0;
        real tuv001 = 0;
        real tuv010 = 0;
        real tuv100 = 0;
        real tuv200 = 0;
        real tuv020 = 0;
        real tuv002 = 0;
        real tuv110 = 0;
        real tuv101 = 0;
        real tuv011 = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            real tu20 = 0;
            real tu11 = 0;
            real tu02 = 0;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
                real3 t = make_real3(0, 0, 0);
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
                    real tq = pmeGrid[gridIndex].x;
                    real4 tadd = theta3[iz];
                    t.x += tq*tadd.x;
                    t.y += tq*tadd.y;
                    t.z += tq*tadd.z;
                }
                tu00 += u.x*t.x;
                tu10 += u.y*t.x;
                tu01 += u.x*t.y;
                tu20 += u.z*t.x;
                tu11 += u.y*t.y;
                tu02 += u.x*t.z;
            }
            tuv000 += v.x*tu00;
            tuv100 += v.y*tu00;
            tuv010 += v.x*tu10;
            tuv001 += v.x*tu01;
            tuv200 += v.z*tu00;
            tuv020 += v.x*tu20;
            tuv002 += v.x*tu02;
            tuv110 += v.y*tu10;
            tuv101 += v.y*tu01;
            tuv011 += v.x*tu11;
        }
        phi[m] = tuv000;
        phi[m+NUM_ATOMS] = tuv100;
        phi[m+NUM_ATOMS*2] = tuv010;
        phi[m+NUM_ATOMS*3] = tuv001;
        phi[m+NUM_ATOMS*4] = tuv200;
        phi[m+NUM_ATOMS*5] = tuv020;
        phi[m+NUM_ATOMS*6] = tuv002;
        phi[m+NUM_ATOMS*7] = tuv110;
        phi[m+NUM_ATOMS*8] = tuv101;
        phi[m+NUM_ATOMS*9] = tuv011;
        mm_long fieldx = realToFixedPoint(-tuv100*fracToCart[0][0]-tuv010*fracToCart[0][1]-tuv001*fracToCart[0][2]);
        mm_long fieldy = realToFixedPoint(-tuv100*fracToCart[1][0]-tuv010*fracToCart[1][1]-tuv001*fracToCart[1][2]);
        mm_long fieldz = realToFixedPoint(-tuv100*fracToCart[2][0]-tuv010*fracToCart[2][1]-tuv001*fracToCart[2][2]);
        fieldBuffers[m] = fieldx;
        fieldBuffers[m+PADDED_NUM_ATOMS] = fieldy;
        fieldBuffers[m+2*PADDED_NUM_ATOMS] = fieldz;
        fieldPolarBuffers[m] = fieldx;
        fieldPolarBuffers[m+PADDED_NUM_ATOMS] = fieldy;
        fieldPolarBuffers[m+2*PADDED_NUM_ATOMS] = fieldz;
    }
}

KERNEL void computeInducedPotentialFromGrid(GLOBAL const real2* RESTRICT pmeGrid, GLOBAL real* RESTRICT phid, GLOBAL real* RESTRICT phip,
        GLOBAL real* RESTRICT phidp, GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 recipBoxVecX,
        real4 recipBoxVecY, real4 recipBoxVecZ) {
#if __CUDA_ARCH__ < 500
    real array[PME_ORDER*PME_ORDER];
#else
    LOCAL real sharedArray[PME_ORDER*PME_ORDER*64];
    real* array = &sharedArray[PME_ORDER*PME_ORDER*LOCAL_ID];
#endif
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        pos -= periodicBoxVecZ*floor(pos.z*recipBoxVecZ.z+0.5f);
        pos -= periodicBoxVecY*floor(pos.y*recipBoxVecY.z+0.5f);
        pos -= periodicBoxVecX*floor(pos.x*recipBoxVecX.z+0.5f);

        real w = pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x;
        real fr = remainder(GRID_SIZE_X*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_X);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta1, w, array);
        w = pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y;
        fr = remainder(GRID_SIZE_Y*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Y);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta2, w, array);
        w = pos.z*recipBoxVecZ.z;
        fr = remainder(GRID_SIZE_Z*(w-(int)(w+0.5f)+0.5f), GRID_SIZE_Z);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr-PME_ORDER+1;
        computeBSplinePoint(theta3, w, array);
        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        real tuv100_1 = 0;
        real tuv010_1 = 0;
        real tuv001_1 = 0;
        real tuv200_1 = 0;
        real tuv020_1 = 0;
        real tuv002_1 = 0;
        real tuv110_1 = 0;
        real tuv101_1 = 0;
        real tuv011_1 = 0;
        real tuv100_2 = 0;
        real tuv010_2 = 0;
        real tuv001_2 = 0;
        real tuv200_2 = 0;
        real tuv020_2 = 0;
        real tuv002_2 = 0;
        real tuv110_2 = 0;
        real tuv101_2 = 0;
        real tuv011_2 = 0;
        real tuv000 = 0;
        real tuv001 = 0;
        real tuv010 = 0;
        real tuv100 = 0;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1+ix-(igrid1+ix >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];
            real tu00_1 = 0;
            real tu01_1 = 0;
            real tu10_1 = 0;
            real tu20_1 = 0;
            real tu11_1 = 0;
            real tu02_1 = 0;
            real tu00_2 = 0;
            real tu01_2 = 0;
            real tu10_2 = 0;
            real tu20_2 = 0;
            real tu11_2 = 0;
            real tu02_2 = 0;
            real tu00 = 0;
            real tu10 = 0;
            real tu01 = 0;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2+iy-(igrid2+iy >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];
                real t0_1 = 0;
                real t1_1 = 0;
                real t2_1 = 0;
                real t0_2 = 0;
                real t1_2 = 0;
                real t2_2 = 0;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3+iz-(igrid3+iz >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
                    real4 tadd = theta3[iz];
                    real2 tq = pmeGrid[gridIndex];
                    t0_1 += tq.x*tadd.x;
                    t1_1 += tq.x*tadd.y;
                    t2_1 += tq.x*tadd.z;
                    t0_2 += tq.y*tadd.x;
                    t1_2 += tq.y*tadd.y;
                    t2_2 += tq.y*tadd.z;
                }
                tu00_1 += u.x*t0_1;
                tu10_1 += u.y*t0_1;
                tu01_1 += u.x*t1_1;
                tu20_1 += u.z*t0_1;
                tu11_1 += u.y*t1_1;
                tu02_1 += u.x*t2_1;
                tu00_2 += u.x*t0_2;
                tu10_2 += u.y*t0_2;
                tu01_2 += u.x*t1_2;
                tu20_2 += u.z*t0_2;
                tu11_2 += u.y*t1_2;
                tu02_2 += u.x*t2_2;
                real t0 = t0_1 + t0_2;
                real t1 = t1_1 + t1_2;
                tu00 += u.x*t0;
                tu10 += u.y*t0;
                tu01 += u.x*t1;
            }
            tuv100_1 += v.y*tu00_1;
            tuv010_1 += v.x*tu10_1;
            tuv001_1 += v.x*tu01_1;
            tuv200_1 += v.z*tu00_1;
            tuv020_1 += v.x*tu20_1;
            tuv002_1 += v.x*tu02_1;
            tuv110_1 += v.y*tu10_1;
            tuv101_1 += v.y*tu01_1;
            tuv011_1 += v.x*tu11_1;
            tuv100_2 += v.y*tu00_2;
            tuv010_2 += v.x*tu10_2;
            tuv001_2 += v.x*tu01_2;
            tuv200_2 += v.z*tu00_2;
            tuv020_2 += v.x*tu20_2;
            tuv002_2 += v.x*tu02_2;
            tuv110_2 += v.y*tu10_2;
            tuv101_2 += v.y*tu01_2;
            tuv011_2 += v.x*tu11_2;
            tuv000 += v.x*tu00;
            tuv100 += v.y*tu00;
            tuv010 += v.x*tu10;
            tuv001 += v.x*tu01;
        }
        phid[m]   = 0;
        phid[m+NUM_ATOMS] = tuv100_1;
        phid[m+NUM_ATOMS*2] = tuv010_1;
        phid[m+NUM_ATOMS*3] = tuv001_1;
        phid[m+NUM_ATOMS*4] = tuv200_1;
        phid[m+NUM_ATOMS*5] = tuv020_1;
        phid[m+NUM_ATOMS*6] = tuv002_1;
        phid[m+NUM_ATOMS*7] = tuv110_1;
        phid[m+NUM_ATOMS*8] = tuv101_1;
        phid[m+NUM_ATOMS*9] = tuv011_1;

        phip[m]   = 0;
        phip[m+NUM_ATOMS] = tuv100_2;
        phip[m+NUM_ATOMS*2] = tuv010_2;
        phip[m+NUM_ATOMS*3] = tuv001_2;
        phip[m+NUM_ATOMS*4] = tuv200_2;
        phip[m+NUM_ATOMS*5] = tuv020_2;
        phip[m+NUM_ATOMS*6] = tuv002_2;
        phip[m+NUM_ATOMS*7] = tuv110_2;
        phip[m+NUM_ATOMS*8] = tuv101_2;
        phip[m+NUM_ATOMS*9] = tuv011_2;

        phidp[m] = tuv000;
        phidp[m+NUM_ATOMS*1] = tuv100;
        phidp[m+NUM_ATOMS*2] = tuv010;
        phidp[m+NUM_ATOMS*3] = tuv001;
    }
}

KERNEL void computeFixedMultipoleForceAndEnergy(GLOBAL real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT phi, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    mixed energy = 0;
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        real charge = posq[i].w;
        energy += charge*phi[i];
        real3 f = make_real3(charge*phi[i+NUM_ATOMS], charge*phi[i+NUM_ATOMS*2], charge*phi[i+NUM_ATOMS*3]);
        f = make_real3(EPSILON_FACTOR*(f.x*fracToCart[0][0] + f.y*fracToCart[0][1] + f.z*fracToCart[0][2]),
                       EPSILON_FACTOR*(f.x*fracToCart[1][0] + f.y*fracToCart[1][1] + f.z*fracToCart[1][2]),
                       EPSILON_FACTOR*(f.x*fracToCart[2][0] + f.y*fracToCart[2][1] + f.z*fracToCart[2][2]));
        forceBuffers[i] -= (mm_ulong) realToFixedPoint(f.x);
        forceBuffers[i+PADDED_NUM_ATOMS] -= (mm_ulong) realToFixedPoint(f.y);
        forceBuffers[i+PADDED_NUM_ATOMS*2] -= (mm_ulong) realToFixedPoint(f.z);
    }
    energyBuffer[GLOBAL_ID] += 0.5f*EPSILON_FACTOR*energy;
}

KERNEL void computeInducedDipoleForceAndEnergy(GLOBAL real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real* RESTRICT inducedDipole_global, GLOBAL const real* RESTRICT inducedDipolePolar_global,
        GLOBAL const real* RESTRICT phi, GLOBAL const real* RESTRICT phid, GLOBAL const real* RESTRICT phip,
        GLOBAL const real* RESTRICT phidp, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real cinducedDipole[3], inducedDipole[3];
    real cinducedDipolePolar[3], inducedDipolePolar[3];
    const int deriv1[] = {4, 7, 8};
    const int deriv2[] = {7, 5, 9};
    const int deriv3[] = {8, 9, 6};
    mixed energy = 0;
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        const real scale = EPSILON_FACTOR/2;
        cinducedDipole[0] = inducedDipole_global[3*i];
        cinducedDipole[1] = inducedDipole_global[3*i+1];
        cinducedDipole[2] = inducedDipole_global[3*i+2];
        cinducedDipolePolar[0] = inducedDipolePolar_global[3*i];
        cinducedDipolePolar[1] = inducedDipolePolar_global[3*i+1];
        cinducedDipolePolar[2] = inducedDipolePolar_global[3*i+2];

        // Multiply the dipoles by cartToFrac, which is just the transpose of fracToCart.

        inducedDipole[0] = cinducedDipole[0]*fracToCart[0][0] + cinducedDipole[1]*fracToCart[1][0] + cinducedDipole[2]*fracToCart[2][0];
        inducedDipole[1] = cinducedDipole[0]*fracToCart[0][1] + cinducedDipole[1]*fracToCart[1][1] + cinducedDipole[2]*fracToCart[2][1];
        inducedDipole[2] = cinducedDipole[0]*fracToCart[0][2] + cinducedDipole[1]*fracToCart[1][2] + cinducedDipole[2]*fracToCart[2][2];
        inducedDipolePolar[0] = cinducedDipolePolar[0]*fracToCart[0][0] + cinducedDipolePolar[1]*fracToCart[1][0] + cinducedDipolePolar[2]*fracToCart[2][0];
        inducedDipolePolar[1] = cinducedDipolePolar[0]*fracToCart[0][1] + cinducedDipolePolar[1]*fracToCart[1][1] + cinducedDipolePolar[2]*fracToCart[2][1];
        inducedDipolePolar[2] = cinducedDipolePolar[0]*fracToCart[0][2] + cinducedDipolePolar[1]*fracToCart[1][2] + cinducedDipolePolar[2]*fracToCart[2][2];
        energy += (inducedDipole[0]+inducedDipolePolar[0])*phi[i+NUM_ATOMS];
        energy += (inducedDipole[1]+inducedDipolePolar[1])*phi[i+NUM_ATOMS*2];
        energy += (inducedDipole[2]+inducedDipolePolar[2])*phi[i+NUM_ATOMS*3];
        real3 f = make_real3(0, 0, 0);
        for (int k = 0; k < 3; k++) {
            int j1 = deriv1[k];
            int j2 = deriv2[k];
            int j3 = deriv3[k];
            f.x += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j1];
            f.y += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j2];
            f.z += (inducedDipole[k]+inducedDipolePolar[k])*phi[i+NUM_ATOMS*j3];
#ifdef MUTUAL_POLARIZATION
            f.x += (inducedDipole[k]*phip[i+NUM_ATOMS*j1] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j1]);
            f.y += (inducedDipole[k]*phip[i+NUM_ATOMS*j2] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j2]);
            f.z += (inducedDipole[k]*phip[i+NUM_ATOMS*j3] + inducedDipolePolar[k]*phid[i+NUM_ATOMS*j3]);
#endif
        }
        real charge = posq[i].w;
        f.x += charge*phidp[i+NUM_ATOMS];
        f.y += charge*phidp[i+NUM_ATOMS*2];
        f.z += charge*phidp[i+NUM_ATOMS*3];
        f = make_real3(scale*(f.x*fracToCart[0][0] + f.y*fracToCart[0][1] + f.z*fracToCart[0][2]),
                       scale*(f.x*fracToCart[1][0] + f.y*fracToCart[1][1] + f.z*fracToCart[1][2]),
                       scale*(f.x*fracToCart[2][0] + f.y*fracToCart[2][1] + f.z*fracToCart[2][2]));
        forceBuffers[i] -= (mm_ulong) realToFixedPoint(f.x);
        forceBuffers[i+PADDED_NUM_ATOMS] -= (mm_ulong) realToFixedPoint(f.y);
        forceBuffers[i+PADDED_NUM_ATOMS*2] -= (mm_ulong) realToFixedPoint(f.z);
    }
    energyBuffer[GLOBAL_ID] += 0.25f*EPSILON_FACTOR*energy;
}

KERNEL void recordInducedFieldDipoles(GLOBAL const real* RESTRICT phid, GLOBAL real* const RESTRICT phip, GLOBAL mm_long* RESTRICT inducedField,
        GLOBAL mm_long* RESTRICT inducedFieldPolar, GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X*recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X*recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X*recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y*recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y*recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y*recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z*recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z*recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z*recipBoxVecZ.z;
    }
    SYNC_THREADS;
    real selfDipoleScale = (4/(real) 3)*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/SQRT_PI;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        inducedField[i] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[0][0] + phid[i+NUM_ATOMS*2]*fracToCart[0][1] + phid[i+NUM_ATOMS*3]*fracToCart[0][2] - selfDipoleScale*inducedDipole[3*i]);
        inducedField[i+PADDED_NUM_ATOMS] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[1][0] + phid[i+NUM_ATOMS*2]*fracToCart[1][1] + phid[i+NUM_ATOMS*3]*fracToCart[1][2] - selfDipoleScale*inducedDipole[3*i+1]);
        inducedField[i+PADDED_NUM_ATOMS*2] -= realToFixedPoint(phid[i+NUM_ATOMS]*fracToCart[2][0] + phid[i+NUM_ATOMS*2]*fracToCart[2][1] + phid[i+NUM_ATOMS*3]*fracToCart[2][2] - selfDipoleScale*inducedDipole[3*i+2]);
        inducedFieldPolar[i] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[0][0] + phip[i+NUM_ATOMS*2]*fracToCart[0][1] + phip[i+NUM_ATOMS*3]*fracToCart[0][2] - selfDipoleScale*inducedDipolePolar[3*i]);
        inducedFieldPolar[i+PADDED_NUM_ATOMS] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[1][0] + phip[i+NUM_ATOMS*2]*fracToCart[1][1] + phip[i+NUM_ATOMS*3]*fracToCart[1][2] - selfDipoleScale*inducedDipolePolar[3*i+1]);
        inducedFieldPolar[i+PADDED_NUM_ATOMS*2] -= realToFixedPoint(phip[i+NUM_ATOMS]*fracToCart[2][0] + phip[i+NUM_ATOMS*2]*fracToCart[2][1] + phip[i+NUM_ATOMS*3]*fracToCart[2][2] - selfDipoleScale*inducedDipolePolar[3*i+2]);
    }
}
