/**
 * Geometry-dependent TTM4-F charges (Partridge-Schwenke dipole moment surface) and their
 * derivatives, the charge-derivative force, and small helper kernels.
 *
 * The DMS evaluation is done in double precision regardless of the platform precision.
 * Layout of chargeDerivs: [water][site s (0=H1', 1=H2', 2=M')][atom P (0=O, 1=H1, 2=H2)][xyz] = dq'_s/dr_P (e/nm).
 */

#define DMS_NTERMS 84

__constant__ int dmsIdx0[DMS_NTERMS] = {
       1, 1, 1, 2, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4,
       1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 1, 1, 1, 1, 1,
       1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 1, 1, 1, 1,
       1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
       5, 6, 6, 7};
__constant__ int dmsIdx1[DMS_NTERMS] = {
       1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1,
       1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 5,
       6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4,
       5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2,
       3, 1, 2, 1};
__constant__ int dmsIdx2[DMS_NTERMS] = {
       1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1,
       5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1, 6, 5, 4, 3, 2,
       1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1, 7, 6, 5, 4,
       3, 2, 1, 6, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2,
       1, 2, 1, 1};
__constant__ double dmsCoef[DMS_NTERMS] = {
      -2.1689686086730e-03, 1.4910379754728e-02, 5.3546078430060e-02,
      -7.4055995388666e-02,-3.7764333017616e-03, 1.4089887256484e-01,
      -6.2584207687264e-02,-1.1260393113022e-01,-5.7824159269319e-02,
       1.4360743650655e-02,-1.5469680141070e-02,-1.3036350092795e-02,
       2.7515837781556e-02, 1.4098478875076e-01,-2.7663168397781e-02,
      -5.2378176254797e-03,-1.0237198381792e-02, 8.9571999265473e-02,
       7.2920263098603e-03,-2.6873260551686e-01, 2.0220870325864e-02,
      -7.0764766270927e-02, 1.2140640273760e-01, 2.0978491966341e-02,
      -1.9443840512668e-01, 4.0826835370618e-02,-4.5365190474650e-02,
       6.2779900072132e-02,-1.3194351021000e-01,-1.4673032718563e-01,
       1.1894031277247e-01,-6.4952851564679e-03, 8.8503610374493e-02,
       1.4899437409291e-01, 1.3962841511565e-01,-2.6459446720450e-02,
      -5.0128914532773e-02, 1.8329676428116e-01,-1.5559089125095e-01,
      -4.0176879767592e-02, 3.6192059996636e-01, 1.0202887240343e-01,
       1.9318668580051e-01,-4.3435977107932e-01,-4.2080828803311e-02,
       1.9144626027273e-01,-1.7851138969948e-01, 1.0524533875070e-01,
      -1.7954071602185e-02, 5.2022455612120e-02,-2.8891891146828e-01,
      -4.7452036576319e-02,-1.0939400546289e-01, 3.5916564473568e-01,
      -2.0162789820172e-01,-3.5838629543696e-01, 5.6706523551202e-03,
       1.3849337488211e-01,-4.1733982195604e-01, 4.1641570764241e-01,
      -1.2243429796296e-01, 4.7141730971228e-02,-1.8224510249551e-01,
      -1.8880981556620e-01,-3.1992359561800e-01,-1.8567550546587e-01,
       6.1850530431280e-01,-6.1142756235141e-02,-1.6996135584933e-01,
       5.4252879499871e-01, 6.6128603899427e-01, 1.2107016404639e-02,
      -1.9633639729189e-01, 2.7652059420824e-03,-2.2684111109778e-01,
      -4.7924491598635e-01, 2.4287790137314e-01,-1.4296023329441e-01,
       8.9664665907006e-02,-1.4003228575602e-01,-1.3321543452254e-01,
      -1.8340983193745e-01, 2.3426707273520e-01, 1.5141050914514e-01};

/**
 * Thread per water: compute the TTM4-F charges of H1, H2, M (O carries no charge) and
 * the derivatives of the three charges with respect to the O, H1, H2 positions.
 */
KERNEL void computeWaterCharges(GLOBAL real4* RESTRICT posq, GLOBAL const int4* RESTRICT waterParticles, GLOBAL real* RESTRICT chargeDerivs) {
    const double Bohr_A = 0.52917721092;
    const double gammaM = 0.426706882;
    const double gamma1 = 1.0 - gammaM;
    const double gamma2 = gammaM/2;
    const double costhe = -0.24780227221366464506;
    const double reoh = 0.958649;
    const double b1D = 1.0;
    const double a = 0.2999e0;
    const double b = -0.6932e0;
    const double c0 = 1.0099e0;
    const double c1 = -0.1801e0;
    const double c2 = 0.0892e0;
    for (int w = GLOBAL_ID; w < NUM_WATERS; w += GLOBAL_SIZE) {
        int4 idx = waterParticles[w];
        real4 pO = posq[idx.x];
        real4 pH1 = posq[idx.y];
        real4 pH2 = posq[idx.z];
        // internal coordinates in Angstrom
        double ROH1[3], ROH2[3];
        ROH1[0] = 10.0*((double) pH1.x - (double) pO.x);
        ROH1[1] = 10.0*((double) pH1.y - (double) pO.y);
        ROH1[2] = 10.0*((double) pH1.z - (double) pO.z);
        ROH2[0] = 10.0*((double) pH2.x - (double) pO.x);
        ROH2[1] = 10.0*((double) pH2.y - (double) pO.y);
        ROH2[2] = 10.0*((double) pH2.z - (double) pO.z);
        double dROH1 = sqrt(ROH1[0]*ROH1[0] + ROH1[1]*ROH1[1] + ROH1[2]*ROH1[2]);
        double dROH2 = sqrt(ROH2[0]*ROH2[0] + ROH2[1]*ROH2[1] + ROH2[2]*ROH2[2]);
        const double costh = (ROH1[0]*ROH2[0] + ROH1[1]*ROH2[1] + ROH1[2]*ROH2[2])/(dROH1*dROH2);
        const double efac = exp(-b1D*((dROH1 - reoh)*(dROH1 - reoh) + (dROH2 - reoh)*(dROH2 - reoh)));
        const double x1 = (dROH1 - reoh)/reoh;
        const double x2 = (dROH2 - reoh)/reoh;
        const double x3 = costh - costhe;
        double fmat[3][16];
        for (int i = 0; i < 3; ++i) {
            fmat[i][0] = 0.0;
            fmat[i][1] = 1.0;
        }
        for (int j = 2; j < 16; ++j) {
            fmat[0][j] = fmat[0][j - 1]*x1;
            fmat[1][j] = fmat[1][j - 1]*x2;
            fmat[2][j] = fmat[2][j - 1]*x3;
        }
        double p1 = 0, p2 = 0;
        double pl1 = costh;
        double pl2 = 0.5*(3*pl1*pl1 - 1.0);
        double dp1dr1 = 0, dp1dr2 = 0, dp1dcabc = 0, dp2dr1 = 0, dp2dr2 = 0, dp2dcabc = 0;
        for (int j = 1; j < DMS_NTERMS; ++j) {
            const int inI = dmsIdx0[j];
            const int inJ = dmsIdx1[j];
            const int inK = dmsIdx2[j];
            const double c = dmsCoef[j];
            p1 += c*fmat[0][inI]*fmat[1][inJ]*fmat[2][inK];
            p2 += c*fmat[0][inJ]*fmat[1][inI]*fmat[2][inK];
            dp1dr1 += c*(inI - 1)*fmat[0][inI - 1]*fmat[1][inJ]*fmat[2][inK];
            dp1dr2 += c*(inJ - 1)*fmat[0][inI]*fmat[1][inJ - 1]*fmat[2][inK];
            dp1dcabc += c*(inK - 1)*fmat[0][inI]*fmat[1][inJ]*fmat[2][inK - 1];
            dp2dr1 += c*(inJ - 1)*fmat[0][inJ - 1]*fmat[1][inI]*fmat[2][inK];
            dp2dr2 += c*(inI - 1)*fmat[0][inJ]*fmat[1][inI - 1]*fmat[2][inK];
            dp2dcabc += c*(inK - 1)*fmat[0][inJ]*fmat[1][inI]*fmat[2][inK - 1];
        }
        const double xx = Bohr_A;
        const double xx2 = xx*xx;
        dp1dr1 /= reoh/xx;
        dp1dr2 /= reoh/xx;
        dp2dr1 /= reoh/xx;
        dp2dr2 /= reoh/xx;
        const double pc0 = a*(pow(dROH1, b) + pow(dROH2, b))*(c0 + pl1*c1 + pl2*c2);
        const double dpc0dr1 = a*b*pow(dROH1, b - 1)*(c0 + pl1*c1 + pl2*c2)*xx2;
        const double dpc0dr2 = a*b*pow(dROH2, b - 1)*(c0 + pl1*c1 + pl2*c2)*xx2;
        const double dpc0dcabc = a*(pow(dROH1, b) + pow(dROH2, b))*(c1 + 0.5*(6.0*pl1)*c2)*xx;
        const double defacdr1 = -2.0*b1D*(dROH1 - reoh)*efac*xx;
        const double defacdr2 = -2.0*b1D*(dROH2 - reoh)*efac*xx;
        dp1dr1 = dp1dr1*efac + p1*defacdr1 + dpc0dr1;
        dp1dr2 = dp1dr2*efac + p1*defacdr2 + dpc0dr2;
        dp1dcabc = dp1dcabc*efac + dpc0dcabc;
        dp2dr1 = dp2dr1*efac + p2*defacdr1 + dpc0dr1;
        dp2dr2 = dp2dr2*efac + p2*defacdr2 + dpc0dr2;
        dp2dcabc = dp2dcabc*efac + dpc0dcabc;
        p1 = dmsCoef[0] + p1*efac + pc0*xx;   // q_H1 (before M-site redistribution)
        p2 = dmsCoef[0] + p2*efac + pc0*xx;   // q_H2
        const double chargeO = -(p1 + p2);
        const double gamma2div1 = gamma2/gamma1;
        posq[idx.x].w = 0;
        posq[idx.y].w = (real) (p1 + gamma2div1*(p1 + p2));
        posq[idx.z].w = (real) (p2 + gamma2div1*(p1 + p2));
        posq[idx.w].w = (real) (chargeO/gamma1);

        // derivatives of the raw charges q_H1, q_H2 with respect to r_H1, r_H2, r_O (e/Angstrom)
        dp1dr1 /= xx;
        dp1dr2 /= xx;
        dp2dr1 /= xx;
        dp2dr2 /= xx;
        const double f1q1r13 = (dp1dr1 - (dp1dcabc*costh/dROH1))/dROH1;
        const double f1q1r23 = dp1dcabc/(dROH1*dROH2);
        const double f2q1r23 = (dp1dr2 - (dp1dcabc*costh/dROH2))/dROH2;
        const double f2q1r13 = dp1dcabc/(dROH2*dROH1);
        const double f1q2r13 = (dp2dr1 - (dp2dcabc*costh/dROH1))/dROH1;
        const double f1q2r23 = dp2dcabc/(dROH1*dROH2);
        const double f2q2r23 = (dp2dr2 - (dp2dcabc*costh/dROH2))/dROH2;
        const double f2q2r13 = dp2dcabc/(dROH2*dROH1);
        // dq1[P][i] = dq_H1/dr_P, dq2[P][i] = dq_H2/dr_P, P: 0=O, 1=H1, 2=H2
        double dq1[3][3], dq2[3][3];
        for (int i = 0; i < 3; ++i) {
            dq1[1][i] = f1q1r13*ROH1[i] + f1q1r23*ROH2[i];
            dq1[2][i] = f2q1r13*ROH1[i] + f2q1r23*ROH2[i];
            dq1[0][i] = -(dq1[1][i] + dq1[2][i]);
            dq2[1][i] = f1q2r13*ROH1[i] + f1q2r23*ROH2[i];
            dq2[2][i] = f2q2r13*ROH1[i] + f2q2r23*ROH2[i];
            dq2[0][i] = -(dq2[1][i] + dq2[2][i]);
        }
        // redistributed charges: q'_H1 = q_H1 + g Q, q'_H2 = q_H2 + g Q, q'_M = -Q/gamma1 (Q = q_H1+q_H2, g = gamma2/gamma1)
        GLOBAL real* out = &chargeDerivs[27*w];
        for (int P = 0; P < 3; ++P) {
            for (int i = 0; i < 3; ++i) {
                double sum = dq1[P][i] + dq2[P][i];
                out[0*9 + P*3 + i] = (real) (10.0*(dq1[P][i] + gamma2div1*sum));   // e/nm
                out[1*9 + P*3 + i] = (real) (10.0*(dq2[P][i] + gamma2div1*sum));
                out[2*9 + P*3 + i] = (real) (10.0*(-sum/gamma1));
            }
        }
    }
}

/**
 * Charge-derivative force F_P = -eps * sum_s phi_s dq'_s/dr_P, using the total potential at the
 * charge sites (real space + reciprocal + Ewald self), and export of the site potentials.
 */
KERNEL void applyChargeDerivativeForces(GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int4* RESTRICT waterParticles, GLOBAL const real* RESTRICT chargeDerivs,
        GLOBAL const mm_long* RESTRICT potentialBuffers, GLOBAL real* RESTRICT sitePotential
#ifdef USE_EWALD
        , GLOBAL const real* RESTRICT phi, GLOBAL const real* RESTRICT phidp
#endif
        ) {
    const real potScale = 1/(real) 0x100000000;
    for (int w = GLOBAL_ID; w < NUM_WATERS; w += GLOBAL_SIZE) {
        int4 idx = waterParticles[w];
        int sites[4] = {idx.x, idx.y, idx.z, idx.w};
        real pot[4];
        for (int k = 0; k < 4; k++) {
            int i = sites[k];
            real p = potentialBuffers[i]*potScale;
#ifdef USE_EWALD
            p += phi[i] + 0.5f*phidp[i] - (2*EWALD_ALPHA/SQRT_PI)*posq[i].w;
#endif
            pot[k] = p;
            sitePotential[i] = p*ENERGY_SCALE_FACTOR;
        }
#ifdef INCLUDE_CHARGE_REDISTRIBUTION
        GLOBAL const real* dq = &chargeDerivs[27*w];
        for (int P = 0; P < 3; P++) {
            real3 f = make_real3(0);
            for (int s = 0; s < 3; s++) {
                real phis = pot[s+1];    // sites H1, H2, M
                f.x += phis*dq[s*9 + P*3 + 0];
                f.y += phis*dq[s*9 + P*3 + 1];
                f.z += phis*dq[s*9 + P*3 + 2];
            }
            f *= -ENERGY_SCALE_FACTOR;
            int i = sites[P];
            ATOMIC_ADD(&forceBuffers[i], (mm_ulong) realToFixedPoint(f.x));
            ATOMIC_ADD(&forceBuffers[i+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(f.y));
            ATOMIC_ADD(&forceBuffers[i+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(f.z));
        }
#endif
    }
}

KERNEL void recordInducedDipoles(GLOBAL const mm_long* RESTRICT fieldBuffers, GLOBAL const mm_long* RESTRICT fieldPolarBuffers,
        GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT inducedDipolePolar, GLOBAL const float* RESTRICT polarizability) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real scale = polarizability[atom]/(real) 0x100000000;
        inducedDipole[3*atom] = scale*fieldBuffers[atom];
        inducedDipole[3*atom+1] = scale*fieldBuffers[atom+PADDED_NUM_ATOMS];
        inducedDipole[3*atom+2] = scale*fieldBuffers[atom+PADDED_NUM_ATOMS*2];
        inducedDipolePolar[3*atom] = scale*fieldPolarBuffers[atom];
        inducedDipolePolar[3*atom+1] = scale*fieldPolarBuffers[atom+PADDED_NUM_ATOMS];
        inducedDipolePolar[3*atom+2] = scale*fieldPolarBuffers[atom+PADDED_NUM_ATOMS*2];
    }
}

// initial guess for the mutual induction from the converged dipoles of the last force evaluations
// (history[k] = solution k+1 evaluations ago): order 1 copy, 2 linear, 3 quadratic extrapolation
KERNEL void predictInducedDipoles(GLOBAL real* RESTRICT inducedDipole, GLOBAL real* RESTRICT inducedDipolePolar,
        GLOBAL const real* RESTRICT history, GLOBAL const real* RESTRICT historyPolar, int order) {
    const int n = 3*NUM_ATOMS;
    for (int i = GLOBAL_ID; i < n; i += GLOBAL_SIZE) {
        if (order == 1) {
            inducedDipole[i] = history[i];
            inducedDipolePolar[i] = historyPolar[i];
        }
        else if (order == 2) {
            inducedDipole[i] = 2*history[i]-history[n+i];
            inducedDipolePolar[i] = 2*historyPolar[i]-historyPolar[n+i];
        }
        else {
            inducedDipole[i] = 3*history[i]-3*history[n+i]+history[2*n+i];
            inducedDipolePolar[i] = 3*historyPolar[i]-3*historyPolar[n+i]+historyPolar[2*n+i];
        }
    }
}

KERNEL void saveInducedDipoleHistory(GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT inducedDipolePolar,
        GLOBAL real* RESTRICT history, GLOBAL real* RESTRICT historyPolar, int depth) {
    const int n = 3*NUM_ATOMS;
    for (int i = GLOBAL_ID; i < n; i += GLOBAL_SIZE) {
        for (int k = depth-1; k > 0; k--) {
            history[k*n+i] = history[(k-1)*n+i];
            historyPolar[k*n+i] = historyPolar[(k-1)*n+i];
        }
        history[i] = inducedDipole[i];
        historyPolar[i] = inducedDipolePolar[i];
    }
}
