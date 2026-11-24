/**
 * Shared device functions for the MB-pol (TTM4-F) electrostatics kernels.
 *
 * Thole damping of TTM4-F: u = r/(A_i A_j)^(1/6),  x = a u^4, e = exp(-x)
 *   s3 = 1 - e                                  (field of a smeared charge / dipole)
 *   s5 = 1 - e - (4/3) x e
 *   s7 = s5 - (4/15) (4x - 1) x e
 *   s1 = 1 - e + a^(1/4) u Gamma(3/4, x)       (potential of a smeared charge)
 * "damp" passed to these functions is the product of the per-atom factors A^(1/6).
 */

/**
 * Upper incomplete gamma function Gamma(3/4, x) for x >= 0: Gamma(3/4) minus the power series of the lower function
 * for x < 7/4, otherwise the continued fraction DLMF 8.9.2 evaluated with the modified Lentz method.
 */
DEVICE real upperGammaThreeQuarters(real x) {
    const real a = 0.75f;
    const real gammaA = (real) 1.2254167024651776451;
#ifdef USE_DOUBLE_PRECISION
    const real eps = 1e-15;
    const real tiny = 1e-300;
#else
    const real eps = 1.2e-7f;
    const real tiny = 1e-30f;
#endif
    const int maxTerms = 100;
    if (x <= 0)
        return gammaA;
    real prefactor = EXP(a*LOG(x) - x);
    if (x < a+1) {
        real term = 1/a;
        real sum = term;
        for (int n = 1; n < maxTerms && fabs(term) > eps*sum; n++) {
            term *= x/(a+n);
            sum += term;
        }
        return gammaA - prefactor*sum;
    }
    real f = x+1-a;
    real c = f;
    real d = 0;
    for (int n = 1; n < maxTerms; n++) {
        real an = -n*(n-a);
        real bn = x+2*n+1-a;
        d = bn + an*d;
        if (fabs(d) < tiny)
            d = tiny;
        d = 1/d;
        c = bn + an/c;
        if (fabs(c) < tiny)
            c = tiny;
        real delta = c*d;
        f *= delta;
        if (fabs(delta-1) < eps)
            break;
    }
    return prefactor/f;
}

/**
 * Thole factors of orders 3, 5 and 7 for damping parameter a.
 */
DEVICE void tholeScales357(real r, real damp, real a, real* s3, real* s5, real* s7) {
    if (damp == 0) {
        *s3 = 1;
        *s5 = 1;
        *s7 = 1;
        return;
    }
    real u = r/damp;
    real u2 = u*u;
    real x = a*u2*u2;
    real e = EXP(-x);
    *s3 = 1 - e;
    *s5 = 1 - e - ((real) 4/3)*x*e;
    *s7 = *s5 - ((real) 4/15)*(4*x - 1)*x*e;
}

/**
 * Thole factors of orders 1 and 3 for the charge-charge damping parameter a (aQuarter = a^(1/4)).
 */
DEVICE void tholeScales13(real r, real damp, real a, real aQuarter, real* s1, real* s3) {
    if (damp == 0) {
        *s1 = 1;
        *s3 = 1;
        return;
    }
    real u = r/damp;
    real u2 = u*u;
    real x = a*u2*u2;
    real e = EXP(-x);
    *s3 = 1 - e;
    *s1 = 1 - e + aQuarter*u*upperGammaThreeQuarters(x);
}

/**
 * Pair class: 0 for sites of different waters, 1 for sites of the same water.
 */
DEVICE int pairClass(int water1, int water2) {
    return (water1 != water2) ? 0 : 1;
}

/**
 * Dipole-dipole Thole parameter for a pair: inter-molecular, or intra-molecular O-H / H-H.
 */
DEVICE real tholeDD(bool sameWater, int type1, int type2) {
    if (!sameWater)
        return THOLE_DD;
    return (type1 == 0 || type2 == 0) ? THOLE_DD_OH : THOLE_DD_HH;
}

/**
 * Charge-dipole (THOLE_CD) and dipole-dipole (tholeDD) damping scales of a pair of class pc.
 */
DEVICE void classScales(int pc, real r, real damp, int type1, int type2, real* s3CD, real* s5CD, real* s7CD, real* s3DD, real* s5DD, real* s7DD) {
    tholeScales357(r, damp, THOLE_CD, s3CD, s5CD, s7CD);
    tholeScales357(r, damp, tholeDD(pc == 1, type1, type2), s3DD, s5DD, s7DD);
}
