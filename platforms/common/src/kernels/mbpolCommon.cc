/**
 * Shared device functions for the MB-pol (TTM4-F) electrostatics kernels.
 *
 * Thole damping of TTM4-F: u = r/(A_i A_j)^(1/6),  x = a u^4, e = exp(-x)
 *   s3 = 1 - e                                  (field of a smeared charge / dipole)
 *   s5 = 1 - e - (4/3) x e
 *   s7 = s5 - (4/15) (4x - 1) x e
 *   s1 = 1 - e + a^(1/4) u Gamma(3/4) Q(3/4, x)  (potential of a smeared charge)
 * "damp" passed to these functions is the product of the per-atom factors A^(1/6).
 */

#define GAMMA_THREE_QUARTERS 1.2254167024651776f

/**
 * Regularized upper incomplete gamma function Q(3/4, x) (Numerical Recipes gser/gcf).
 */
DEVICE real gammq34(real x) {
    const real a = 0.75f;
    const real gln = 0.20328095143129538f;   // ln Gamma(3/4)
#ifdef USE_DOUBLE_PRECISION
    const real eps = 1e-15;
    const real fpmin = 1e-300;
    const int maxit = 300;
#else
    const real eps = 1e-7f;
    const real fpmin = 1e-30f;
    const int maxit = 100;
#endif
    if (x <= 0)
        return 1;
    if (x < a+1) {
        // series for P(a,x)
        real ap = a;
        real sum = 1/a;
        real del = sum;
        for (int n = 0; n < maxit; n++) {
            ap += 1;
            del *= x/ap;
            sum += del;
            if (fabs(del) < fabs(sum)*eps)
                break;
        }
        return 1 - sum*EXP(-x + a*LOG(x) - gln);
    }
    else {
        // continued fraction for Q(a,x)
        real b = x + 1 - a;
        real c = 1/fpmin;
        real d = 1/b;
        real h = d;
        for (int i = 1; i <= maxit; i++) {
            real an = -i*(i-a);
            b += 2;
            d = an*d + b;
            if (fabs(d) < fpmin)
                d = fpmin;
            c = b + an/c;
            if (fabs(c) < fpmin)
                c = fpmin;
            d = 1/d;
            real del = d*c;
            h *= del;
            if (fabs(del-1) <= eps)
                break;
        }
        return EXP(-x + a*LOG(x) - gln)*h;
    }
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
    *s1 = 1 - e + aQuarter*u*GAMMA_THREE_QUARTERS*gammq34(x);
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
 * Charge-dipole and dipole-dipole damping scales of a pair: the intermolecular dipole-dipole parameter between
 * molecules, the intramolecular O-H / H-H parameters inside a water.
 */
DEVICE void classScales(int pc, real r, real damp, int type1, int type2, real* s3CD, real* s5CD, real* s7CD, real* s3DD, real* s5DD, real* s7DD) {
    tholeScales357(r, damp, THOLE_CD, s3CD, s5CD, s7CD);
    tholeScales357(r, damp, tholeDD(pc == 1, type1, type2), s3DD, s5DD, s7DD);
}
