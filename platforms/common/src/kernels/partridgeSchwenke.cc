// One water of PartridgeSchwenkeForce (BondedUtilities interaction; atoms O, H1, H2).  The Morse and H-H terms and the
// constant of the polynomial cancel at the minimum (about 1e5 cm^-1 each) and are evaluated in double precision; the
// damped polynomial is evaluated in the precision of the context.
real3 bond1 = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
real3 bond2 = make_real3(pos3.x-pos1.x, pos3.y-pos1.y, pos3.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(bond1)
APPLY_PERIODIC_TO_DELTA(bond2)
#endif
double r1x = 10.0*bond1.x, r1y = 10.0*bond1.y, r1z = 10.0*bond1.z;     // Angstrom
double r2x = 10.0*bond2.x, r2y = 10.0*bond2.y, r2z = 10.0*bond2.z;
double rhx = r1x-r2x, rhy = r1y-r2y, rhz = r1z-r2z;
double d1 = sqrt(r1x*r1x + r1y*r1y + r1z*r1z);
double d2 = sqrt(r2x*r2x + r2y*r2y + r2z*r2z);
double dhh = sqrt(rhx*rhx + rhy*rhy + rhz*rhz);
double costh = (r1x*r2x + r1y*r2y + r1z*r2z)/(d1*d2);
double psExp1 = exp(-PS_ALPHAOH*(d1-PS_ROH));
double psExp2 = exp(-PS_ALPHAOH*(d2-PS_ROH));
double va = PS_DEOH*(psExp1*(psExp1-2.0) + psExp2*(psExp2-2.0));
double vb = PS_PHH1*exp(-PS_PHH2*dhh);
double dva1 = 2.0*PS_ALPHAOH*PS_DEOH*psExp1*(1.0-psExp1)/d1;
double dva2 = 2.0*PS_ALPHAOH*PS_DEOH*psExp2*(1.0-psExp2)/d2;
double dvb = -PS_PHH2*vb/dhh;
double psX1 = (d1-PS_REOH)/PS_REOH, psX2 = (d2-PS_REOH)/PS_REOH, psX3 = costh-PS_COSTHE;
double efac = exp(-PS_B1*((d1-PS_REOH)*(d1-PS_REOH) + (d2-PS_REOH)*(d2-PS_REOH)));
real sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
// The polynomial and its derivatives with respect to x1, x2 and x3, generated term by term with literal coefficients
// (powers p1_n = x1^n etc.), so it is evaluated in registers.
PS_POLYNOMIAL
double vc = PS_CONSTANT + efac*sum0;
energy += (va + vb + vc + PS_SHIFT)*PS_CM1_KJ;
double dvcdr1 = (-2.0*PS_B1*efac*(d1-PS_REOH)*sum0 + efac*sum1/PS_REOH)/d1;
double dvcdr2 = (-2.0*PS_B1*efac*(d2-PS_REOH)*sum0 + efac*sum2/PS_REOH)/d2;
double dvcdcth = efac*sum3;
double psScale = 10.0*PS_CM1_KJ;
double a1 = (dva1+dvcdr1 - dvcdcth*costh/(d1*d1))*psScale, b1 = dvcdcth/(d1*d2)*psScale, h = dvb*psScale;
double a2 = (dva2+dvcdr2 - dvcdcth*costh/(d2*d2))*psScale;
real3 force2 = make_real3((real) -(a1*r1x + b1*r2x + h*rhx), (real) -(a1*r1y + b1*r2y + h*rhy), (real) -(a1*r1z + b1*r2z + h*rhz));
real3 force3 = make_real3((real) -(a2*r2x + b1*r1x - h*rhx), (real) -(a2*r2y + b1*r1y - h*rhy), (real) -(a2*r2z + b1*r1z - h*rhz));
real3 force1 = -force2-force3;
