/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "ReferenceSAPTMLKernels.h"
#include "internal/PartridgeSchwenkeCoefficients.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <algorithm>
#include <cmath>
#include <sstream>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->periodicBoxSize;
}

// Switches of the polynomial forces: pairSwitch is a cosine from 1 (rOn) to 0 (rCut), smoothRise a quintic from 0
// (zero) to 1 (full), and smoothWindow is 1 on [window[1], window[2]] and 0 outside (window[0], window[3]).
static void pairSwitch(double r, double rOn, double rCut, double& value, double& derivative) {
    if (r <= rOn) {
        value = 1.0;
        derivative = 0.0;
    }
    else if (r >= rCut) {
        value = 0.0;
        derivative = 0.0;
    }
    else {
        double argument = M_PI*(r-rOn)/(rCut-rOn);
        value = 0.5*(1.0+cos(argument));
        derivative = -0.5*M_PI*sin(argument)/(rCut-rOn);
    }
}

static void smoothRise(double r, double zero, double full, double& value, double& derivative) {
    if (r <= zero) {
        value = 0.0;
        derivative = 0.0;
    }
    else if (r >= full) {
        value = 1.0;
        derivative = 0.0;
    }
    else {
        double t = (r-zero)/(full-zero);
        value = t*t*t*(10.0+t*(-15.0+6.0*t));
        derivative = 30.0*t*t*(t-1.0)*(t-1.0)/(full-zero);
    }
}

static void smoothWindow(double r, const double window[4], double& value, double& derivative) {
    if (r <= window[0] || r >= window[3]) {
        value = 0.0;
        derivative = 0.0;
    }
    else if (r < window[1])
        smoothRise(r, window[0], window[1], value, derivative);
    else if (r <= window[2]) {
        value = 1.0;
        derivative = 0.0;
    }
    else {
        double rise;
        smoothRise(r, window[2], window[3], rise, derivative);
        value = 1.0-rise;
        derivative = -derivative;
    }
}

static Vec3 minimumImage(Vec3 delta, bool periodic, const Vec3& a, const Vec3& b, const Vec3& c) {
    if (!periodic)
        return delta;
    delta -= c*floor(delta[2]/c[2]+0.5);
    delta -= b*floor(delta[1]/b[1]+0.5);
    delta -= a*floor(delta[0]/a[0]+0.5);
    return delta;
}

static bool finiteVector(const Vec3& value) {
    return isfinite(value[0]) && isfinite(value[1]) && isfinite(value[2]);
}

static bool safeForceVector(const Vec3& value, double maxSafeForce) {
    return finiteVector(value) && fabs(value[0]) <= maxSafeForce && fabs(value[1]) <= maxSafeForce && fabs(value[2]) <= maxSafeForce;
}

static void checkBox(bool periodic, const Vec3& a, const Vec3& b, const Vec3& c, const string& name) {
    if (!periodic)
        return;
    const double boxValues[9] = {a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]};
    for (int i = 0; i < 9; i++)
        if (!isfinite(boxValues[i]))
            throw OpenMMException(name+": The periodic box contains a non-finite value");
    if (!(a[0] > 0.0 && b[1] > 0.0 && c[2] > 0.0))
        throw OpenMMException(name+": The periodic box has an invalid diagonal");
}

static int monomialDegree(const array<int, 4>& monomial) {
    int degree = 0;
    while (degree < 4 && monomial[degree] != -1)
        degree++;
    return degree;
}

ReferenceCalcMBPolElectrostaticsForceKernel::ReferenceCalcMBPolElectrostaticsForceKernel(const std::string& name, const Platform& platform) :
        CalcMBPolElectrostaticsForceKernel(name, platform), numParticles(0), usePme(false), includeChargeRedistribution(true),
        polarizationType(MBPolElectrostaticsForce::Mutual), alphaEwald(0.0), cutoffDistance(0.9), mutualInducedTargetEpsilon(1e-5), sorParameter(0.55),
        mutualInducedMaxIterations(60) {
}

ReferenceCalcMBPolElectrostaticsForceKernel::~ReferenceCalcMBPolElectrostaticsForceKernel() {
}

void ReferenceCalcMBPolElectrostaticsForceKernel::loadParameters(const MBPolElectrostaticsForce& force) {
    int numWaters = force.getNumWaters();
    numParticles = 4*numWaters;
    charges.assign(numParticles, 0.0);
    tholes.assign(5*numParticles, 0.0);
    dampingFactors.assign(numParticles, 0.0);
    polarity.assign(numParticles, 0.0);
    multipoleAtomZs.assign(numParticles, -1);
    multipoleAtomXs.assign(numParticles, -1);
    multipoleAtomYs.assign(numParticles, -1);
    siteTypes.assign(numParticles, 0);
    waters.assign(numWaters, vector<int>(4, -1));
    double qO, qH, qM, aO, aH, dO, dH, dM, tcc, tcd, tdd, tddOH, tddHH;
    force.getFixedCharges(qO, qH, qM);
    force.getPolarizabilities(aO, aH);
    force.getDampingFactors(dO, dH, dM);
    force.getTholeParameters(tcc, tcd, tdd, tddOH, tddHH);
    for (int w = 0; w < numWaters; w++) {
        int idx[4];
        force.getWaterParameters(w, idx[0], idx[1], idx[2], idx[3]);
        double q[4] = {qO, qH, qH, qM};
        double a[4] = {aO, aH, aH, 0.0};
        double d[4] = {dO, dH, dH, dM};
        int type[4] = {0, 1, 1, 2};
        for (int k = 0; k < 4; k++) {
            int i = idx[k];
            waters[w][k] = i;
            charges[i] = q[k];
            polarity[i] = a[k];
            dampingFactors[i] = d[k];
            siteTypes[i] = type[k];
            tholes[5*i] = tcc;
            tholes[5*i+1] = tcd;
            tholes[5*i+2] = tdd;
            tholes[5*i+3] = tddOH;
            tholes[5*i+4] = tddHH;
            // The three other sites of the water, used to identify intramolecular pairs.
            int others[3];
            int n = 0;
            for (int kk = 0; kk < 4; kk++)
                if (kk != k)
                    others[n++] = idx[kk];
            multipoleAtomZs[i] = others[0];
            multipoleAtomXs[i] = others[1];
            multipoleAtomYs[i] = others[2];
        }
    }
    includeChargeRedistribution = force.getIncludeChargeRedistribution();
    polarizationType = force.getPolarizationType();
    mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
    mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    sorParameter = force.getSORParameter();
}

void ReferenceCalcMBPolElectrostaticsForceKernel::initialize(const System& system, const MBPolElectrostaticsForce& force) {
    loadParameters(force);
    usePme = (force.getNonbondedMethod() == MBPolElectrostaticsForce::PME);
    if (usePme) {
        cutoffDistance = force.getCutoffDistance();
        int nx, ny, nz;
        force.getPMEParameters(alphaEwald, nx, ny, nz);
        if (nx == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, nx, ny, nz, false);
        }
        pmeGridDimension.resize(3);
        pmeGridDimension[0] = nx;
        pmeGridDimension[1] = ny;
        pmeGridDimension[2] = nz;
    }
}

ReferenceMBPolElectrostatics* ReferenceCalcMBPolElectrostaticsForceKernel::setupReferenceMBPolElectrostatics(ContextImpl& context) {
    ReferenceMBPolElectrostatics* electrostatics;
    if (usePme) {
        ReferencePmeMBPolElectrostatics* pme = new ReferencePmeMBPolElectrostatics();
        pme->setAlphaEwald(alphaEwald);
        pme->setCutoffDistance(cutoffDistance);
        pme->setPmeGridDimensions(pmeGridDimension);
        Vec3& box = extractBoxSize(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (box[0] < minAllowedSize || box[1] < minAllowedSize || box[2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        pme->setPeriodicBoxSize(box);
        electrostatics = pme;
    }
    else
        electrostatics = new ReferenceMBPolElectrostatics();
    if (polarizationType == MBPolElectrostaticsForce::Mutual) {
        electrostatics->setPolarizationType(ReferenceMBPolElectrostatics::Mutual);
        electrostatics->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        electrostatics->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
        electrostatics->setPolarSOR(sorParameter);
    }
    else
        electrostatics->setPolarizationType(ReferenceMBPolElectrostatics::Direct);
    electrostatics->setIncludeChargeRedistribution(includeChargeRedistribution);
    electrostatics->setWaters(siteTypes, waters);
    return electrostatics;
}

double ReferenceCalcMBPolElectrostaticsForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ReferenceMBPolElectrostatics* electrostatics = setupReferenceMBPolElectrostatics(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = electrostatics->calculateForceAndEnergy(posData, charges, tholes, dampingFactors, polarity,
                                                            multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, forceData);
    lastInducedDipoles = electrostatics->getInducedDipoles();
    lastCharges = electrostatics->getCurrentCharges();
    delete electrostatics;
    return energy;
}

void ReferenceCalcMBPolElectrostaticsForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    if ((int) lastInducedDipoles.size() != numParticles)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
    dipoles = lastInducedDipoles;
}

void ReferenceCalcMBPolElectrostaticsForceKernel::getCharges(ContextImpl& context, vector<double>& outputCharges) {
    if ((int) lastCharges.size() != numParticles)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
    outputCharges = lastCharges;
}

void ReferenceCalcMBPolElectrostaticsForceKernel::copyParametersToContext(ContextImpl& context, const MBPolElectrostaticsForce& force) {
    if (4*force.getNumWaters() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of waters has changed");
    loadParameters(force);
}

void ReferenceCalcMBPolElectrostaticsForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}

void ReferenceCalcWaterDimerPIPForceKernel::initialize(const System& system, const WaterDimerPIPForce& force) {
    periodic = (force.getNonbondedMethod() == WaterDimerPIPForce::CutoffPeriodic);
    maxNeighbors = force.getMaxNeighbors();
    maxSafeForce = min(1.0e6, 0.25*2147483647.0/(1.5*maxNeighbors));
    switchingDistance = force.getSwitchingDistance();
    cutoffDistance = force.getCutoffDistance();
    force.getExtraPointParameters(inPlane, outOfPlane);
    force.getOxygenOxygenSwitch(ooSwitch[0], ooSwitch[1]);
    force.getIntermolecularOHSwitch(ohSwitch[0], ohSwitch[1]);
    force.getIntermolecularHHSwitch(hhSwitch[0], hhSwitch[1]);
    force.getIntramolecularOHWindow(ohWindow[0], ohWindow[1], ohWindow[2], ohWindow[3]);
    force.getIntramolecularHHWindow(hhWindow[0], hhWindow[1], hhWindow[2], hhWindow[3]);
    waters.resize(force.getNumWaters());
    for (int i = 0; i < force.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2;
        force.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        waters[i] = {{oxygen, hydrogen1, hydrogen2}};
    }
    int numVariables = force.getNumVariables();
    variablePairs.resize(numVariables);
    variableK.resize(numVariables);
    variableD0.resize(numVariables);
    variableForm.resize(numVariables);
    for (int i = 0; i < numVariables; i++) {
        int site1, site2;
        WaterDimerPIPForce::VariableForm form;
        force.getVariableParameters(i, site1, site2, variableK[i], variableD0[i], form);
        variablePairs[i] = make_pair(site1, site2);
        variableForm[i] = form;
    }
    monomials.resize(force.getNumMonomials());
    for (int i = 0; i < force.getNumMonomials(); i++)
        force.getMonomialParameters(i, monomials[i][0], monomials[i][1], monomials[i][2], monomials[i][3]);
    terms.resize(force.getNumTerms());
    for (int i = 0; i < force.getNumTerms(); i++) {
        force.getTermParameters(i, terms[i].coefficient, terms[i].start, terms[i].count);
        terms[i].degree = monomialDegree(monomials[terms[i].start]);
    }
}

double ReferenceCalcWaterDimerPIPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return 0.0;
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    checkBox(periodic, boxA, boxB, boxC, "WaterDimerPIPForce");
    int numWaters = waters.size();
    for (int water = 0; water < numWaters; water++)
        for (int site = 0; site < 3; site++)
            if (!finiteVector(positions[waters[water][site]])) {
                stringstream message;
                message << "WaterDimerPIPForce: Water " << water << ", site " << site << " has a non-finite coordinate";
                throw OpenMMException(message.str());
            }
    double cutoff2 = cutoffDistance*cutoffDistance;
    double totalEnergy = 0.0;
    for (int waterA = 0; waterA < numWaters; waterA++) {
        const Vec3& originRaw = positions[waters[waterA][0]];
        int neighborCount = 0;
        for (int waterB = waterA+1; waterB < numWaters; waterB++) {
            Vec3 deltaOO = minimumImage(positions[waters[waterB][0]]-originRaw, periodic, boxA, boxB, boxC);
            if (deltaOO.dot(deltaOO) >= cutoff2)
                continue;
            neighborCount++;

            // Build the ten sites of the dimer: O, H1, H2, X1, X2 of each water, imaged next to the first oxygen.

            int atomIndices[6];
            Vec3 sites[10], siteForces[10], oh[2][2];
            int waterSet[2] = {waterA, waterB};
            for (int w = 0; w < 2; w++) {
                int water = waterSet[w];
                Vec3 rawO = positions[waters[water][0]];
                Vec3 imagedO = (w == 0 ? rawO : originRaw+minimumImage(rawO-originRaw, periodic, boxA, boxB, boxC));
                sites[5*w] = imagedO;
                atomIndices[3*w] = waters[water][0];
                for (int site = 1; site < 3; site++) {
                    int atom = waters[water][site];
                    atomIndices[3*w+site] = atom;
                    sites[5*w+site] = imagedO+minimumImage(positions[atom]-rawO, periodic, boxA, boxB, boxC);
                }
                oh[w][0] = sites[5*w+1]-imagedO;
                oh[w][1] = sites[5*w+2]-imagedO;
                Vec3 cross = oh[w][0].cross(oh[w][1]);
                Vec3 mid = imagedO+(oh[w][0]+oh[w][1])*(0.5*inPlane);
                sites[5*w+3] = mid+cross*outOfPlane;
                sites[5*w+4] = mid-cross*outOfPlane;
                for (int site = 0; site < 5; site++)
                    siteForces[5*w+site] = Vec3();
            }
            int numVariables = variablePairs.size();
            double distances[31];
            for (int variable = 0; variable < numVariables; variable++) {
                Vec3 delta = sites[variablePairs[variable].second]-sites[variablePairs[variable].first];
                double r = sqrt(delta.dot(delta));
                if (!isfinite(r) || !(r > 1.0e-8)) {
                    stringstream message;
                    message << "WaterDimerPIPForce: Bad distance in dimer (" << waterA << ", " << waterB << "), variable " << variable;
                    throw OpenMMException(message.str());
                }
                distances[variable] = r;
            }

            // The switching function of the O-O distance (variable 14) and the short-range switches.

            double switchValue, dSwitch;
            pairSwitch(distances[14], switchingDistance, cutoffDistance, switchValue, dSwitch);
            if (switchValue == 0.0)
                continue;
            int guardVariables[15] = {14, 0, 1, 2, 3, 4, 5};
            double guardFactors[15], guardDerivatives[15];
            int numGuards = 7;
            smoothRise(distances[14], ooSwitch[0], ooSwitch[1], guardFactors[0], guardDerivatives[0]);
            for (int i = 0; i < 2; i++)
                smoothWindow(distances[i], hhWindow, guardFactors[1+i], guardDerivatives[1+i]);
            for (int i = 0; i < 4; i++)
                smoothWindow(distances[2+i], ohWindow, guardFactors[3+i], guardDerivatives[3+i]);
            if (hhSwitch[1] > 0.0)
                for (int v = 6; v < 10; v++) {
                    smoothRise(distances[v], hhSwitch[0], hhSwitch[1], guardFactors[numGuards], guardDerivatives[numGuards]);
                    guardVariables[numGuards++] = v;
                }
            if (ohSwitch[1] > 0.0)
                for (int v = 10; v < 14; v++) {
                    smoothRise(distances[v], ohSwitch[0], ohSwitch[1], guardFactors[numGuards], guardDerivatives[numGuards]);
                    guardVariables[numGuards++] = v;
                }
            double guardValue = 1.0;
            for (int i = 0; i < numGuards; i++)
                guardValue *= guardFactors[i];
            if (guardValue == 0.0)
                continue;
            double dGuardDr[31] = {0.0};
            for (int i = 0; i < numGuards; i++) {
                double product = guardDerivatives[i];
                for (int j = 0; j < numGuards; j++)
                    if (j != i)
                        product *= guardFactors[j];
                dGuardDr[guardVariables[i]] = product;
            }

            double variables[31], dVdr[31];
            for (int variable = 0; variable < numVariables; variable++) {
                double rAngstrom = 10.0*distances[variable];
                double exponent = -variableK[variable]*(rAngstrom-variableD0[variable]);
                if (!isfinite(exponent) || exponent > 80.0) {
                    stringstream message;
                    message << "WaterDimerPIPForce: Unsafe exponential argument " << exponent << " in dimer (" << waterA << ", " << waterB << "), variable " << variable;
                    throw OpenMMException(message.str());
                }
                double value = exp(exponent);
                double derivative = -variableK[variable]*value;
                if (variableForm[variable] == WaterDimerPIPForce::Coulomb) {
                    derivative = (derivative-value/rAngstrom)/rAngstrom;
                    value /= rAngstrom;
                }
                if (!isfinite(value) || !isfinite(derivative))
                    throw OpenMMException("WaterDimerPIPForce: Non-finite variable");
                variables[variable] = value;
                dVdr[variable] = 10.0*derivative;
            }
            double polynomial = 0.0;
            double dPdx[31] = {0.0};
            for (int termIndex = 0; termIndex < (int) terms.size(); termIndex++) {
                const TermData& term = terms[termIndex];
                for (int offset = 0; offset < term.count; offset++) {
                    const array<int, 4>& monomial = monomials[term.start+offset];
                    double product = 1.0;
                    for (int p = 0; p < term.degree; p++)
                        product *= variables[monomial[p]];
                    polynomial += term.coefficient*product;
                    if (includeForces)
                        for (int p = 0; p < term.degree; p++) {
                            double derivative = term.coefficient;
                            for (int q = 0; q < term.degree; q++)
                                if (q != p)
                                    derivative *= variables[monomial[q]];
                            dPdx[monomial[p]] += derivative;
                        }
                }
            }
            if (!isfinite(polynomial))
                throw OpenMMException("WaterDimerPIPForce: Non-finite polynomial");
            if (includeEnergy) {
                double dimerEnergy = guardValue*switchValue*polynomial;
                if (!isfinite(dimerEnergy))
                    throw OpenMMException("WaterDimerPIPForce: Non-finite energy");
                totalEnergy += dimerEnergy;
            }
            if (includeForces) {
                double envelope = guardValue*switchValue;
                for (int variable = 0; variable < numVariables; variable++) {
                    double dEdr = envelope*dPdx[variable]*dVdr[variable]+polynomial*switchValue*dGuardDr[variable];
                    if (variable == 14)
                        dEdr += polynomial*guardValue*dSwitch;
                    if (!isfinite(dEdr))
                        throw OpenMMException("WaterDimerPIPForce: Non-finite radial derivative");
                    Vec3 delta = sites[variablePairs[variable].second]-sites[variablePairs[variable].first];
                    Vec3 force = delta*(dEdr/distances[variable]);
                    if (!safeForceVector(force, maxSafeForce))
                        throw OpenMMException("WaterDimerPIPForce: Non-finite site force");
                    siteForces[variablePairs[variable].first] += force;
                    siteForces[variablePairs[variable].second] -= force;
                }

                // Redistribute the forces on the extra points onto O, H1 and H2.

                for (int w = 0; w < 2; w++) {
                    Vec3 f1 = siteForces[5*w+3];
                    Vec3 f2 = siteForces[5*w+4];
                    Vec3 gm = f1-f2;
                    Vec3 t1 = oh[w][1].cross(gm);
                    Vec3 t2 = oh[w][0].cross(gm);
                    Vec3 gsum = f1+f2;
                    Vec3 inPlaneForce = gsum*(0.5*inPlane);
                    Vec3 fh1 = inPlaneForce+t1*outOfPlane;
                    Vec3 fh2 = inPlaneForce-t2*outOfPlane;
                    siteForces[5*w] += gsum-(fh1+fh2);
                    siteForces[5*w+1] += fh1;
                    siteForces[5*w+2] += fh2;
                }
                for (int w = 0; w < 2; w++)
                    for (int site = 0; site < 3; site++) {
                        if (!safeForceVector(siteForces[5*w+site], maxSafeForce))
                            throw OpenMMException("WaterDimerPIPForce: Non-finite accumulated site force");
                        forces[atomIndices[3*w+site]] += siteForces[5*w+site];
                    }
            }
        }
        if (neighborCount > maxNeighbors) {
            stringstream message;
            message << "WaterDimerPIPForce: Water " << waterA << " has " << neighborCount << " neighbors, exceeding maxNeighbors=" << maxNeighbors;
            throw OpenMMException(message.str());
        }
    }
    return totalEnergy;
}

void ReferenceCalcWaterTrimerPIPForceKernel::initialize(const System& system, const WaterTrimerPIPForce& force) {
    periodic = (force.getNonbondedMethod() == WaterTrimerPIPForce::CutoffPeriodic);
    maxNeighbors = force.getMaxNeighbors();
    maxSafeForce = min(1.0e5, 0.25*2147483647.0/(1.5*maxNeighbors*(maxNeighbors-1)));
    switchingDistance = force.getSwitchingDistance();
    cutoffDistance = force.getCutoffDistance();
    force.getOxygenOxygenSwitch(ooSwitch[0], ooSwitch[1]);
    force.getIntermolecularOHSwitch(ohSwitch[0], ohSwitch[1]);
    force.getIntermolecularHHSwitch(hhSwitch[0], hhSwitch[1]);
    force.getIntramolecularOHWindow(ohWindow[0], ohWindow[1], ohWindow[2], ohWindow[3]);
    force.getIntramolecularHHWindow(hhWindow[0], hhWindow[1], hhWindow[2], hhWindow[3]);
    waters.resize(force.getNumWaters());
    for (int i = 0; i < force.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2;
        force.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        waters[i] = {{oxygen, hydrogen1, hydrogen2}};
    }
    int numVariables = force.getNumVariables();
    variablePairs.resize(numVariables);
    variableK.resize(numVariables);
    variableD0.resize(numVariables);
    variableCenter.resize(numVariables);
    for (int i = 0; i < numVariables; i++) {
        int site1, site2;
        force.getVariableParameters(i, site1, site2, variableK[i], variableD0[i], variableCenter[i]);
        variablePairs[i] = make_pair(site1, site2);
    }
    monomials.resize(force.getNumMonomials());
    for (int i = 0; i < force.getNumMonomials(); i++)
        force.getMonomialParameters(i, monomials[i][0], monomials[i][1], monomials[i][2], monomials[i][3]);
    terms.resize(force.getNumTerms());
    for (int i = 0; i < force.getNumTerms(); i++) {
        force.getTermParameters(i, terms[i].coefficient, terms[i].start, terms[i].count);
        terms[i].degree = monomialDegree(monomials[terms[i].start]);
    }
}

double ReferenceCalcWaterTrimerPIPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return 0.0;
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    checkBox(periodic, boxA, boxB, boxC, "WaterTrimerPIPForce");
    int numWaters = waters.size();
    for (int water = 0; water < numWaters; water++)
        for (int site = 0; site < 3; site++)
            if (!finiteVector(positions[waters[water][site]])) {
                stringstream message;
                message << "WaterTrimerPIPForce: Water " << water << ", site " << site << " has a non-finite coordinate";
                throw OpenMMException(message.str());
            }

    double cutoff2 = cutoffDistance*cutoffDistance;
    vector<vector<int> > neighbors(numWaters);
    for (int i = 0; i < numWaters; i++) {
        const Vec3& posI = positions[waters[i][0]];
        for (int j = 0; j < numWaters; j++) {
            if (i == j)
                continue;
            Vec3 delta = minimumImage(positions[waters[j][0]]-posI, periodic, boxA, boxB, boxC);
            if (delta.dot(delta) < cutoff2)
                neighbors[i].push_back(j);
        }
        if ((int) neighbors[i].size() > maxNeighbors) {
            stringstream message;
            message << "WaterTrimerPIPForce: Water " << i << " has " << neighbors[i].size() << " neighbors, exceeding maxNeighbors=" << maxNeighbors;
            throw OpenMMException(message.str());
        }
    }

    // Every trimer is visited once: an open chain from its central water, a closed triangle from its lowest water.

    double totalEnergy = 0.0;
    for (int center = 0; center < numWaters; center++) {
        for (int a = 0; a < (int) neighbors[center].size(); a++) {
            int waterJ = neighbors[center][a];
            for (int b = a+1; b < (int) neighbors[center].size(); b++) {
                int waterK = neighbors[center][b];
                Vec3 deltaJK = minimumImage(positions[waters[waterK][0]]-positions[waters[waterJ][0]], periodic, boxA, boxB, boxC);
                bool openChain = deltaJK.dot(deltaJK) >= cutoff2;
                bool uniqueClosedTriangle = center < waterJ && center < waterK;
                if (!openChain && !uniqueClosedTriangle)
                    continue;
                int waterSet[3] = {center, waterJ, waterK};
                int atomIndices[9];
                Vec3 sites[9], siteForces[9];
                Vec3 origin = positions[waters[center][0]];
                for (int w = 0; w < 3; w++) {
                    int water = waterSet[w];
                    int oxygen = waters[water][0];
                    atomIndices[3*w] = oxygen;
                    atomIndices[3*w+1] = waters[water][1];
                    atomIndices[3*w+2] = waters[water][2];
                    Vec3 rawO = positions[oxygen];
                    Vec3 imagedO = (w == 0 ? origin : origin+minimumImage(rawO-origin, periodic, boxA, boxB, boxC));
                    sites[3*w] = imagedO;
                    sites[3*w+1] = imagedO+minimumImage(positions[waters[water][1]]-rawO, periodic, boxA, boxB, boxC);
                    sites[3*w+2] = imagedO+minimumImage(positions[waters[water][2]]-rawO, periodic, boxA, boxB, boxC);
                    for (int site = 0; site < 3; site++)
                        siteForces[3*w+site] = Vec3();
                }
                double distances[36];
                for (int variable = 0; variable < 36; variable++) {
                    Vec3 delta = sites[variablePairs[variable].second]-sites[variablePairs[variable].first];
                    double r = sqrt(delta.dot(delta));
                    if (!isfinite(r)) {
                        stringstream message;
                        message << "WaterTrimerPIPForce: Non-finite distance in trimer (" << center << ", " << waterJ << ", " << waterK << "), variable " << variable;
                        throw OpenMMException(message.str());
                    }
                    distances[variable] = r;
                }

                // The MB-pol three-body switch of the O-O distances (variables 33-35) and the short-range switches.

                double sAB, sAC, sBC, dsAB, dsAC, dsBC;
                pairSwitch(distances[33], switchingDistance, cutoffDistance, sAB, dsAB);
                pairSwitch(distances[34], switchingDistance, cutoffDistance, sAC, dsAC);
                pairSwitch(distances[35], switchingDistance, cutoffDistance, sBC, dsBC);
                double switchValue = sAB*sAC+sAB*sBC+sAC*sBC;
                if (switchValue == 0.0)
                    continue;
                int guardVariables[36] = {33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8};
                int numGuards = 12;
                double guardFactors[36], guardDerivatives[36];
                for (int i = 0; i < 3; i++)
                    smoothRise(distances[33+i], ooSwitch[0], ooSwitch[1], guardFactors[i], guardDerivatives[i]);
                for (int i = 0; i < 3; i++)
                    smoothWindow(distances[i], hhWindow, guardFactors[3+i], guardDerivatives[3+i]);
                for (int i = 0; i < 6; i++)
                    smoothWindow(distances[3+i], ohWindow, guardFactors[6+i], guardDerivatives[6+i]);
                if (ohSwitch[1] > 0.0)
                    for (int v = 21; v < 33; v++) {
                        smoothRise(distances[v], ohSwitch[0], ohSwitch[1], guardFactors[numGuards], guardDerivatives[numGuards]);
                        guardVariables[numGuards++] = v;
                    }
                if (hhSwitch[1] > 0.0)
                    for (int v = 9; v < 21; v++) {
                        smoothRise(distances[v], hhSwitch[0], hhSwitch[1], guardFactors[numGuards], guardDerivatives[numGuards]);
                        guardVariables[numGuards++] = v;
                    }
                double guardValue = 1.0;
                for (int i = 0; i < numGuards; i++)
                    guardValue *= guardFactors[i];
                if (guardValue == 0.0)
                    continue;
                double dGuardDr[36] = {0.0};
                for (int i = 0; i < numGuards; i++) {
                    double product = guardDerivatives[i];
                    for (int j = 0; j < numGuards; j++)
                        if (j != i)
                            product *= guardFactors[j];
                    dGuardDr[guardVariables[i]] = product;
                }

                double variables[36], shifted[36];
                for (int variable = 0; variable < 36; variable++) {
                    if (!(distances[variable] > 1.0e-8)) {
                        stringstream message;
                        message << "WaterTrimerPIPForce: Coincident sites in trimer (" << center << ", " << waterJ << ", " << waterK << "), variable " << variable;
                        throw OpenMMException(message.str());
                    }
                    double exponent = -variableK[variable]*(10.0*distances[variable]-variableD0[variable]);
                    if (!isfinite(exponent) || exponent > 80.0) {
                        stringstream message;
                        message << "WaterTrimerPIPForce: Unsafe exponential argument " << exponent << " in trimer (" << center << ", " << waterJ << ", " << waterK << "), variable " << variable;
                        throw OpenMMException(message.str());
                    }
                    variables[variable] = exp(exponent);
                    shifted[variable] = variables[variable]-variableCenter[variable];
                }
                double polynomial = 0.0;
                double dPdx[36] = {0.0};
                for (int termIndex = 0; termIndex < (int) terms.size(); termIndex++) {
                    const TermData& term = terms[termIndex];
                    for (int offset = 0; offset < term.count; offset++) {
                        const array<int, 4>& monomial = monomials[term.start+offset];
                        double product = 1.0;
                        for (int p = 0; p < term.degree; p++)
                            product *= shifted[monomial[p]];
                        polynomial += term.coefficient*product;
                        if (includeForces)
                            for (int p = 0; p < term.degree; p++) {
                                double derivative = term.coefficient;
                                for (int q = 0; q < term.degree; q++)
                                    if (q != p)
                                        derivative *= shifted[monomial[q]];
                                dPdx[monomial[p]] += derivative;
                            }
                    }
                }
                if (!isfinite(polynomial))
                    throw OpenMMException("WaterTrimerPIPForce: Non-finite polynomial");
                if (includeEnergy) {
                    double trimerEnergy = guardValue*switchValue*polynomial;
                    if (!isfinite(trimerEnergy))
                        throw OpenMMException("WaterTrimerPIPForce: Non-finite energy");
                    totalEnergy += trimerEnergy;
                }
                if (includeForces) {
                    double dSwitchDr[36] = {0.0};
                    dSwitchDr[33] = dsAB*(sAC+sBC);
                    dSwitchDr[34] = dsAC*(sAB+sBC);
                    dSwitchDr[35] = dsBC*(sAB+sAC);
                    double envelope = guardValue*switchValue;
                    for (int variable = 0; variable < 36; variable++) {
                        double dEdr = envelope*dPdx[variable]*(-10.0*variableK[variable]*variables[variable])+
                                      polynomial*(guardValue*dSwitchDr[variable]+switchValue*dGuardDr[variable]);
                        if (!isfinite(dEdr))
                            throw OpenMMException("WaterTrimerPIPForce: Non-finite radial derivative");
                        Vec3 delta = sites[variablePairs[variable].second]-sites[variablePairs[variable].first];
                        Vec3 force = delta*(dEdr/distances[variable]);
                        if (!safeForceVector(force, maxSafeForce))
                            throw OpenMMException("WaterTrimerPIPForce: Non-finite site force");
                        siteForces[variablePairs[variable].first] += force;
                        siteForces[variablePairs[variable].second] -= force;
                    }
                    for (int site = 0; site < 9; site++) {
                        if (!safeForceVector(siteForces[site], maxSafeForce))
                            throw OpenMMException("WaterTrimerPIPForce: Non-finite accumulated site force");
                        forces[atomIndices[site]] += siteForces[site];
                    }
                }
            }
        }
    }
    return totalEnergy;
}

void ReferenceCalcPartridgeSchwenkeForceKernel::initialize(const System& system, const PartridgeSchwenkeForce& force) {
    using namespace PartridgeSchwenke;
    periodic = force.usesPeriodicBoundaryConditions();
    waters.resize(force.getNumWaters());
    for (int i = 0; i < force.getNumWaters(); i++)
        force.getWaterParameters(i, waters[i][0], waters[i][1], waters[i][2]);
    coefficients.resize(NumTerms);
    for (int j = 0; j < NumTerms; j++)
        coefficients[j] = f5z*c5zA[j] + fbasis*cbasis[j] + fcore*ccore[j] + frest*crest[j];
}

double ReferenceCalcPartridgeSchwenkeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    using namespace PartridgeSchwenke;
    vector<Vec3>& positions = extractPositions(context);
    vector<Vec3>& forces = extractForces(context);
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    const double deoh = f5z*deohA;
    const double phh1 = f5z*phh1A*exp(phh2);
    double totalEnergy = 0.0;
    for (const array<int, 3>& water : waters) {
        // Bond vectors in Angstrom.

        Vec3 r1 = minimumImage(positions[water[1]]-positions[water[0]], periodic, boxA, boxB, boxC)*10.0;
        Vec3 r2 = minimumImage(positions[water[2]]-positions[water[0]], periodic, boxA, boxB, boxC)*10.0;
        Vec3 rhh = r1-r2;
        double d1 = sqrt(r1.dot(r1)), d2 = sqrt(r2.dot(r2)), dhh = sqrt(rhh.dot(rhh));
        double costh = r1.dot(r2)/(d1*d2);

        // Morse bonds and H-H repulsion.

        double exp1 = exp(-alphaoh*(d1-roh));
        double exp2 = exp(-alphaoh*(d2-roh));
        double va = deoh*(exp1*(exp1-2.0) + exp2*(exp2-2.0));
        double vb = phh1*exp(-phh2*dhh);
        double dva1 = 2.0*alphaoh*deoh*exp1*(1.0-exp1)/d1;
        double dva2 = 2.0*alphaoh*deoh*exp2*(1.0-exp2)/d2;
        double dvb = -phh2*vb/dhh;

        // Damped polynomial: power[k][n] = x_k^(n-1) with power[k][0] = 0, as in the 1-based exponent tables.

        double x[3] = {(d1-reoh)/reoh, (d2-reoh)/reoh, costh-costhe};
        double power[3][16];
        for (int k = 0; k < 3; k++) {
            power[k][0] = 0.0;
            power[k][1] = 1.0;
            for (int n = 2; n < 16; n++)
                power[k][n] = power[k][n-1]*x[k];
        }
        double efac = exp(-b1*((d1-reoh)*(d1-reoh) + (d2-reoh)*(d2-reoh)));
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        for (int j = 1; j < NumTerms; j++) {
            int a = idx1[j], b = idx2[j], c = idx3[j];
            double cj = coefficients[j];
            sum0 += cj*(power[0][a]*power[1][b] + power[0][b]*power[1][a])*power[2][c];
            sum1 += cj*((a-1)*power[0][a-1]*power[1][b] + (b-1)*power[0][b-1]*power[1][a])*power[2][c];
            sum2 += cj*((b-1)*power[0][a]*power[1][b-1] + (a-1)*power[0][b]*power[1][a-1])*power[2][c];
            sum3 += cj*(power[0][a]*power[1][b] + power[0][b]*power[1][a])*(c-1)*power[2][c-1];
        }
        double vc = 2.0*coefficients[0] + efac*sum0;
        totalEnergy += (va + vb + vc + shift)*cm1_kJmol;
        if (includeForces) {
            double dvcdr1 = (-2.0*b1*efac*(d1-reoh)*sum0 + efac*sum1/reoh)/d1;
            double dvcdr2 = (-2.0*b1*efac*(d2-reoh)*sum0 + efac*sum2/reoh)/d2;
            double dvcdcth = efac*sum3;
            double scale = 10.0*cm1_kJmol;
            Vec3 g1 = (r1*(dva1+dvcdr1) + rhh*dvb + (r2/(d1*d2) - r1*(costh/(d1*d1)))*dvcdcth)*scale;
            Vec3 g2 = (r2*(dva2+dvcdr2) - rhh*dvb + (r1/(d1*d2) - r2*(costh/(d2*d2)))*dvcdcth)*scale;
            forces[water[0]] += g1+g2;
            forces[water[1]] -= g1;
            forces[water[2]] -= g2;
        }
    }
    return totalEnergy;
}
