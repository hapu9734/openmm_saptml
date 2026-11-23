/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "internal/WaterTrimerPIPForceImpl.h"
#include "SAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <set>
#include <sstream>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

WaterTrimerPIPForceImpl::WaterTrimerPIPForceImpl(const WaterTrimerPIPForce& owner) : owner(owner) {
}

WaterTrimerPIPForceImpl::~WaterTrimerPIPForceImpl() {
}

void WaterTrimerPIPForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumWaters() < 2)
        throw OpenMMException("WaterTrimerPIPForce: At least two waters are required");
    set<int> seen;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int particles[3];
        owner.getWaterParameters(i, particles[0], particles[1], particles[2]);
        for (int k = 0; k < 3; k++) {
            if (particles[k] < 0 || particles[k] >= numParticles) {
                stringstream msg;
                msg << "WaterTrimerPIPForce: Illegal particle index for a water: " << particles[k];
                throw OpenMMException(msg.str());
            }
            if (seen.count(particles[k]) > 0) {
                stringstream msg;
                msg << "WaterTrimerPIPForce: Particle " << particles[k] << " belongs to more than one water";
                throw OpenMMException(msg.str());
            }
            seen.insert(particles[k]);
        }
    }
    if (owner.getNumVariables() != 36)
        throw OpenMMException("WaterTrimerPIPForce: The polynomial must have exactly 36 variables");
    if (owner.getNumTerms() == 0)
        throw OpenMMException("WaterTrimerPIPForce: The polynomial has no terms");
    for (int i = 0; i < owner.getNumVariables(); i++) {
        int site1, site2;
        double k, d0, center;
        owner.getVariableParameters(i, site1, site2, k, d0, center);
        if (site1 < 0 || site1 > 8 || site2 < 0 || site2 > 8 || site1 == site2)
            throw OpenMMException("WaterTrimerPIPForce: Illegal site pair for a variable");
        if (!isfinite(k) || !isfinite(d0) || !isfinite(center))
            throw OpenMMException("WaterTrimerPIPForce: Non-finite variable parameters");
    }
    for (int i = 0; i < owner.getNumMonomials(); i++) {
        int v[4];
        owner.getMonomialParameters(i, v[0], v[1], v[2], v[3]);
        bool ended = false;
        for (int p = 0; p < 4; p++) {
            if (v[p] == -1)
                ended = true;
            else if (ended || v[p] < 0 || v[p] >= owner.getNumVariables())
                throw OpenMMException("WaterTrimerPIPForce: Illegal variable index in a monomial");
        }
    }
    for (int i = 0; i < owner.getNumTerms(); i++) {
        double coefficient;
        int first, count;
        owner.getTermParameters(i, coefficient, first, count);
        if (first < 0 || count < 1 || first+count > owner.getNumMonomials())
            throw OpenMMException("WaterTrimerPIPForce: Illegal monomial range for a term");
        if (!isfinite(coefficient))
            throw OpenMMException("WaterTrimerPIPForce: Non-finite term coefficient");
    }
    if (!(owner.getSwitchingDistance() >= 0.0 && owner.getSwitchingDistance() < owner.getCutoffDistance()))
        throw OpenMMException("WaterTrimerPIPForce: The switching distance must be between 0 and the cutoff distance");
    double zero, full;
    owner.getOxygenOxygenSwitch(zero, full);
    if (!(0.0 < zero && zero < full && full < owner.getCutoffDistance()))
        throw OpenMMException("WaterTrimerPIPForce: The oxygen-oxygen switch must lie between 0 and the cutoff distance");
    if (owner.getNonbondedMethod() == WaterTrimerPIPForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("WaterTrimerPIPForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel = context.getPlatform().createKernel(CalcWaterTrimerPIPForceKernel::Name(), context);
    kernel.getAs<CalcWaterTrimerPIPForceKernel>().initialize(context.getSystem(), owner);
}

double WaterTrimerPIPForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcWaterTrimerPIPForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> WaterTrimerPIPForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcWaterTrimerPIPForceKernel::Name());
    return names;
}

vector<pair<int, int> > WaterTrimerPIPForceImpl::getBondedParticles() const {
    vector<pair<int, int> > bonds;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2;
        owner.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        bonds.push_back(make_pair(oxygen, hydrogen1));
        bonds.push_back(make_pair(oxygen, hydrogen2));
    }
    return bonds;
}
