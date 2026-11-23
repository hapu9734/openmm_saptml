/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "internal/PartridgeSchwenkeForceImpl.h"
#include "SAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <set>
#include <sstream>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

PartridgeSchwenkeForceImpl::PartridgeSchwenkeForceImpl(const PartridgeSchwenkeForce& owner) : owner(owner) {
}

PartridgeSchwenkeForceImpl::~PartridgeSchwenkeForceImpl() {
}

void PartridgeSchwenkeForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    set<int> seen;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int particles[3];
        owner.getWaterParameters(i, particles[0], particles[1], particles[2]);
        for (int k = 0; k < 3; k++) {
            if (particles[k] < 0 || particles[k] >= numParticles) {
                stringstream msg;
                msg << "PartridgeSchwenkeForce: Illegal particle index for a water: " << particles[k];
                throw OpenMMException(msg.str());
            }
            if (seen.count(particles[k]) > 0) {
                stringstream msg;
                msg << "PartridgeSchwenkeForce: Particle " << particles[k] << " belongs to more than one water";
                throw OpenMMException(msg.str());
            }
            seen.insert(particles[k]);
        }
    }
    kernel = context.getPlatform().createKernel(CalcPartridgeSchwenkeForceKernel::Name(), context);
    kernel.getAs<CalcPartridgeSchwenkeForceKernel>().initialize(context.getSystem(), owner);
}

double PartridgeSchwenkeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcPartridgeSchwenkeForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> PartridgeSchwenkeForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcPartridgeSchwenkeForceKernel::Name());
    return names;
}

vector<pair<int, int> > PartridgeSchwenkeForceImpl::getBondedParticles() const {
    vector<pair<int, int> > bonds;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2;
        owner.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        bonds.push_back(make_pair(oxygen, hydrogen1));
        bonds.push_back(make_pair(oxygen, hydrogen2));
    }
    return bonds;
}
