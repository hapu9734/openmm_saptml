/* -------------------------------------------------------------------------- *
 *                              OpenMM SAPT-ML                                *
 * -------------------------------------------------------------------------- *
 * This is a plugin for the OpenMM molecular simulation toolkit               *
 * (https://openmm.org) implementing the SAPT-ML polarizable water model.     *
 *                                                                            *
 * Portions copyright (c) 2025-2026 Seoul National University and the Authors.*
 * Authors: SAPT-ML water developers                                          *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "internal/MBPolElectrostaticsForceImpl.h"
#include "SAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <set>
#include <sstream>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

MBPolElectrostaticsForceImpl::MBPolElectrostaticsForceImpl(const MBPolElectrostaticsForce& owner) : owner(owner) {
}

MBPolElectrostaticsForceImpl::~MBPolElectrostaticsForceImpl() {
}

void MBPolElectrostaticsForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (4*owner.getNumWaters() != numParticles)
        throw OpenMMException("MBPolElectrostaticsForce: Every particle of the System must belong to a water (four sites per water)");
    set<int> seen;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int particles[4];
        owner.getWaterParameters(i, particles[0], particles[1], particles[2], particles[3]);
        for (int k = 0; k < 4; k++) {
            if (particles[k] < 0 || particles[k] >= numParticles) {
                stringstream msg;
                msg << "MBPolElectrostaticsForce: Illegal particle index for a water: " << particles[k];
                throw OpenMMException(msg.str());
            }
            if (seen.count(particles[k]) > 0) {
                stringstream msg;
                msg << "MBPolElectrostaticsForce: Particle " << particles[k] << " belongs to more than one water";
                throw OpenMMException(msg.str());
            }
            seen.insert(particles[k]);
        }
        if (!system.isVirtualSite(particles[3]))
            throw OpenMMException("MBPolElectrostaticsForce: The M site of every water must be a virtual site");
    }
    if (owner.getNonbondedMethod() == MBPolElectrostaticsForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MBPolElectrostaticsForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel = context.getPlatform().createKernel(CalcMBPolElectrostaticsForceKernel::Name(), context);
    kernel.getAs<CalcMBPolElectrostaticsForceKernel>().initialize(context.getSystem(), owner);
}

double MBPolElectrostaticsForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMBPolElectrostaticsForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> MBPolElectrostaticsForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcMBPolElectrostaticsForceKernel::Name());
    return names;
}

vector<pair<int, int> > MBPolElectrostaticsForceImpl::getBondedParticles() const {
    vector<pair<int, int> > bonds;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2, msite;
        owner.getWaterParameters(i, oxygen, hydrogen1, hydrogen2, msite);
        bonds.push_back(make_pair(oxygen, hydrogen1));
        bonds.push_back(make_pair(oxygen, hydrogen2));
        bonds.push_back(make_pair(oxygen, msite));
    }
    return bonds;
}

void MBPolElectrostaticsForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMBPolElectrostaticsForceKernel>().getInducedDipoles(context, dipoles);
}

void MBPolElectrostaticsForceImpl::getCharges(ContextImpl& context, vector<double>& charges) {
    kernel.getAs<CalcMBPolElectrostaticsForceKernel>().getCharges(context, charges);
}

void MBPolElectrostaticsForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMBPolElectrostaticsForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MBPolElectrostaticsForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcMBPolElectrostaticsForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
