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

#include "internal/WaterDimerPIPForceImpl.h"
#include "SAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <set>
#include <sstream>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

WaterDimerPIPForceImpl::WaterDimerPIPForceImpl(const WaterDimerPIPForce& owner) : owner(owner) {
}

WaterDimerPIPForceImpl::~WaterDimerPIPForceImpl() {
}

void WaterDimerPIPForceImpl::initialize(ContextImpl& context) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumWaters() < 2)
        throw OpenMMException("WaterDimerPIPForce: At least two waters are required");
    set<int> seen;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int particles[3];
        owner.getWaterParameters(i, particles[0], particles[1], particles[2]);
        for (int k = 0; k < 3; k++) {
            if (particles[k] < 0 || particles[k] >= numParticles) {
                stringstream msg;
                msg << "WaterDimerPIPForce: Illegal particle index for a water: " << particles[k];
                throw OpenMMException(msg.str());
            }
            if (seen.count(particles[k]) > 0) {
                stringstream msg;
                msg << "WaterDimerPIPForce: Particle " << particles[k] << " belongs to more than one water";
                throw OpenMMException(msg.str());
            }
            seen.insert(particles[k]);
        }
    }
    if (owner.getNumVariables() != 31)
        throw OpenMMException("WaterDimerPIPForce: The polynomial must have exactly 31 variables");
    if (owner.getNumTerms() == 0)
        throw OpenMMException("WaterDimerPIPForce: The polynomial has no terms");
    for (int i = 0; i < owner.getNumVariables(); i++) {
        int site1, site2;
        double k, d0;
        WaterDimerPIPForce::VariableForm form;
        owner.getVariableParameters(i, site1, site2, k, d0, form);
        if (site1 < 0 || site1 > 9 || site2 < 0 || site2 > 9 || site1 == site2)
            throw OpenMMException("WaterDimerPIPForce: Illegal site pair for a variable");
        if (!isfinite(k) || !isfinite(d0))
            throw OpenMMException("WaterDimerPIPForce: Non-finite variable parameters");
    }
    for (int i = 0; i < owner.getNumMonomials(); i++) {
        int v[4];
        owner.getMonomialParameters(i, v[0], v[1], v[2], v[3]);
        bool ended = false;
        for (int p = 0; p < 4; p++) {
            if (v[p] == -1)
                ended = true;
            else if (ended || v[p] < 0 || v[p] >= owner.getNumVariables())
                throw OpenMMException("WaterDimerPIPForce: Illegal variable index in a monomial");
        }
    }
    for (int i = 0; i < owner.getNumTerms(); i++) {
        double coefficient;
        int first, count;
        owner.getTermParameters(i, coefficient, first, count);
        if (first < 0 || count < 1 || first+count > owner.getNumMonomials())
            throw OpenMMException("WaterDimerPIPForce: Illegal monomial range for a term");
        if (!isfinite(coefficient))
            throw OpenMMException("WaterDimerPIPForce: Non-finite term coefficient");
    }
    if (!(owner.getSwitchingDistance() >= 0.0 && owner.getSwitchingDistance() < owner.getCutoffDistance()))
        throw OpenMMException("WaterDimerPIPForce: The switching distance must be between 0 and the cutoff distance");
    double zero, full;
    owner.getOxygenOxygenSwitch(zero, full);
    if (!(0.0 < zero && zero < full && full < owner.getCutoffDistance()))
        throw OpenMMException("WaterDimerPIPForce: The oxygen-oxygen switch must lie between 0 and the cutoff distance");
    if (owner.getNonbondedMethod() == WaterDimerPIPForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("WaterDimerPIPForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel = context.getPlatform().createKernel(CalcWaterDimerPIPForceKernel::Name(), context);
    kernel.getAs<CalcWaterDimerPIPForceKernel>().initialize(context.getSystem(), owner);
}

double WaterDimerPIPForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcWaterDimerPIPForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> WaterDimerPIPForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcWaterDimerPIPForceKernel::Name());
    return names;
}

vector<pair<int, int> > WaterDimerPIPForceImpl::getBondedParticles() const {
    vector<pair<int, int> > bonds;
    for (int i = 0; i < owner.getNumWaters(); i++) {
        int oxygen, hydrogen1, hydrogen2;
        owner.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        bonds.push_back(make_pair(oxygen, hydrogen1));
        bonds.push_back(make_pair(oxygen, hydrogen2));
    }
    return bonds;
}
