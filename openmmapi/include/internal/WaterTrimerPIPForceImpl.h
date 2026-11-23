#ifndef OPENMM_WATERTRIMERPIPFORCEIMPL_H_
#define OPENMM_WATERTRIMERPIPFORCEIMPL_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "WaterTrimerPIPForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

class OPENMM_EXPORT_SAPTML WaterTrimerPIPForceImpl : public OpenMM::ForceImpl {
public:
    WaterTrimerPIPForceImpl(const WaterTrimerPIPForce& owner);
    ~WaterTrimerPIPForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const WaterTrimerPIPForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>();
    }
    std::vector<std::string> getKernelNames();
    std::vector<std::pair<int, int> > getBondedParticles() const;
private:
    const WaterTrimerPIPForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_WATERTRIMERPIPFORCEIMPL_H_*/
