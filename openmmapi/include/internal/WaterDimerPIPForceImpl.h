#ifndef OPENMM_WATERDIMERPIPFORCEIMPL_H_
#define OPENMM_WATERDIMERPIPFORCEIMPL_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "WaterDimerPIPForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

class OPENMM_EXPORT_SAPTML WaterDimerPIPForceImpl : public OpenMM::ForceImpl {
public:
    WaterDimerPIPForceImpl(const WaterDimerPIPForce& owner);
    ~WaterDimerPIPForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const WaterDimerPIPForce& getOwner() const {
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
    const WaterDimerPIPForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_WATERDIMERPIPFORCEIMPL_H_*/
