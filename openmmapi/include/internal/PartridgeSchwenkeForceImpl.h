#ifndef OPENMM_PARTRIDGESCHWENKEFORCEIMPL_H_
#define OPENMM_PARTRIDGESCHWENKEFORCEIMPL_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "PartridgeSchwenkeForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

class OPENMM_EXPORT_SAPTML PartridgeSchwenkeForceImpl : public OpenMM::ForceImpl {
public:
    PartridgeSchwenkeForceImpl(const PartridgeSchwenkeForce& owner);
    ~PartridgeSchwenkeForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const PartridgeSchwenkeForce& getOwner() const {
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
    const PartridgeSchwenkeForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_PARTRIDGESCHWENKEFORCEIMPL_H_*/
