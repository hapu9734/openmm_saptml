#ifndef OPENMM_MBPOLELECTROSTATICSFORCEIMPL_H_
#define OPENMM_MBPOLELECTROSTATICSFORCEIMPL_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "MBPolElectrostaticsForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

class OPENMM_EXPORT_SAPTML MBPolElectrostaticsForceImpl : public OpenMM::ForceImpl {
public:
    MBPolElectrostaticsForceImpl(const MBPolElectrostaticsForce& owner);
    ~MBPolElectrostaticsForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const MBPolElectrostaticsForce& getOwner() const {
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
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges);
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    const MBPolElectrostaticsForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_MBPOLELECTROSTATICSFORCEIMPL_H_*/
