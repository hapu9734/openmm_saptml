#ifndef SAPTML_KERNELS_H_
#define SAPTML_KERNELS_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "MBPolElectrostaticsForce.h"
#include "PartridgeSchwenkeForce.h"
#include "WaterDimerPIPForce.h"
#include "WaterTrimerPIPForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <string>
#include <vector>

namespace SAPTMLPlugin {

/**
 * Computes the forces and energy of an MBPolElectrostaticsForce.
 */
class CalcMBPolElectrostaticsForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcMBPolElectrostaticsForce";
    }
    CalcMBPolElectrostaticsForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    virtual void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /** Get the induced dipoles (in e*nm) of all particles. */
    virtual void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    /** Get the current charges (in e) of all particles. */
    virtual void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges) = 0;
    /** Copy changed parameters of the force to the context. */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const MBPolElectrostaticsForce& force) = 0;
    /** Get the PME parameters in use. */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

/**
 * Computes the forces and energy of a WaterDimerPIPForce.
 */
class CalcWaterDimerPIPForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcWaterDimerPIPForce";
    }
    CalcWaterDimerPIPForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    virtual void initialize(const OpenMM::System& system, const WaterDimerPIPForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

/**
 * Computes the forces and energy of a WaterTrimerPIPForce.
 */
class CalcWaterTrimerPIPForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcWaterTrimerPIPForce";
    }
    CalcWaterTrimerPIPForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    virtual void initialize(const OpenMM::System& system, const WaterTrimerPIPForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

/**
 * Computes the forces and energy of a PartridgeSchwenkeForce.
 */
class CalcPartridgeSchwenkeForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcPartridgeSchwenkeForce";
    }
    CalcPartridgeSchwenkeForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    virtual void initialize(const OpenMM::System& system, const PartridgeSchwenkeForce& force) = 0;
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace SAPTMLPlugin

#endif /*SAPTML_KERNELS_H_*/
