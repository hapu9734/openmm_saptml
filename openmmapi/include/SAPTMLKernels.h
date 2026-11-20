#ifndef SAPTML_KERNELS_H_
#define SAPTML_KERNELS_H_

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
 * This kernel is invoked by MBPolElectrostaticsForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMBPolElectrostaticsForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcMBPolElectrostaticsForce";
    }
    CalcMBPolElectrostaticsForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MBPolElectrostaticsForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    virtual void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    /**
     * Get the current charges of all particles.
     *
     * @param context    the Context for which to get the charges
     * @param charges    the charge of particle i is stored into the i'th element
     */
    virtual void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MBPolElectrostaticsForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const MBPolElectrostaticsForce& force) = 0;
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

/**
 * This kernel is invoked by WaterDimerPIPForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcWaterDimerPIPForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcWaterDimerPIPForce";
    }
    CalcWaterDimerPIPForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the WaterDimerPIPForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const WaterDimerPIPForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

/**
 * This kernel is invoked by WaterTrimerPIPForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcWaterTrimerPIPForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcWaterTrimerPIPForce";
    }
    CalcWaterTrimerPIPForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the WaterTrimerPIPForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const WaterTrimerPIPForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

/**
 * This kernel is invoked by PartridgeSchwenkeForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPartridgeSchwenkeForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcPartridgeSchwenkeForce";
    }
    CalcPartridgeSchwenkeForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the PartridgeSchwenkeForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const PartridgeSchwenkeForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace SAPTMLPlugin

#endif /*SAPTML_KERNELS_H_*/
