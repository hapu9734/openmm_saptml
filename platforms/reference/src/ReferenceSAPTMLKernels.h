#ifndef REFERENCE_SAPTML_KERNELS_H_
#define REFERENCE_SAPTML_KERNELS_H_

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

#include "SAPTMLKernels.h"
#include "ReferenceMBPolElectrostatics.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <array>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

/**
 * This kernel is invoked by MBPolElectrostaticsForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMBPolElectrostaticsForceKernel : public CalcMBPolElectrostaticsForceKernel {
public:
    ReferenceCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, const OpenMM::System& system);
    ~ReferenceCalcMBPolElectrostaticsForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MBPolElectrostaticsForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the current charges of all particles.
     *
     * @param context    the Context for which to get the charges
     * @param charges    the charge of particle i is stored into the i'th element
     */
    void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MBPolElectrostaticsForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const MBPolElectrostaticsForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    ReferenceMBPolElectrostatics* setupReferenceMBPolElectrostatics(OpenMM::ContextImpl& context);
    void loadParameters(const MBPolElectrostaticsForce& force);
    const OpenMM::System& system;
    int numParticles;
    bool usePme, includeChargeRedistribution;
    MBPolElectrostaticsForce::PolarizationType polarizationType;
    double alphaEwald, cutoffDistance, mutualInducedTargetEpsilon, sorParameter;
    int mutualInducedMaxIterations;
    std::vector<int> pmeGridDimension;
    std::vector<double> charges, tholes, dampingFactors, polarity;
    std::vector<int> axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, siteTypes, waterIndices;
    std::vector<std::vector<int> > waters;
    std::vector<std::vector<std::vector<int> > > covalentInfo;
    std::vector<OpenMM::Vec3> lastInducedDipoles;
    std::vector<double> lastCharges;
};

/**
 * This kernel is invoked by WaterDimerPIPForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcWaterDimerPIPForceKernel : public CalcWaterDimerPIPForceKernel {
public:
    ReferenceCalcWaterDimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcWaterDimerPIPForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the WaterDimerPIPForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const WaterDimerPIPForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    struct TermData {
        double coefficient;
        int start, count, degree;
    };
    bool periodic;
    int maxNeighbors;
    double switchingDistance, cutoffDistance, maxSafeForce, inPlane, outOfPlane;
    double ooSwitch[2], ohSwitch[2], hhSwitch[2], ohWindow[4], hhWindow[4];
    std::vector<std::array<int, 3> > waters;
    std::vector<std::pair<int, int> > variablePairs;
    std::vector<double> variableK, variableD0;
    std::vector<int> variableForm;
    std::vector<TermData> terms;
    std::vector<std::array<int, 4> > monomials;
};

/**
 * This kernel is invoked by WaterTrimerPIPForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcWaterTrimerPIPForceKernel : public CalcWaterTrimerPIPForceKernel {
public:
    ReferenceCalcWaterTrimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcWaterTrimerPIPForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the WaterTrimerPIPForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const WaterTrimerPIPForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    struct TermData {
        double coefficient;
        int start, count, degree;
    };
    bool periodic;
    int maxNeighbors;
    double switchingDistance, cutoffDistance, maxSafeForce;
    double ooSwitch[2], ohSwitch[2], hhSwitch[2], ohWindow[4], hhWindow[4];
    std::vector<std::array<int, 3> > waters;
    std::vector<std::pair<int, int> > variablePairs;
    std::vector<double> variableK, variableD0, variableCenter;
    std::vector<TermData> terms;
    std::vector<std::array<int, 4> > monomials;
};

/**
 * This kernel is invoked by PartridgeSchwenkeForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcPartridgeSchwenkeForceKernel : public CalcPartridgeSchwenkeForceKernel {
public:
    ReferenceCalcPartridgeSchwenkeForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcPartridgeSchwenkeForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the PartridgeSchwenkeForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const PartridgeSchwenkeForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    bool periodic;
    std::vector<std::array<int, 3> > waters;
    std::vector<double> coefficients;
};

} // namespace SAPTMLPlugin

#endif /*REFERENCE_SAPTML_KERNELS_H_*/
