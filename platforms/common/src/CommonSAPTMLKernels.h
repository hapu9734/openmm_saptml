#ifndef COMMON_SAPTML_KERNELS_H_
#define COMMON_SAPTML_KERNELS_H_

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
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeEvent.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace SAPTMLPlugin {

/**
 * This kernel is invoked by MBPolElectrostaticsForce to calculate the forces acting on the system and the energy of the system.
 * The real space, charge and induction kernels are forked from the AMOEBA multipole kernels of OpenMM; the PME machinery of
 * AMOEBA is reused with zero permanent dipoles and quadrupoles.
 */
class CommonCalcMBPolElectrostaticsForceKernel : public CalcMBPolElectrostaticsForceKernel {
public:
    CommonCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system);
    ~CommonCalcMBPolElectrostaticsForceKernel();
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
    /**
     * Compute the FFT of the PME grid.
     */
    virtual void computeFFT(bool forward) = 0;
    /**
     * Get whether to use fixed point charge spreading.
     */
    virtual bool useFixedPointChargeSpreading() const = 0;
protected:
    class ForceInfo;
    void uploadParameters(const MBPolElectrostaticsForce& force);
    void computeInducedField();
    bool iterateDipoles(int iteration);
    void predictDipoles();
    void finishDipoles(bool includeForces);
    void ensureValid(OpenMM::ContextImpl& context);
    int numAtoms, numWaters, maxInducedIterations, maxPrevDIISDipoles, predictOrder, historyCount;
    int fixedFieldThreads, inducedFieldThreads, electrostaticsThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    double pmeAlpha, inducedEpsilon, sorParameter;
    bool usePME, includeChargeRedistribution, useDIIS, resultsValid, hasCreatedEvent;
    MBPolElectrostaticsForce::PolarizationType polarizationType;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    OpenMM::ComputeArray waterParticles, chargeDerivs, damping, atomInfo, polarizability;
    OpenMM::ComputeArray field, fieldPolar, inducedField, inducedFieldPolar, inducedDipole, inducedDipolePolar;
    OpenMM::ComputeArray potential, sitePotential;
    OpenMM::ComputeArray inducedDipoleErrors, prevDipoles, prevDipolesPolar, prevErrors, diisMatrix, diisCoefficients;
    OpenMM::ComputeArray dipoleHistory, dipolePolarHistory;
    OpenMM::ComputeArray zeroDipoles, zeroQuadrupoles, zeroFracDipoles, zeroFracQuadrupoles, torque;
    OpenMM::ComputeArray pmeGrid1, pmeGrid2, pmeGridLong, pmeBsplineModuliX, pmeBsplineModuliY, pmeBsplineModuliZ;
    OpenMM::ComputeArray pmePhi, pmePhid, pmePhip, pmePhidp, pmeCphi, lastPositions;
    OpenMM::ComputeKernel chargesKernel, applyChargeForcesKernel, recordInducedDipolesKernel, predictKernel, saveHistoryKernel;
    OpenMM::ComputeKernel fixedFieldKernel, inducedFieldKernel, electrostaticsKernel;
    OpenMM::ComputeKernel updateInducedFieldKernel, recordDIISDipolesKernel, buildMatrixKernel, solveMatrixKernel, sorKernel;
    OpenMM::ComputeKernel pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    OpenMM::ComputeKernel pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel;
    OpenMM::ComputeEvent syncEvent;
    static const int PmeOrder = 5;
};

/**
 * This kernel is invoked by WaterDimerPIPForce to calculate the forces acting on the system and the energy of the system.
 * The polynomial is evaluated as P = 1/2 y^T A y with y = (1, x_i, x_i x_j), walked as a stream of the entries of A with
 * one warp per 32 dimers (lane = dimer), in double precision.
 */
class CommonCalcWaterDimerPIPForceKernel : public CalcWaterDimerPIPForceKernel {
public:
    CommonCalcWaterDimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcWaterDimerPIPForceKernel(name, platform), cc(cc), system(system), hasInitializedKernels(false), executionCount(0) {
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
    class ForceInfo;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    bool hasInitializedKernels, periodic;
    int numWaters, maxNeighbors, maxDimers, overflowCheckInterval;
    int streamCapacity, streamBatches, numSegments, streamWarps, streamThreads;
    long long executionCount;
    OpenMM::ComputeArray waterAtoms, variablePairs, variableK, variableD0, variableForm;
    OpenMM::ComputeArray originalToCurrent, neighborCounts, neighbors, diagnosticStatus, dimerCount, dimers;
    OpenMM::ComputeArray clusterVars, clusterR, clusterX, streamInfo, streamCoef, segments, partialP, partialG;
    OpenMM::ComputeKernel invertKernel, neighborKernel, dimerKernel, prepareKernel, streamKernel, finishKernel;
};

/**
 * This kernel is invoked by WaterTrimerPIPForce to calculate the forces acting on the system and the energy of the system.
 * The polynomial is evaluated as P = 1/2 y^T A y with y = (1, x_i, x_i x_j), walked as a stream of the entries of A with
 * one warp per 32 trimers (lane = trimer), in the precision of the context (single for single and mixed precision).
 */
class CommonCalcWaterTrimerPIPForceKernel : public CalcWaterTrimerPIPForceKernel {
public:
    CommonCalcWaterTrimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcWaterTrimerPIPForceKernel(name, platform), cc(cc), system(system), hasInitializedKernels(false), executionCount(0) {
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
    class ForceInfo;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
    bool hasInitializedKernels, periodic;
    int numWaters, maxNeighbors, maxTrimers, overflowCheckInterval;
    int streamCapacity, streamBatches, numSegments, streamWarps, streamThreads;
    long long executionCount;
    OpenMM::ComputeArray waterAtoms, variablePairs, variableK, variableD0, variableCenter, oxygenPos;
    OpenMM::ComputeArray originalToCurrent, neighborCounts, neighbors, diagnosticStatus, trimerCount, trimers;
    OpenMM::ComputeArray clusterVars, clusterR, clusterX, streamInfo, streamCoef, segments, partialP, partialG;
    OpenMM::ComputeKernel invertKernel, neighborKernel, trimerKernel, prepareKernel, streamKernel, finishKernel;
};

class CommonCalcPartridgeSchwenkeForceKernel : public CalcPartridgeSchwenkeForceKernel {
public:
    CommonCalcPartridgeSchwenkeForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcPartridgeSchwenkeForceKernel(name, platform), cc(cc), system(system) {
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
    class ForceInfo;
    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;
};

} // namespace SAPTMLPlugin

#endif /*COMMON_SAPTML_KERNELS_H_*/
