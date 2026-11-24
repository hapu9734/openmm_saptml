#ifndef COMMON_SAPTML_KERNELS_H_
#define COMMON_SAPTML_KERNELS_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

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
 * Real-space, charge, induction and PME kernels forked from OpenMM's AMOEBA multipole kernels, reduced to point charges
 * and induced dipoles.
 */
class CommonCalcMBPolElectrostaticsForceKernel : public CalcMBPolElectrostaticsForceKernel {
public:
    CommonCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system);
    ~CommonCalcMBPolElectrostaticsForceKernel();
    void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges);
    void copyParametersToContext(OpenMM::ContextImpl& context, const MBPolElectrostaticsForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    // Implemented by the platform subclass.
    virtual void computeFFT(bool forward) = 0;
    virtual bool useFixedPointChargeSpreading() const = 0;
protected:
    class ForceInfo;
    void uploadParameters(const MBPolElectrostaticsForce& force);
    void computeInducedField();
    bool iterateDipoles(int iteration);
    void predictDipoles();
    void finishDipoles(bool includeForces);
    void ensureValid(OpenMM::ContextImpl& context);
    OpenMM::ComputeContext& cc;
    int numAtoms, numWaters, maxInducedIterations, maxPrevDIISDipoles, predictOrder, historyCount;
    int fixedFieldThreads, inducedFieldThreads, electrostaticsThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    double pmeAlpha, inducedEpsilon, sorParameter;
    bool usePME, includeChargeRedistribution, useDIIS, resultsValid;
    MBPolElectrostaticsForce::PolarizationType polarizationType;
    OpenMM::ComputeArray waterParticles, chargeDerivs, damping, atomInfo, polarizability;
    OpenMM::ComputeArray field, fieldPolar, inducedField, inducedFieldPolar, inducedDipole, inducedDipolePolar, potential;
    OpenMM::ComputeArray inducedDipoleErrors, prevDipoles, prevDipolesPolar, prevErrors, diisMatrix, diisCoefficients;
    OpenMM::ComputeArray dipoleHistory, dipolePolarHistory;
    OpenMM::ComputeArray pmeGrid1, pmeGrid2, pmeGridLong, pmeBsplineModuliX, pmeBsplineModuliY, pmeBsplineModuliZ;
    OpenMM::ComputeArray pmePhi, pmePhid, pmePhip, pmePhidp, lastPositions;
    OpenMM::ComputeKernel chargesKernel, applyChargeForcesKernel, recordInducedDipolesKernel, predictKernel, saveHistoryKernel;
    OpenMM::ComputeKernel fixedFieldKernel, inducedFieldKernel, electrostaticsKernel;
    OpenMM::ComputeKernel updateInducedFieldKernel, recordDIISDipolesKernel, buildMatrixKernel, solveMatrixKernel, sorKernel;
    OpenMM::ComputeKernel pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    OpenMM::ComputeKernel pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel;
    OpenMM::ComputeEvent syncEvent;
    static const int PmeOrder = 5;
};

/**
 * The polynomial is evaluated as P = 1/2 y^T A y with y = (1, x_i, x_i x_j), walked as a stream of the entries of A with
 * one warp per 32 dimers (lane = dimer), in double precision.
 */
class CommonCalcWaterDimerPIPForceKernel : public CalcWaterDimerPIPForceKernel {
public:
    CommonCalcWaterDimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcWaterDimerPIPForceKernel(name, platform), cc(cc), hasInitializedKernels(false), executionCount(0) {
    }
    void initialize(const OpenMM::System& system, const WaterDimerPIPForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    class ForceInfo;
    OpenMM::ComputeContext& cc;
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
 * As the dimer kernel, with one warp per 32 trimers, in the precision of the context (single for single and mixed precision).
 */
class CommonCalcWaterTrimerPIPForceKernel : public CalcWaterTrimerPIPForceKernel {
public:
    CommonCalcWaterTrimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcWaterTrimerPIPForceKernel(name, platform), cc(cc), hasInitializedKernels(false), executionCount(0) {
    }
    void initialize(const OpenMM::System& system, const WaterTrimerPIPForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    class ForceInfo;
    OpenMM::ComputeContext& cc;
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
            CalcPartridgeSchwenkeForceKernel(name, platform), cc(cc) {
    }
    void initialize(const OpenMM::System& system, const PartridgeSchwenkeForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    class ForceInfo;
    OpenMM::ComputeContext& cc;
};

} // namespace SAPTMLPlugin

#endif /*COMMON_SAPTML_KERNELS_H_*/
