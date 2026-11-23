#ifndef REFERENCE_SAPTML_KERNELS_H_
#define REFERENCE_SAPTML_KERNELS_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "SAPTMLKernels.h"
#include "ReferenceMBPolElectrostatics.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <array>
#include <utility>
#include <vector>

namespace SAPTMLPlugin {

class ReferenceCalcMBPolElectrostaticsForceKernel : public CalcMBPolElectrostaticsForceKernel {
public:
    ReferenceCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, const OpenMM::System& system);
    ~ReferenceCalcMBPolElectrostaticsForceKernel();
    void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getCharges(OpenMM::ContextImpl& context, std::vector<double>& charges);
    void copyParametersToContext(OpenMM::ContextImpl& context, const MBPolElectrostaticsForce& force);
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

class ReferenceCalcWaterDimerPIPForceKernel : public CalcWaterDimerPIPForceKernel {
public:
    ReferenceCalcWaterDimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcWaterDimerPIPForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const WaterDimerPIPForce& force);
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

class ReferenceCalcWaterTrimerPIPForceKernel : public CalcWaterTrimerPIPForceKernel {
public:
    ReferenceCalcWaterTrimerPIPForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcWaterTrimerPIPForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const WaterTrimerPIPForce& force);
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

class ReferenceCalcPartridgeSchwenkeForceKernel : public CalcPartridgeSchwenkeForceKernel {
public:
    ReferenceCalcPartridgeSchwenkeForceKernel(const std::string& name, const OpenMM::Platform& platform) : CalcPartridgeSchwenkeForceKernel(name, platform) {
    }
    void initialize(const OpenMM::System& system, const PartridgeSchwenkeForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    bool periodic;
    std::vector<std::array<int, 3> > waters;
    std::vector<double> coefficients;
};

} // namespace SAPTMLPlugin

#endif /*REFERENCE_SAPTML_KERNELS_H_*/
