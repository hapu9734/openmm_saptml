/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * The electrostatics kernels are forked from the AMOEBA kernels of OpenMM 8.2 (portions copyright (c)
 * 2008-2024 Stanford University and the Authors).
 * MIT License; see LICENSE.
 */

#include "CommonSAPTMLKernels.h"
#include "CommonSAPTMLKernelSources.h"
#include "PIPStream.h"
#include "internal/PartridgeSchwenkeCoefficients.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ComputeProgram.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

// MBPolElectrostaticsForce

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
}

class CommonCalcMBPolElectrostaticsForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const MBPolElectrostaticsForce& force) : force(force) {
        type.assign(4*force.getNumWaters(), -1);
        for (int w = 0; w < force.getNumWaters(); w++) {
            int idx[4];
            force.getWaterParameters(w, idx[0], idx[1], idx[2], idx[3]);
            type[idx[0]] = 0;
            type[idx[1]] = 1;
            type[idx[2]] = 1;
            type[idx[3]] = 2;
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        return type[particle1] == type[particle2];
    }
    int getNumParticleGroups() {
        return force.getNumWaters();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.resize(4);
        force.getWaterParameters(index, particles[0], particles[1], particles[2], particles[3]);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const MBPolElectrostaticsForce& force;
    vector<int> type;
};

CommonCalcMBPolElectrostaticsForceKernel::CommonCalcMBPolElectrostaticsForceKernel(const std::string& name, const Platform& platform, ComputeContext& cc, const System& system) :
        CalcMBPolElectrostaticsForceKernel(name, platform), cc(cc), resultsValid(false) {
}

CommonCalcMBPolElectrostaticsForceKernel::~CommonCalcMBPolElectrostaticsForceKernel() {
}

void CommonCalcMBPolElectrostaticsForceKernel::uploadParameters(const MBPolElectrostaticsForce& force) {
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    double qO, qH, qM, aO, aH, dO, dH, dM;
    force.getFixedCharges(qO, qH, qM);
    force.getPolarizabilities(aO, aH);
    force.getDampingFactors(dO, dH, dM);
    vector<float> dampingVec(paddedNumAtoms, 0.0f), polarVec(paddedNumAtoms, 0.0f);
    vector<mm_int2> infoVec(paddedNumAtoms, mm_int2(0, -1));
    vector<mm_int4> waterVec(numWaters);
    vector<double> chargeVec(numAtoms, 0.0);
    for (int w = 0; w < numWaters; w++) {
        int idx[4];
        force.getWaterParameters(w, idx[0], idx[1], idx[2], idx[3]);
        waterVec[w] = mm_int4(idx[0], idx[1], idx[2], idx[3]);
        double q[4] = {qO, qH, qH, qM};
        double a[4] = {aO, aH, aH, 0.0};
        double d[4] = {dO, dH, dH, dM};
        int type[4] = {0, 1, 1, 2};
        for (int k = 0; k < 4; k++) {
            int i = idx[k];
            chargeVec[i] = q[k];
            polarVec[i] = (float) a[k];
            dampingVec[i] = (float) pow(d[k], 1.0/6.0);
            infoVec[i] = mm_int2(type[k], w);
        }
    }
    damping.upload(dampingVec);
    polarizability.upload(polarVec);
    atomInfo.upload(infoVec);
    if (numWaters > 0)
        waterParticles.upload(waterVec);
    // Charges live in posq.w (as in AmoebaMultipoleForce); overwritten every step with charge redistribution.
    ArrayInterface& posq = cc.getPosq();
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> p;
        posq.download(p);
        for (int i = 0; i < numAtoms; i++)
            p[i].w = chargeVec[i];
        posq.upload(p);
    }
    else {
        vector<mm_float4> p;
        posq.download(p);
        for (int i = 0; i < numAtoms; i++)
            p[i].w = (float) chargeVec[i];
        posq.upload(p);
    }
}

void CommonCalcMBPolElectrostaticsForceKernel::initialize(const System& system, const MBPolElectrostaticsForce& force) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("MBPolElectrostaticsForce does not support using multiple devices");
    numAtoms = cc.getNumAtoms();
    numWaters = force.getNumWaters();
    if (4*numWaters != numAtoms)
        throw OpenMMException("MBPolElectrostaticsForce: Every particle of the System must belong to a water");
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    polarizationType = force.getPolarizationType();
    includeChargeRedistribution = force.getIncludeChargeRedistribution();
    usePME = (force.getNonbondedMethod() == MBPolElectrostaticsForce::PME);

    damping.initialize<float>(cc, paddedNumAtoms, "mbpolDamping");
    polarizability.initialize<float>(cc, paddedNumAtoms, "mbpolPolarizability");
    atomInfo.initialize<mm_int2>(cc, paddedNumAtoms, "mbpolAtomInfo");
    waterParticles.initialize<mm_int4>(cc, numWaters, "mbpolWaterParticles");
    chargeDerivs.initialize(cc, 27*numWaters, elementSize, "mbpolChargeDerivs");
    cc.clearBuffer(chargeDerivs);
    uploadParameters(force);
    field.initialize<long long>(cc, 3*paddedNumAtoms, "mbpolField");
    fieldPolar.initialize<long long>(cc, 3*paddedNumAtoms, "mbpolFieldPolar");
    potential.initialize<long long>(cc, paddedNumAtoms, "mbpolPotential");
    inducedDipole.initialize(cc, 3*paddedNumAtoms, elementSize, "mbpolInducedDipole");
    inducedDipolePolar.initialize(cc, 3*paddedNumAtoms, elementSize, "mbpolInducedDipolePolar");
    lastPositions.initialize(cc, cc.getPosq().getSize(), cc.getPosq().getElementSize(), "mbpolLastPositions");
    cc.addAutoclearBuffer(field);
    cc.addAutoclearBuffer(fieldPolar);
    cc.addAutoclearBuffer(potential);
    if (polarizationType == MBPolElectrostaticsForce::Mutual) {
        maxInducedIterations = force.getMutualInducedMaxIterations();
        inducedEpsilon = force.getMutualInducedTargetEpsilon();
        useDIIS = (force.getInducedDipoleMethod() == MBPolElectrostaticsForce::DIIS);
        maxPrevDIISDipoles = force.getDIISHistorySize();
        sorParameter = force.getSORParameter();
        inducedField.initialize<long long>(cc, 3*paddedNumAtoms, "mbpolInducedField");
        inducedFieldPolar.initialize<long long>(cc, 3*paddedNumAtoms, "mbpolInducedFieldPolar");
        inducedDipoleErrors.initialize(cc, cc.getNumThreadBlocks(), sizeof(mm_float2), "mbpolInducedDipoleErrors");
        prevDipoles.initialize(cc, 3*numAtoms*maxPrevDIISDipoles, elementSize, "mbpolPrevDipoles");
        prevDipolesPolar.initialize(cc, 3*numAtoms*maxPrevDIISDipoles, elementSize, "mbpolPrevDipolesPolar");
        prevErrors.initialize(cc, 3*numAtoms*maxPrevDIISDipoles, elementSize, "mbpolPrevErrors");
        diisMatrix.initialize(cc, maxPrevDIISDipoles*maxPrevDIISDipoles, elementSize, "mbpolDiisMatrix");
        diisCoefficients.initialize(cc, maxPrevDIISDipoles+1, sizeof(float), "mbpolDiisCoefficients");
        syncEvent = cc.createEvent();
    }
    else
        maxInducedIterations = 0;

    // The four sites of a water exclude each other; intramolecular pairs are handled explicitly in the exclusion tiles.
    vector<vector<int> > exclusions(numAtoms);
    for (int w = 0; w < numWaters; w++) {
        int idx[4];
        force.getWaterParameters(w, idx[0], idx[1], idx[2], idx[3]);
        for (int k = 0; k < 4; k++)
            for (int kk = 0; kk < 4; kk++)
                exclusions[idx[k]].push_back(idx[kk]);
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/ComputeContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/ComputeContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }

    double tcc, tcd, tdd, tddOH, tddHH;
    force.getTholeParameters(tcc, tcd, tdd, tddOH, tddHH);
    map<string, string> defines;
    defines["NUM_ATOMS"] = cc.intToString(numAtoms);
    defines["NUM_WATERS"] = cc.intToString(numWaters);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(paddedNumAtoms);
    defines["NUM_BLOCKS"] = cc.intToString(cc.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
    defines["THOLE_CC"] = cc.doubleToString(tcc);
    defines["THOLE_CC_QUARTER"] = cc.doubleToString(pow(tcc, 0.25));
    defines["THOLE_CD"] = cc.doubleToString(tcd);
    defines["THOLE_DD"] = cc.doubleToString(tdd);
    defines["THOLE_DD_OH"] = cc.doubleToString(tddOH);
    defines["THOLE_DD_HH"] = cc.doubleToString(tddHH);
    defines["TILE_SIZE"] = cc.intToString(ComputeContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
    defines["FIRST_EXCLUSION_TILE"] = "0";
    defines["LAST_EXCLUSION_TILE"] = cc.intToString(numExclusionTiles);
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = cc.findLegalFFTDimension(gridSizeX);
            gridSizeY = cc.findLegalFFTDimension(gridSizeY);
            gridSizeZ = cc.findLegalFFTDimension(gridSizeZ);
        }
        else {
            gridSizeX = cc.findLegalFFTDimension(nx);
            gridSizeY = cc.findLegalFFTDimension(ny);
            gridSizeZ = cc.findLegalFFTDimension(nz);
        }
        defines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int maxThreads = max(32, nb.getForceThreadBlockSize());
    double fixedThreadMemory = 8*elementSize+4+4;
    double inducedThreadMemory = 15*elementSize+8;
    double electrostaticsThreadMemory = 14*elementSize+8;
    fixedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(fixedThreadMemory));
    inducedFieldThreads = min(maxThreads, cc.computeThreadBlockSize(inducedThreadMemory));
    electrostaticsThreads = min(maxThreads, cc.computeThreadBlockSize(electrostaticsThreadMemory));

    ComputeProgram program = cc.compileProgram(CommonSAPTMLKernelSources::mbpolCharges, defines);
    chargesKernel = program->createKernel("computeWaterCharges");
    chargesKernel->addArg(cc.getPosq());
    chargesKernel->addArg(waterParticles);
    chargesKernel->addArg(chargeDerivs);
    applyChargeForcesKernel = program->createKernel("applyChargeDerivativeForces");
    applyChargeForcesKernel->addArg(cc.getLongForceBuffer());
    applyChargeForcesKernel->addArg(cc.getPosq());
    applyChargeForcesKernel->addArg(waterParticles);
    applyChargeForcesKernel->addArg(chargeDerivs);
    applyChargeForcesKernel->addArg(potential);
    recordInducedDipolesKernel = program->createKernel("recordInducedDipoles");
    recordInducedDipolesKernel->addArg(field);
    recordInducedDipolesKernel->addArg(fieldPolar);
    recordInducedDipolesKernel->addArg(inducedDipole);
    recordInducedDipolesKernel->addArg(inducedDipolePolar);
    recordInducedDipolesKernel->addArg(polarizability);
    predictOrder = force.getPredictorOrder();
    historyCount = 0;
    if (predictOrder > 0) {
        dipoleHistory.initialize(cc, 3*numAtoms*predictOrder, elementSize, "mbpolDipoleHistory");
        dipolePolarHistory.initialize(cc, 3*numAtoms*predictOrder, elementSize, "mbpolDipolePolarHistory");
        predictKernel = program->createKernel("predictInducedDipoles");
        predictKernel->addArg(inducedDipole);
        predictKernel->addArg(inducedDipolePolar);
        predictKernel->addArg(dipoleHistory);
        predictKernel->addArg(dipolePolarHistory);
        predictKernel->addArg(1);
        saveHistoryKernel = program->createKernel("saveInducedDipoleHistory");
        saveHistoryKernel->addArg(inducedDipole);
        saveHistoryKernel->addArg(inducedDipolePolar);
        saveHistoryKernel->addArg(dipoleHistory);
        saveHistoryKernel->addArg(dipolePolarHistory);
        saveHistoryKernel->addArg(predictOrder);
    }

    defines["THREAD_BLOCK_SIZE"] = cc.intToString(fixedFieldThreads);
    program = cc.compileProgram(CommonSAPTMLKernelSources::mbpolCommon+CommonSAPTMLKernelSources::mbpolFixedField, defines);
    fixedFieldKernel = program->createKernel("computeFixedField");
    fixedFieldKernel->addArg(field);
    fixedFieldKernel->addArg(fieldPolar);
    fixedFieldKernel->addArg(cc.getPosq());
    fixedFieldKernel->addArg(nb.getExclusionTiles());
    fixedFieldKernel->addArg();
    fixedFieldKernel->addArg();
    if (usePME) {
        fixedFieldKernel->addArg(nb.getInteractingTiles());
        fixedFieldKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            fixedFieldKernel->addArg();
        fixedFieldKernel->addArg(nb.getInteractingAtoms());
    }
    fixedFieldKernel->addArg(damping);
    fixedFieldKernel->addArg(atomInfo);

    if (polarizationType == MBPolElectrostaticsForce::Mutual) {
        defines["THREAD_BLOCK_SIZE"] = cc.intToString(inducedFieldThreads);
        defines["MAX_PREV_DIIS_DIPOLES"] = cc.intToString(maxPrevDIISDipoles);
        program = cc.compileProgram(CommonSAPTMLKernelSources::mbpolCommon+CommonSAPTMLKernelSources::mbpolInducedField, defines);
        inducedFieldKernel = program->createKernel("computeInducedField");
        inducedFieldKernel->addArg(inducedField);
        inducedFieldKernel->addArg(inducedFieldPolar);
        inducedFieldKernel->addArg(cc.getPosq());
        inducedFieldKernel->addArg(nb.getExclusionTiles());
        inducedFieldKernel->addArg(inducedDipole);
        inducedFieldKernel->addArg(inducedDipolePolar);
        inducedFieldKernel->addArg();
        inducedFieldKernel->addArg();
        if (usePME) {
            inducedFieldKernel->addArg(nb.getInteractingTiles());
            inducedFieldKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 6; i++)
                inducedFieldKernel->addArg();
            inducedFieldKernel->addArg(nb.getInteractingAtoms());
        }
        inducedFieldKernel->addArg(damping);
        inducedFieldKernel->addArg(atomInfo);
        updateInducedFieldKernel = program->createKernel("updateInducedFieldByDIIS");
        updateInducedFieldKernel->addArg(inducedDipole);
        updateInducedFieldKernel->addArg(inducedDipolePolar);
        updateInducedFieldKernel->addArg(prevDipoles);
        updateInducedFieldKernel->addArg(prevDipolesPolar);
        updateInducedFieldKernel->addArg(diisCoefficients);
        updateInducedFieldKernel->addArg();
        sorKernel = program->createKernel("updateInducedDipolesBySOR");
        sorKernel->addArg(inducedDipole);
        sorKernel->addArg(inducedDipolePolar);
        sorKernel->addArg(prevDipoles);
        sorKernel->addArg(prevDipolesPolar);
        sorKernel->addArg((float) sorParameter);
        recordDIISDipolesKernel = program->createKernel("recordInducedDipolesForDIIS");
        recordDIISDipolesKernel->addArg(field);
        recordDIISDipolesKernel->addArg(fieldPolar);
        recordDIISDipolesKernel->addArg(polarizability);
        recordDIISDipolesKernel->addArg(inducedDipoleErrors);
        recordDIISDipolesKernel->addArg(prevErrors);
        recordDIISDipolesKernel->addArg(diisMatrix);
        recordDIISDipolesKernel->addArg(inducedField);
        recordDIISDipolesKernel->addArg(inducedFieldPolar);
        recordDIISDipolesKernel->addArg(inducedDipole);
        recordDIISDipolesKernel->addArg(inducedDipolePolar);
        recordDIISDipolesKernel->addArg(prevDipoles);
        recordDIISDipolesKernel->addArg(prevDipolesPolar);
        recordDIISDipolesKernel->addArg();
        buildMatrixKernel = program->createKernel("computeDIISMatrix");
        buildMatrixKernel->addArg(prevErrors);
        buildMatrixKernel->addArg();
        buildMatrixKernel->addArg(diisMatrix);
        solveMatrixKernel = program->createKernel("solveDIISMatrix");
        solveMatrixKernel->addArg();
        solveMatrixKernel->addArg(diisMatrix);
        solveMatrixKernel->addArg(diisCoefficients);
    }

    defines["THREAD_BLOCK_SIZE"] = cc.intToString(electrostaticsThreads);
    program = cc.compileProgram(CommonSAPTMLKernelSources::mbpolCommon+CommonSAPTMLKernelSources::mbpolElectrostatics, defines);
    electrostaticsKernel = program->createKernel("computeElectrostatics");
    electrostaticsKernel->addArg(cc.getLongForceBuffer());
    electrostaticsKernel->addArg(potential);
    electrostaticsKernel->addArg(cc.getEnergyBuffer());
    electrostaticsKernel->addArg(cc.getPosq());
    electrostaticsKernel->addArg(nb.getExclusionTiles());
    electrostaticsKernel->addArg();
    electrostaticsKernel->addArg();
    if (usePME) {
        electrostaticsKernel->addArg(nb.getInteractingTiles());
        electrostaticsKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            electrostaticsKernel->addArg();
        electrostaticsKernel->addArg(nb.getInteractingAtoms());
    }
    electrostaticsKernel->addArg(inducedDipole);
    electrostaticsKernel->addArg(inducedDipolePolar);
    electrostaticsKernel->addArg(damping);
    electrostaticsKernel->addArg(atomInfo);

    if (usePME) {
        pmeGrid1.initialize(cc, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "mbpolPmeGrid1");
        pmeGrid2.initialize(cc, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "mbpolPmeGrid2");
        if (useFixedPointChargeSpreading()) {
            pmeGridLong.initialize(cc, 2*gridSizeX*gridSizeY*gridSizeZ, sizeof(long long), "mbpolPmeGridLong");
            cc.addAutoclearBuffer(pmeGridLong);
        }
        else
            cc.addAutoclearBuffer(pmeGrid1);
        pmeBsplineModuliX.initialize(cc, gridSizeX, elementSize, "mbpolPmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cc, gridSizeY, elementSize, "mbpolPmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cc, gridSizeZ, elementSize, "mbpolPmeBsplineModuliZ");
        pmePhi.initialize(cc, 10*numAtoms, elementSize, "mbpolPmePhi");
        pmePhid.initialize(cc, 10*numAtoms, elementSize, "mbpolPmePhid");
        pmePhip.initialize(cc, 10*numAtoms, elementSize, "mbpolPmePhip");
        pmePhidp.initialize(cc, 4*numAtoms, elementSize, "mbpolPmePhidp");
        applyChargeForcesKernel->addArg(pmePhi);
        applyChargeForcesKernel->addArg(pmePhidp);

        map<string, string> pmeDefines;
        pmeDefines["EWALD_ALPHA"] = cc.doubleToString(pmeAlpha);
        pmeDefines["PME_ORDER"] = cc.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cc.intToString(numAtoms);
        pmeDefines["PADDED_NUM_ATOMS"] = cc.intToString(paddedNumAtoms);
        pmeDefines["EPSILON_FACTOR"] = cc.doubleToString(ONE_4PI_EPS0);
        pmeDefines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cc.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cc.doubleToString(sqrt(M_PI));
        if (polarizationType == MBPolElectrostaticsForce::Mutual)
            pmeDefines["MUTUAL_POLARIZATION"] = "";
        if (useFixedPointChargeSpreading())
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "";
        program = cc.compileProgram(CommonSAPTMLKernelSources::mbpolPme, pmeDefines);
        pmeSpreadFixedMultipolesKernel = program->createKernel("gridSpreadFixedMultipoles");
        pmeSpreadFixedMultipolesKernel->addArg(cc.getPosq());
        if (useFixedPointChargeSpreading())
            pmeSpreadFixedMultipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadFixedMultipolesKernel->addArg(pmeGrid1);
        for (int i = 0; i < 6; i++)
            pmeSpreadFixedMultipolesKernel->addArg();
        pmeSpreadInducedDipolesKernel = program->createKernel("gridSpreadInducedDipoles");
        pmeSpreadInducedDipolesKernel->addArg(cc.getPosq());
        pmeSpreadInducedDipolesKernel->addArg(inducedDipole);
        pmeSpreadInducedDipolesKernel->addArg(inducedDipolePolar);
        if (useFixedPointChargeSpreading())
            pmeSpreadInducedDipolesKernel->addArg(pmeGridLong);
        else
            pmeSpreadInducedDipolesKernel->addArg(pmeGrid1);
        for (int i = 0; i < 6; i++)
            pmeSpreadInducedDipolesKernel->addArg();
        if (useFixedPointChargeSpreading()) {
            pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
            pmeFinishSpreadChargeKernel->addArg(pmeGridLong);
            pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
        }
        pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
        pmeConvolutionKernel->addArg(pmeGrid2);
        pmeConvolutionKernel->addArg(pmeBsplineModuliX);
        pmeConvolutionKernel->addArg(pmeBsplineModuliY);
        pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
        for (int i = 0; i < 4; i++)
            pmeConvolutionKernel->addArg();
        pmeFixedPotentialKernel = program->createKernel("computeFixedPotentialFromGrid");
        pmeFixedPotentialKernel->addArg(pmeGrid1);
        pmeFixedPotentialKernel->addArg(pmePhi);
        pmeFixedPotentialKernel->addArg(field);
        pmeFixedPotentialKernel->addArg(fieldPolar);
        pmeFixedPotentialKernel->addArg(cc.getPosq());
        for (int i = 0; i < 6; i++)
            pmeFixedPotentialKernel->addArg();
        pmeInducedPotentialKernel = program->createKernel("computeInducedPotentialFromGrid");
        pmeInducedPotentialKernel->addArg(pmeGrid1);
        pmeInducedPotentialKernel->addArg(pmePhid);
        pmeInducedPotentialKernel->addArg(pmePhip);
        pmeInducedPotentialKernel->addArg(pmePhidp);
        pmeInducedPotentialKernel->addArg(cc.getPosq());
        for (int i = 0; i < 6; i++)
            pmeInducedPotentialKernel->addArg();
        pmeFixedForceKernel = program->createKernel("computeFixedMultipoleForceAndEnergy");
        pmeFixedForceKernel->addArg(cc.getPosq());
        pmeFixedForceKernel->addArg(cc.getLongForceBuffer());
        pmeFixedForceKernel->addArg(cc.getEnergyBuffer());
        pmeFixedForceKernel->addArg(pmePhi);
        for (int i = 0; i < 3; i++)
            pmeFixedForceKernel->addArg();
        pmeInducedForceKernel = program->createKernel("computeInducedDipoleForceAndEnergy");
        pmeInducedForceKernel->addArg(cc.getPosq());
        pmeInducedForceKernel->addArg(cc.getLongForceBuffer());
        pmeInducedForceKernel->addArg(cc.getEnergyBuffer());
        pmeInducedForceKernel->addArg(inducedDipole);
        pmeInducedForceKernel->addArg(inducedDipolePolar);
        pmeInducedForceKernel->addArg(pmePhi);
        pmeInducedForceKernel->addArg(pmePhid);
        pmeInducedForceKernel->addArg(pmePhip);
        pmeInducedForceKernel->addArg(pmePhidp);
        for (int i = 0; i < 3; i++)
            pmeInducedForceKernel->addArg();
        if (polarizationType == MBPolElectrostaticsForce::Mutual) {
            pmeRecordInducedFieldDipolesKernel = program->createKernel("recordInducedFieldDipoles");
            pmeRecordInducedFieldDipolesKernel->addArg(pmePhid);
            pmeRecordInducedFieldDipolesKernel->addArg(pmePhip);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedField);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedFieldPolar);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipole);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipolePolar);
            for (int i = 0; i < 3; i++)
                pmeRecordInducedFieldDipolesKernel->addArg();
        }

        // B-spline moduli, as in AmoebaMultipoleForce.
        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);
            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }
            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];
            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cc.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX.upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(moduli);
                else
                    pmeBsplineModuliZ.upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX.upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(modulif);
                else
                    pmeBsplineModuliZ.upload(modulif);
            }
        }
    }

    // Registered (without source) so that NonbondedUtilities builds the exclusion tiles and neighbor list.
    nb.addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    nb.setUsePadding(false);
    cc.addForce(new ForceInfo(force));
}

double CommonCalcMBPolElectrostaticsForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(cc);
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    if (includeChargeRedistribution && numWaters > 0)
        chargesKernel->execute(numWaters);
    fixedFieldKernel->setArg(4, startTileIndex);
    fixedFieldKernel->setArg(5, numTileIndices);
    electrostaticsKernel->setArg(5, startTileIndex);
    electrostaticsKernel->setArg(6, numTileIndices);
    if (!usePME) {
        fixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(numAtoms);
        predictDipoles();
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField();
            if (iterateDipoles(i))
                break;
        }
        finishDipoles(includeForces);
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
    }
    else {
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double determinant = a[0]*b[1]*c[2];
        double scale = 1.0/determinant;
        mm_double4 recipBoxVectors[3];
        recipBoxVectors[0] = mm_double4(b[1]*c[2]*scale, 0, 0, 0);
        recipBoxVectors[1] = mm_double4(-b[0]*c[2]*scale, a[0]*c[2]*scale, 0, 0);
        recipBoxVectors[2] = mm_double4((b[0]*c[1]-b[1]*c[0])*scale, -a[0]*c[1]*scale, a[0]*b[1]*scale, 0);
        if (cc.getUseDoublePrecision()) {
            mm_double4 boxVectors[] = {mm_double4(a[0], a[1], a[2], 0), mm_double4(b[0], b[1], b[2], 0), mm_double4(c[0], c[1], c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_double4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeSpreadFixedMultipolesKernel->setArg(2+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(5+i, recipBoxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectors[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectors[i]);
                pmeFixedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeFixedForceKernel->setArg(4+i, recipBoxVectors[i]);
                pmeInducedForceKernel->setArg(9+i, recipBoxVectors[i]);
                if (polarizationType == MBPolElectrostaticsForce::Mutual)
                    pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectors[i]);
            }
        }
        else {
            mm_float4 recipBoxVectorsFloat[3];
            recipBoxVectorsFloat[0] = mm_float4((float) recipBoxVectors[0].x, 0, 0, 0);
            recipBoxVectorsFloat[1] = mm_float4((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0, 0);
            recipBoxVectorsFloat[2] = mm_float4((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z, 0);
            mm_float4 boxVectors[] = {mm_float4((float) a[0], (float) a[1], (float) a[2], 0), mm_float4((float) b[0], (float) b[1], (float) b[2], 0), mm_float4((float) c[0], (float) c[1], (float) c[2], 0)};
            pmeConvolutionKernel->setArg(4, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeSpreadFixedMultipolesKernel->setArg(2+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(5+i, recipBoxVectorsFloat[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectorsFloat[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectorsFloat[i]);
                pmeFixedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeFixedForceKernel->setArg(4+i, recipBoxVectorsFloat[i]);
                pmeInducedForceKernel->setArg(9+i, recipBoxVectorsFloat[i]);
                if (polarizationType == MBPolElectrostaticsForce::Mutual)
                    pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectorsFloat[i]);
            }
        }

        // Reciprocal-space permanent charges, then the direct-space fixed field, then the reciprocal induced field of
        // the initial dipoles.
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        pmeSpreadFixedMultipolesKernel->execute(numAtoms);
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        computeFFT(true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeFFT(false);
        pmeFixedPotentialKernel->execute(numAtoms);
        pmeFixedForceKernel->execute(numAtoms);

        setPeriodicBoxArgs(cc, fixedFieldKernel, 8);
        fixedFieldKernel->setArg(13, maxTiles);
        fixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(numAtoms);
        predictDipoles();
        if (useFixedPointChargeSpreading())
            cc.clearBuffer(pmeGridLong);
        else
            cc.clearBuffer(pmeGrid1);
        pmeSpreadInducedDipolesKernel->execute(numAtoms);
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        computeFFT(true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeFFT(false);
        pmeInducedPotentialKernel->execute(numAtoms);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField();
            if (iterateDipoles(i))
                break;
        }
        finishDipoles(includeForces);
        setPeriodicBoxArgs(cc, electrostaticsKernel, 9);
        electrostaticsKernel->setArg(14, maxTiles);
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        pmeInducedForceKernel->execute(numAtoms);
    }
    if (includeChargeRedistribution && numWaters > 0)
        applyChargeForcesKernel->execute(numWaters);
    cc.getPosq().copyTo(lastPositions);
    resultsValid = true;
    return 0.0;
}

void CommonCalcMBPolElectrostaticsForceKernel::predictDipoles() {
    if (predictOrder > 0 && historyCount > 0 && maxInducedIterations > 0) {
        predictKernel->setArg(4, min(predictOrder, historyCount));
        predictKernel->execute(3*numAtoms);
    }
}

void CommonCalcMBPolElectrostaticsForceKernel::finishDipoles(bool includeForces) {
    // Only force evaluations enter the predictor history, not energy-only ones (e.g. barostat trials).
    if (predictOrder > 0 && includeForces && maxInducedIterations > 0) {
        saveHistoryKernel->execute(3*numAtoms);
        historyCount = min(historyCount+1, predictOrder);
    }
}

void CommonCalcMBPolElectrostaticsForceKernel::computeInducedField() {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    inducedFieldKernel->setArg(6, nb.getStartTileIndex());
    inducedFieldKernel->setArg(7, nb.getNumTiles());
    if (usePME) {
        setPeriodicBoxArgs(cc, inducedFieldKernel, 10);
        inducedFieldKernel->setArg(15, (int) nb.getInteractingTiles().getSize());
    }
    cc.clearBuffer(inducedField);
    cc.clearBuffer(inducedFieldPolar);
    inducedFieldKernel->execute(nb.getNumForceThreadBlocks()*inducedFieldThreads, inducedFieldThreads);
    if (usePME) {
        if (useFixedPointChargeSpreading())
            cc.clearBuffer(pmeGridLong);
        else
            cc.clearBuffer(pmeGrid1);
        pmeSpreadInducedDipolesKernel->execute(numAtoms);
        if (useFixedPointChargeSpreading())
            pmeFinishSpreadChargeKernel->execute(pmeGrid1.getSize());
        computeFFT(true);
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeFFT(false);
        pmeInducedPotentialKernel->execute(numAtoms);
        pmeRecordInducedFieldDipolesKernel->execute(numAtoms);
    }
}

bool CommonCalcMBPolElectrostaticsForceKernel::iterateDipoles(int iteration) {
    // Records the Jacobi update alpha (E_fixed + E_induced) and its change from the current dipoles (RMS = convergence
    // measure) in DIIS slot `iteration` (slot 0 without DIIS), then replaces the dipoles by the DIIS extrapolation or,
    // without DIIS, by the SOR mix of the current and Jacobi dipoles.
    int slot = useDIIS ? iteration : 0;
    recordDIISDipolesKernel->setArg(12, slot);
    recordDIISDipolesKernel->execute(cc.getNumThreadBlocks()*64, 64);
    mm_float2* errors = (mm_float2*) cc.getPinnedBuffer();
    inducedDipoleErrors.download(errors, false);
    syncEvent->enqueue();
    int numPrev = (slot+1 < maxPrevDIISDipoles ? slot+1 : maxPrevDIISDipoles);
    if (useDIIS) {
        int threadBlocks = min(numPrev, cc.getNumThreadBlocks());
        int blockSize = min(512, buildMatrixKernel->getMaxBlockSize());
        buildMatrixKernel->setArg(1, iteration);
        buildMatrixKernel->execute(threadBlocks*blockSize, blockSize);
        solveMatrixKernel->setArg(0, iteration);
        solveMatrixKernel->execute(32, 32);
    }
    syncEvent->wait();
    double total1 = 0.0, total2 = 0.0;
    for (int j = 0; j < inducedDipoleErrors.getSize(); j++) {
        total1 += errors[j].x;
        total2 += errors[j].y;
    }
    if (48.033324*sqrt(max(total1, total2)/numAtoms) < inducedEpsilon)
        return true;
    if (useDIIS) {
        updateInducedFieldKernel->setArg(5, numPrev);
        updateInducedFieldKernel->execute(3*numAtoms, 256);
    }
    else
        sorKernel->execute(3*numAtoms, 256);
    return false;
}

void CommonCalcMBPolElectrostaticsForceKernel::ensureValid(ContextImpl& context) {
    if (resultsValid) {
        int numParticles = cc.getNumAtoms();
        if (cc.getUseDoublePrecision()) {
            vector<mm_double4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    resultsValid = false;
                    break;
                }
        }
        else {
            vector<mm_float4> pos1, pos2;
            cc.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    resultsValid = false;
                    break;
                }
        }
    }
    if (!resultsValid)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
}

void CommonCalcMBPolElectrostaticsForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ContextSelector selector(cc);
    ensureValid(context);
    dipoles.resize(numAtoms);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole.download(d);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole.download(d);
        for (int i = 0; i < numAtoms; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}

void CommonCalcMBPolElectrostaticsForceKernel::getCharges(ContextImpl& context, vector<double>& charges) {
    ContextSelector selector(cc);
    ensureValid(context);
    charges.resize(numAtoms);
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision()) {
        vector<mm_double4> p;
        cc.getPosq().download(p);
        for (int i = 0; i < numAtoms; i++)
            charges[order[i]] = p[i].w;
    }
    else {
        vector<mm_float4> p;
        cc.getPosq().download(p);
        for (int i = 0; i < numAtoms; i++)
            charges[order[i]] = p[i].w;
    }
}

void CommonCalcMBPolElectrostaticsForceKernel::copyParametersToContext(ContextImpl& context, const MBPolElectrostaticsForce& force) {
    ContextSelector selector(cc);
    if (4*force.getNumWaters() != numAtoms)
        throw OpenMMException("updateParametersInContext: The number of waters has changed");
    uploadParameters(force);
    cc.invalidateMolecules();
    resultsValid = false;
}

void CommonCalcMBPolElectrostaticsForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

// WaterDimerPIPForce and WaterTrimerPIPForce

// Expands every term into its monomials (variable indices, term coefficient) for PIPStream::build.
static PIPStream::Tables buildStreamTables(const vector<mm_int4>& monomials, const vector<mm_int4>& terms, const vector<double>& coefficients,
        int numVariables, int numSegments) {
    vector<vector<int> > monoVars;
    vector<double> monoCoef;
    for (int term = 0; term < (int) terms.size(); term++)
        for (int m = terms[term].x; m < terms[term].x+terms[term].y; m++) {
            int raw[4] = {monomials[m].x, monomials[m].y, monomials[m].z, monomials[m].w};
            monoVars.push_back(vector<int>(raw, raw+terms[term].z));
            monoCoef.push_back(coefficients[term]);
        }
    return PIPStream::build(monoVars, monoCoef, numVariables, numSegments);
}

static int monomialDegree(const mm_int4& monomial) {
    int degree = 0;
    const int v[4] = {monomial.x, monomial.y, monomial.z, monomial.w};
    while (degree < 4 && v[degree] != -1)
        degree++;
    return degree;
}

static void checkClusterDiagnostics(ComputeArray& diagnosticStatus, int capacity, int maxNeighbors, const string& name, const string& clusters) {
    vector<int> value(3);
    diagnosticStatus.download(value);
    if (value[2] != 0) {
        stringstream message;
        message << name << ": More than " << capacity << " " << clusters << "; increase maxNeighbors";
        throw OpenMMException(message.str());
    }
    if (value[0] != 0) {
        stringstream message;
        message << name << ": At least one water has more than " << maxNeighbors << " neighbors inside the cutoff; increase maxNeighbors";
        throw OpenMMException(message.str());
    }
    if (value[1] != 0)
        throw OpenMMException(name+": A non-finite or unsafe value was detected in the device kernel");
}

class CommonCalcWaterDimerPIPForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const WaterDimerPIPForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumWaters();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.resize(3);
        force.getWaterParameters(index, particles[0], particles[1], particles[2]);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const WaterDimerPIPForce& force;
};

void CommonCalcWaterDimerPIPForceKernel::initialize(const System& system, const WaterDimerPIPForce& force) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("WaterDimerPIPForce does not support using multiple devices");
    numWaters = force.getNumWaters();
    maxNeighbors = force.getMaxNeighbors();
    overflowCheckInterval = force.getOverflowCheckInterval();
    periodic = (force.getNonbondedMethod() == WaterDimerPIPForce::CutoffPeriodic);
    long long capacity = (long long) numWaters*maxNeighbors;
    if (capacity > INT_MAX)
        throw OpenMMException("WaterDimerPIPForce: The number of dimers exceeds 32-bit indexing");
    maxDimers = (int) capacity;

    vector<mm_int4> waterVector(numWaters);
    for (int i = 0; i < numWaters; i++) {
        int oxygen, hydrogen1, hydrogen2;
        force.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        waterVector[i] = mm_int4(oxygen, hydrogen1, hydrogen2, 0);
    }
    waterAtoms.initialize<mm_int4>(cc, numWaters, "waterDimerPIPWaterAtoms");
    waterAtoms.upload(waterVector);
    int numVariables = force.getNumVariables();
    vector<mm_int2> pairVector(numVariables);
    vector<double> kVector(numVariables), d0Vector(numVariables);
    vector<int> formVector(numVariables);
    for (int i = 0; i < numVariables; i++) {
        int site1, site2;
        WaterDimerPIPForce::VariableForm form;
        force.getVariableParameters(i, site1, site2, kVector[i], d0Vector[i], form);
        pairVector[i] = mm_int2(site1, site2);
        formVector[i] = form;
    }
    variablePairs.initialize<mm_int2>(cc, numVariables, "waterDimerPIPVariablePairs");
    variablePairs.upload(pairVector);
    variableK.initialize<double>(cc, numVariables, "waterDimerPIPVariableK");
    variableK.upload(kVector);
    variableD0.initialize<double>(cc, numVariables, "waterDimerPIPVariableD0");
    variableD0.upload(d0Vector);
    variableForm.initialize<int>(cc, numVariables, "waterDimerPIPVariableForm");
    variableForm.upload(formVector);

    // Stream tables of A (P = 1/2 y^T A y), always in double precision.
    int numMonomials = force.getNumMonomials();
    vector<mm_int4> monomialVector(numMonomials);
    for (int i = 0; i < numMonomials; i++)
        force.getMonomialParameters(i, monomialVector[i].x, monomialVector[i].y, monomialVector[i].z, monomialVector[i].w);
    int numTerms = force.getNumTerms();
    vector<mm_int4> termVector(numTerms);
    vector<double> coefficientVector(numTerms);
    for (int i = 0; i < numTerms; i++) {
        int start, count;
        force.getTermParameters(i, coefficientVector[i], start, count);
        termVector[i] = mm_int4(start, count, monomialDegree(monomialVector[start]), 0);
    }
    streamCapacity = maxDimers;
    streamBatches = (streamCapacity+31)/32;
    double batches = max(1.0, numWaters*20.0/32.0);
    int segments = (int) max(1.0, min(32.0, floor(4096.0/batches+0.5)));
    PIPStream::Tables tables = buildStreamTables(monomialVector, termVector, coefficientVector, numVariables, segments);
    numSegments = tables.segStart.size();
    streamWarps = 4;
    streamInfo.initialize<unsigned int>(cc, tables.info.size(), "waterDimerPIPStreamInfo");
    streamInfo.upload(tables.info);
    streamCoef.initialize<double>(cc, tables.coef.size(), "waterDimerPIPStreamCoef");
    streamCoef.upload(tables.coef);
    vector<mm_int4> segmentVector;
    for (int i = 0; i < numSegments; i++)
        segmentVector.push_back(mm_int4(tables.segStart[i], tables.segEnd[i], tables.segFirstGroup[i], tables.segRows[i]));
    this->segments.initialize<mm_int4>(cc, numSegments, "waterDimerPIPSegments");
    this->segments.upload(segmentVector);
    clusterVars.initialize<double>(cc, (size_t) streamBatches*32*(numVariables+1), "waterDimerPIPClusterVars");
    clusterR.initialize<float>(cc, (size_t) streamBatches*32*numVariables, "waterDimerPIPClusterR");
    clusterX.initialize<float>(cc, (size_t) streamBatches*32*numVariables, "waterDimerPIPClusterX");
    partialP.initialize<double>(cc, (size_t) numSegments*streamBatches*32, "waterDimerPIPPartialP");
    partialG.initialize<double>(cc, (size_t) numSegments*streamBatches*32*numVariables, "waterDimerPIPPartialG");
    streamThreads = 32*streamWarps*1024;

    originalToCurrent.initialize<int>(cc, system.getNumParticles(), "waterDimerPIPOriginalToCurrent");
    neighborCounts.initialize<int>(cc, numWaters, "waterDimerPIPNeighborCounts");
    neighbors.initialize<int>(cc, (size_t) numWaters*maxNeighbors, "waterDimerPIPNeighbors");
    diagnosticStatus.initialize<int>(cc, 3, "waterDimerPIPDiagnosticStatus");
    cc.clearBuffer(diagnosticStatus);
    dimerCount.initialize<int>(cc, 1, "waterDimerPIPDimerCount");
    dimers.initialize<mm_int2>(cc, maxDimers, "waterDimerPIPDimers");
    cc.addAutoclearBuffer(dimerCount);

    double zero, full, inPlane, outOfPlane, window[4];
    map<string, string> defines;
    if (periodic)
        defines["USE_PERIODIC"] = "1";
    defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_WATERS"] = cc.intToString(numWaters);
    defines["NUM_VARIABLES"] = cc.intToString(numVariables);
    defines["STREAM_REAL"] = "double";
    defines["STREAM_CAPACITY"] = cc.intToString(streamCapacity);
    defines["STREAM_BATCHES"] = cc.intToString(streamBatches);
    defines["STREAM_CAPACITY_PADDED"] = cc.intToString(streamBatches*32);
    defines["NUM_SEGMENTS"] = cc.intToString(numSegments);
    defines["STREAM_WARPS"] = cc.intToString(streamWarps);
    defines["STREAM_CHUNK"] = "512";
    defines["FINISH_WARPS"] = "4";
    defines["MAX_NEIGHBORS"] = cc.intToString(maxNeighbors);
    defines["MAX_DIMERS"] = cc.intToString(maxDimers);
    defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    defines["SWITCH_ON"] = cc.doubleToString(force.getSwitchingDistance());
    defines["SWITCH_CUT"] = cc.doubleToString(force.getCutoffDistance());
    force.getOxygenOxygenSwitch(zero, full);
    defines["OO_SHORT_ZERO"] = cc.doubleToString(zero);
    defines["OO_SHORT_FULL"] = cc.doubleToString(full);
    force.getIntermolecularOHSwitch(zero, full);
    defines["OH_INTER_ZERO"] = cc.doubleToString(zero);
    defines["OH_INTER_FULL"] = cc.doubleToString(full);
    force.getIntermolecularHHSwitch(zero, full);
    defines["HH_INTER_ZERO"] = cc.doubleToString(zero);
    defines["HH_INTER_FULL"] = cc.doubleToString(full);
    force.getIntramolecularOHWindow(window[0], window[1], window[2], window[3]);
    defines["OH_LOWER_ZERO"] = cc.doubleToString(window[0]);
    defines["OH_LOWER_FULL"] = cc.doubleToString(window[1]);
    defines["OH_UPPER_FULL"] = cc.doubleToString(window[2]);
    defines["OH_UPPER_ZERO"] = cc.doubleToString(window[3]);
    force.getIntramolecularHHWindow(window[0], window[1], window[2], window[3]);
    defines["HH_LOWER_ZERO"] = cc.doubleToString(window[0]);
    defines["HH_LOWER_FULL"] = cc.doubleToString(window[1]);
    defines["HH_UPPER_FULL"] = cc.doubleToString(window[2]);
    defines["HH_UPPER_ZERO"] = cc.doubleToString(window[3]);
    defines["MAX_SAFE_FORCE"] = cc.doubleToString(min(1.0e6, 0.25*2147483647.0/(1.5*maxNeighbors)));
    force.getExtraPointParameters(inPlane, outOfPlane);
    defines["GAMMA_IN"] = cc.doubleToString(inPlane);
    defines["GAMMA_OUT"] = cc.doubleToString(outOfPlane);
    defines["M_PI"] = cc.doubleToString(M_PI);
    ComputeProgram program = cc.compileProgram(CommonSAPTMLKernelSources::waterDimerPIP, defines);
    invertKernel = program->createKernel("invertAtomOrder");
    neighborKernel = program->createKernel("findOxygenNeighbors");
    dimerKernel = program->createKernel("buildWaterDimers");
    prepareKernel = program->createKernel("wdpipPrepare");
    streamKernel = program->createKernel("wdpipStream");
    finishKernel = program->createKernel("wdpipFinish");
    cc.addForce(new ForceInfo(force));
}

double CommonCalcWaterDimerPIPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return 0.0;
    ContextSelector selector(cc);
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        invertKernel->addArg(cc.getAtomIndexArray());
        invertKernel->addArg(originalToCurrent);
        neighborKernel->addArg(originalToCurrent);
        neighborKernel->addArg(waterAtoms);
        neighborKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            neighborKernel->addArg();
        neighborKernel->addArg(neighborCounts);
        neighborKernel->addArg(neighbors);
        neighborKernel->addArg(diagnosticStatus);
        dimerKernel->addArg(neighborCounts);
        dimerKernel->addArg(neighbors);
        dimerKernel->addArg(dimerCount);
        dimerKernel->addArg(dimers);
        prepareKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            prepareKernel->addArg();
        prepareKernel->addArg(originalToCurrent);
        prepareKernel->addArg(waterAtoms);
        prepareKernel->addArg(dimerCount);
        prepareKernel->addArg(dimers);
        prepareKernel->addArg(variablePairs);
        prepareKernel->addArg(variableK);
        prepareKernel->addArg(variableD0);
        prepareKernel->addArg(variableForm);
        prepareKernel->addArg(clusterVars);
        prepareKernel->addArg(clusterR);
        prepareKernel->addArg(clusterX);
        prepareKernel->addArg(diagnosticStatus);
        streamKernel->addArg(dimerCount);
        streamKernel->addArg(clusterVars);
        streamKernel->addArg(streamInfo);
        streamKernel->addArg(streamCoef);
        streamKernel->addArg(segments);
        streamKernel->addArg(partialP);
        streamKernel->addArg(partialG);
        streamKernel->addArg((int) includeForces);
        finishKernel->addArg(cc.getLongForceBuffer());
        finishKernel->addArg(cc.getEnergyBuffer());
        finishKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            finishKernel->addArg();
        finishKernel->addArg(originalToCurrent);
        finishKernel->addArg(waterAtoms);
        finishKernel->addArg(dimerCount);
        finishKernel->addArg(dimers);
        finishKernel->addArg(variablePairs);
        finishKernel->addArg(variableK);
        finishKernel->addArg(variableForm);
        finishKernel->addArg(clusterR);
        finishKernel->addArg(clusterX);
        finishKernel->addArg(partialP);
        finishKernel->addArg(partialG);
        finishKernel->addArg(diagnosticStatus);
        finishKernel->addArg((int) includeForces);
        finishKernel->addArg((int) includeEnergy);
    }
    setPeriodicBoxArgs(cc, neighborKernel, 3);
    setPeriodicBoxArgs(cc, prepareKernel, 1);
    setPeriodicBoxArgs(cc, finishKernel, 3);
    streamKernel->setArg(7, (int) includeForces);
    finishKernel->setArg(20, (int) includeForces);
    finishKernel->setArg(21, (int) includeEnergy);
    invertKernel->execute(cc.getNumAtoms());
    neighborKernel->execute((numWaters+127)/128*128, 128);
    dimerKernel->execute(numWaters);
    int finishThreads = min(32*min(maxDimers, 8192), 32*16384);
    prepareKernel->execute(32*min(streamBatches*32, 16384), 128);
    streamKernel->execute(streamThreads, 32*streamWarps);
    finishKernel->execute(max(128, finishThreads/128*128), 128);
    executionCount++;
    if (executionCount == 1 || executionCount%overflowCheckInterval == 0)
        checkClusterDiagnostics(diagnosticStatus, streamCapacity, maxNeighbors, "WaterDimerPIPForce", "dimers");
    return 0.0;
}

class CommonCalcWaterTrimerPIPForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const WaterTrimerPIPForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumWaters();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.resize(3);
        force.getWaterParameters(index, particles[0], particles[1], particles[2]);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const WaterTrimerPIPForce& force;
};

void CommonCalcWaterTrimerPIPForceKernel::initialize(const System& system, const WaterTrimerPIPForce& force) {
    ContextSelector selector(cc);
    if (cc.getNumContexts() > 1)
        throw OpenMMException("WaterTrimerPIPForce does not support using multiple devices");
    numWaters = force.getNumWaters();
    maxNeighbors = force.getMaxNeighbors();
    overflowCheckInterval = force.getOverflowCheckInterval();
    periodic = (force.getNonbondedMethod() == WaterTrimerPIPForce::CutoffPeriodic);
    long long capacity = (long long) numWaters*maxNeighbors*(maxNeighbors-1)/2;
    if (capacity > INT_MAX)
        throw OpenMMException("WaterTrimerPIPForce: The number of trimers exceeds 32-bit indexing");
    maxTrimers = (int) capacity;

    vector<mm_int4> waterVector(numWaters);
    for (int i = 0; i < numWaters; i++) {
        int oxygen, hydrogen1, hydrogen2;
        force.getWaterParameters(i, oxygen, hydrogen1, hydrogen2);
        waterVector[i] = mm_int4(oxygen, hydrogen1, hydrogen2, 0);
    }
    waterAtoms.initialize<mm_int4>(cc, numWaters, "waterTrimerPIPWaterAtoms");
    waterAtoms.upload(waterVector);
    int numVariables = force.getNumVariables();
    vector<mm_int2> pairVector(numVariables);
    vector<double> kVector(numVariables), d0Vector(numVariables), centerVector(numVariables);
    for (int i = 0; i < numVariables; i++) {
        int site1, site2;
        force.getVariableParameters(i, site1, site2, kVector[i], d0Vector[i], centerVector[i]);
        pairVector[i] = mm_int2(site1, site2);
    }
    variablePairs.initialize<mm_int2>(cc, numVariables, "waterTrimerPIPVariablePairs");
    variablePairs.upload(pairVector);
    variableK.initialize<double>(cc, numVariables, "waterTrimerPIPVariableK");
    variableK.upload(kVector);
    variableD0.initialize<double>(cc, numVariables, "waterTrimerPIPVariableD0");
    variableD0.upload(d0Vector);
    variableCenter.initialize<double>(cc, numVariables, "waterTrimerPIPVariableCenter");
    variableCenter.upload(centerVector);

    // Stream tables of A (P = 1/2 y^T A y), in the precision of the context.
    int numMonomials = force.getNumMonomials();
    vector<mm_int4> monomialVector(numMonomials);
    for (int i = 0; i < numMonomials; i++)
        force.getMonomialParameters(i, monomialVector[i].x, monomialVector[i].y, monomialVector[i].z, monomialVector[i].w);
    int numTerms = force.getNumTerms();
    vector<mm_int4> termVector(numTerms);
    vector<double> coefficientVector(numTerms);
    for (int i = 0; i < numTerms; i++) {
        int start, count;
        force.getTermParameters(i, coefficientVector[i], start, count);
        termVector[i] = mm_int4(start, count, monomialDegree(monomialVector[start]), 0);
    }
    streamCapacity = (int) min((long long) maxTrimers, (long long) numWaters*3*maxNeighbors);
    streamBatches = (streamCapacity+31)/32;
    double batches = max(1.0, numWaters*52.0/32.0);
    int segments = (int) max(1.0, min(32.0, floor(4096.0/batches+0.5)));
    PIPStream::Tables tables = buildStreamTables(monomialVector, termVector, coefficientVector, numVariables, segments);
    numSegments = tables.segStart.size();
    bool useDouble = cc.getUseDoublePrecision();
    streamWarps = (useDouble ? 4 : 8);
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    streamInfo.initialize<unsigned int>(cc, tables.info.size(), "waterTrimerPIPStreamInfo");
    streamInfo.upload(tables.info);
    streamCoef.initialize(cc, tables.coef.size(), elementSize, "waterTrimerPIPStreamCoef");
    streamCoef.upload(tables.coef, true);
    vector<mm_int4> segmentVector;
    for (int i = 0; i < numSegments; i++)
        segmentVector.push_back(mm_int4(tables.segStart[i], tables.segEnd[i], tables.segFirstGroup[i], tables.segRows[i]));
    this->segments.initialize<mm_int4>(cc, numSegments, "waterTrimerPIPSegments");
    this->segments.upload(segmentVector);
    clusterVars.initialize(cc, (size_t) streamBatches*32*(numVariables+1), elementSize, "waterTrimerPIPClusterVars");
    clusterR.initialize<float>(cc, (size_t) streamBatches*32*numVariables, "waterTrimerPIPClusterR");
    clusterX.initialize<float>(cc, (size_t) streamBatches*32*numVariables, "waterTrimerPIPClusterX");
    partialP.initialize(cc, (size_t) numSegments*streamBatches*32, elementSize, "waterTrimerPIPPartialP");
    partialG.initialize(cc, (size_t) numSegments*streamBatches*32*numVariables, elementSize, "waterTrimerPIPPartialG");
    streamThreads = 32*streamWarps*1024;

    originalToCurrent.initialize<int>(cc, system.getNumParticles(), "waterTrimerPIPOriginalToCurrent");
    neighborCounts.initialize<int>(cc, numWaters, "waterTrimerPIPNeighborCounts");
    neighbors.initialize<int>(cc, (size_t) numWaters*maxNeighbors, "waterTrimerPIPNeighbors");
    diagnosticStatus.initialize<int>(cc, 3, "waterTrimerPIPDiagnosticStatus");
    cc.clearBuffer(diagnosticStatus);
    trimerCount.initialize<int>(cc, 1, "waterTrimerPIPTrimerCount");
    trimers.initialize<mm_int4>(cc, maxTrimers, "waterTrimerPIPTrimers");
    cc.addAutoclearBuffer(trimerCount);
    oxygenPos.initialize(cc, numWaters, cc.getPosq().getElementSize(), "waterTrimerPIPOxygenPos");

    double zero, full, window[4];
    map<string, string> defines;
    if (periodic)
        defines["USE_PERIODIC"] = "1";
    defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_WATERS"] = cc.intToString(numWaters);
    defines["NUM_VARIABLES"] = cc.intToString(numVariables);
    defines["STREAM_REAL"] = (useDouble ? "double" : "float");
    defines["STREAM_CAPACITY"] = cc.intToString(streamCapacity);
    defines["STREAM_BATCHES"] = cc.intToString(streamBatches);
    defines["STREAM_CAPACITY_PADDED"] = cc.intToString(streamBatches*32);
    defines["NUM_SEGMENTS"] = cc.intToString(numSegments);
    defines["STREAM_WARPS"] = cc.intToString(streamWarps);
    defines["STREAM_CHUNK"] = (useDouble ? "512" : "1024");
    defines["FINISH_WARPS"] = "4";
    defines["MAX_NEIGHBORS"] = cc.intToString(maxNeighbors);
    defines["MAX_TRIMERS"] = cc.intToString(maxTrimers);
    defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    defines["SWITCH_ON"] = cc.doubleToString(force.getSwitchingDistance());
    defines["SWITCH_CUT"] = cc.doubleToString(force.getCutoffDistance());
    force.getOxygenOxygenSwitch(zero, full);
    defines["OO_SHORT_ZERO"] = cc.doubleToString(zero);
    defines["OO_SHORT_FULL"] = cc.doubleToString(full);
    force.getIntermolecularOHSwitch(zero, full);
    defines["OH_INTER_ZERO"] = cc.doubleToString(zero);
    defines["OH_INTER_FULL"] = cc.doubleToString(full);
    force.getIntermolecularHHSwitch(zero, full);
    defines["HH_INTER_ZERO"] = cc.doubleToString(zero);
    defines["HH_INTER_FULL"] = cc.doubleToString(full);
    force.getIntramolecularOHWindow(window[0], window[1], window[2], window[3]);
    defines["OH_LOWER_ZERO"] = cc.doubleToString(window[0]);
    defines["OH_LOWER_FULL"] = cc.doubleToString(window[1]);
    defines["OH_UPPER_FULL"] = cc.doubleToString(window[2]);
    defines["OH_UPPER_ZERO"] = cc.doubleToString(window[3]);
    force.getIntramolecularHHWindow(window[0], window[1], window[2], window[3]);
    defines["HH_LOWER_ZERO"] = cc.doubleToString(window[0]);
    defines["HH_LOWER_FULL"] = cc.doubleToString(window[1]);
    defines["HH_UPPER_FULL"] = cc.doubleToString(window[2]);
    defines["HH_UPPER_ZERO"] = cc.doubleToString(window[3]);
    defines["MAX_SAFE_FORCE"] = cc.doubleToString(min(1.0e6, 0.25*2147483647.0/(1.5*maxNeighbors*(maxNeighbors-1))));
    defines["M_PI"] = cc.doubleToString(M_PI);
    ComputeProgram program = cc.compileProgram(CommonSAPTMLKernelSources::waterTrimerPIP, defines);
    invertKernel = program->createKernel("invertAtomOrder");
    neighborKernel = program->createKernel("findOxygenNeighbors");
    trimerKernel = program->createKernel("buildWaterTrimers");
    prepareKernel = program->createKernel("wtpipPrepare");
    streamKernel = program->createKernel("wtpipStream");
    finishKernel = program->createKernel("wtpipFinish");
    cc.addForce(new ForceInfo(force));
}

double CommonCalcWaterTrimerPIPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!includeForces && !includeEnergy)
        return 0.0;
    ContextSelector selector(cc);
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        invertKernel->addArg(cc.getAtomIndexArray());
        invertKernel->addArg(originalToCurrent);
        neighborKernel->addArg(originalToCurrent);
        neighborKernel->addArg(waterAtoms);
        neighborKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            neighborKernel->addArg();
        neighborKernel->addArg(neighborCounts);
        neighborKernel->addArg(neighbors);
        neighborKernel->addArg(diagnosticStatus);
        neighborKernel->addArg(oxygenPos);
        for (int i = 0; i < 5; i++)
            trimerKernel->addArg();
        trimerKernel->addArg(neighborCounts);
        trimerKernel->addArg(neighbors);
        trimerKernel->addArg(trimerCount);
        trimerKernel->addArg(trimers);
        trimerKernel->addArg(oxygenPos);
        prepareKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            prepareKernel->addArg();
        prepareKernel->addArg(originalToCurrent);
        prepareKernel->addArg(waterAtoms);
        prepareKernel->addArg(trimerCount);
        prepareKernel->addArg(trimers);
        prepareKernel->addArg(variablePairs);
        prepareKernel->addArg(variableK);
        prepareKernel->addArg(variableD0);
        prepareKernel->addArg(variableCenter);
        prepareKernel->addArg(clusterVars);
        prepareKernel->addArg(clusterR);
        prepareKernel->addArg(clusterX);
        prepareKernel->addArg(diagnosticStatus);
        streamKernel->addArg(trimerCount);
        streamKernel->addArg(clusterVars);
        streamKernel->addArg(streamInfo);
        streamKernel->addArg(streamCoef);
        streamKernel->addArg(segments);
        streamKernel->addArg(partialP);
        streamKernel->addArg(partialG);
        streamKernel->addArg((int) includeForces);
        finishKernel->addArg(cc.getLongForceBuffer());
        finishKernel->addArg(cc.getEnergyBuffer());
        finishKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            finishKernel->addArg();
        finishKernel->addArg(originalToCurrent);
        finishKernel->addArg(waterAtoms);
        finishKernel->addArg(trimerCount);
        finishKernel->addArg(trimers);
        finishKernel->addArg(variablePairs);
        finishKernel->addArg(variableK);
        finishKernel->addArg(clusterR);
        finishKernel->addArg(clusterX);
        finishKernel->addArg(partialP);
        finishKernel->addArg(partialG);
        finishKernel->addArg(diagnosticStatus);
        finishKernel->addArg((int) includeForces);
        finishKernel->addArg((int) includeEnergy);
    }
    setPeriodicBoxArgs(cc, neighborKernel, 3);
    setPeriodicBoxArgs(cc, trimerKernel, 0);
    setPeriodicBoxArgs(cc, prepareKernel, 1);
    setPeriodicBoxArgs(cc, finishKernel, 3);
    streamKernel->setArg(7, (int) includeForces);
    finishKernel->setArg(19, (int) includeForces);
    finishKernel->setArg(20, (int) includeEnergy);
    invertKernel->execute(cc.getNumAtoms());
    neighborKernel->execute((numWaters+127)/128*128, 128);
    trimerKernel->execute(numWaters*maxNeighbors);
    int finishThreads = min(32*min(maxTrimers, 8192), 32*16384);
    prepareKernel->execute(32*min(streamBatches*32, 16384), 128);
    streamKernel->execute(streamThreads, 32*streamWarps);
    finishKernel->execute(max(128, finishThreads/128*128), 128);
    executionCount++;
    if (executionCount == 1 || executionCount%overflowCheckInterval == 0)
        checkClusterDiagnostics(diagnosticStatus, streamCapacity, maxNeighbors, "WaterTrimerPIPForce", "trimers");
    return 0.0;
}

// PartridgeSchwenkeForce

class CommonCalcPartridgeSchwenkeForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const PartridgeSchwenkeForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumWaters();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.resize(3);
        force.getWaterParameters(index, particles[0], particles[1], particles[2]);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const PartridgeSchwenkeForce& force;
};

static string doubleLiteral(double value) {
    // Double-precision literal regardless of the context precision.
    stringstream s;
    s.precision(17);
    s << "(" << scientific << value << ")";
    return s.str();
}

static string realInt(int value) {
    // Integer literal that does not change the precision of an expression.
    return "(real) " + to_string(value);
}

static string polynomialSource(ComputeContext& cc) {
    // Straight-line code for sum_j c_j (x1^a x2^b + x1^b x2^a) x3^c (j >= 1) and its partial derivatives, in the
    // precision of the context (the large cancellation of the surface is between constants evaluated in double).
    using namespace PartridgeSchwenke;
    int maxPower12 = 0, maxPower3 = 0;
    for (int j = 1; j < NumTerms; j++) {
        maxPower12 = max(maxPower12, max(idx1[j], idx2[j])-1);
        maxPower3 = max(maxPower3, idx3[j]-1);
    }
    stringstream src;
    src << "real p1_0 = 1, p2_0 = 1, p3_0 = 1;\n";
    for (int n = 1; n <= maxPower12; n++)
        src << "real p1_" << n << " = p1_" << n-1 << "*(real) psX1, p2_" << n << " = p2_" << n-1 << "*(real) psX2;\n";
    for (int n = 1; n <= maxPower3; n++)
        src << "real p3_" << n << " = p3_" << n-1 << "*(real) psX3;\n";
    for (int j = 1; j < NumTerms; j++) {
        int a = idx1[j]-1, b = idx2[j]-1, c = idx3[j]-1;
        string cj = "(" + cc.doubleToString(f5z*c5zA[j] + fbasis*cbasis[j] + fcore*ccore[j] + frest*crest[j]) + ")";
        src << "{\n";
        src << "real sym = p1_" << a << "*p2_" << b << " + p1_" << b << "*p2_" << a << ";\n";
        src << "sum0 += " << cj << "*sym*p3_" << c << ";\n";
        string d1, d2;
        if (a > 0) {
            d1 += " + " + realInt(a) + "*p1_" + to_string(a-1) + "*p2_" + to_string(b);
            d2 += " + " + realInt(a) + "*p1_" + to_string(b) + "*p2_" + to_string(a-1);
        }
        if (b > 0) {
            d1 += " + " + realInt(b) + "*p1_" + to_string(b-1) + "*p2_" + to_string(a);
            d2 += " + " + realInt(b) + "*p1_" + to_string(a) + "*p2_" + to_string(b-1);
        }
        if (!d1.empty()) {
            src << "sum1 += " << cj << "*(" << d1.substr(3) << ")*p3_" << c << ";\n";
            src << "sum2 += " << cj << "*(" << d2.substr(3) << ")*p3_" << c << ";\n";
        }
        if (c > 0)
            src << "sum3 += " << cj << "*sym*" << realInt(c) << "*p3_" << c-1 << ";\n";
        src << "}\n";
    }
    return src.str();
}

void CommonCalcPartridgeSchwenkeForceKernel::initialize(const System& system, const PartridgeSchwenkeForce& force) {
    using namespace PartridgeSchwenke;
    ContextSelector selector(cc);
    int numWaters = force.getNumWaters();
    if (numWaters == 0)
        return;
    vector<vector<int> > atoms(numWaters, vector<int>(3));
    for (int i = 0; i < numWaters; i++)
        force.getWaterParameters(i, atoms[i][0], atoms[i][1], atoms[i][2]);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["PS_POLYNOMIAL"] = polynomialSource(cc);
    replacements["PS_CONSTANT"] = doubleLiteral(2.0*(f5z*c5zA[0] + fbasis*cbasis[0] + fcore*ccore[0] + frest*crest[0]));
    replacements["PS_ALPHAOH"] = doubleLiteral(alphaoh);
    replacements["PS_ROH"] = doubleLiteral(roh);
    replacements["PS_REOH"] = doubleLiteral(reoh);
    replacements["PS_DEOH"] = doubleLiteral(f5z*deohA);
    replacements["PS_PHH1"] = doubleLiteral(f5z*phh1A*exp(phh2));
    replacements["PS_PHH2"] = doubleLiteral(phh2);
    replacements["PS_COSTHE"] = doubleLiteral(costhe);
    replacements["PS_B1"] = doubleLiteral(b1);
    replacements["PS_SHIFT"] = doubleLiteral(shift);
    replacements["PS_CM1_KJ"] = doubleLiteral(cm1_kJmol);
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonSAPTMLKernelSources::partridgeSchwenke, replacements), force.getForceGroup());
    cc.addForce(new ForceInfo(force));
}

double CommonCalcPartridgeSchwenkeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // The interaction is evaluated by the BondedUtilities kernel.
    return 0.0;
}
