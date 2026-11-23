/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "MBPolElectrostaticsForce.h"
#include "internal/MBPolElectrostaticsForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

MBPolElectrostaticsForce::MBPolElectrostaticsForce() : nonbondedMethod(NoCutoff), polarizationType(Mutual), inducedDipoleMethod(DIIS),
        cutoffDistance(0.9), ewaldErrorTol(1e-5), alpha(0.0), nx(0), ny(0), nz(0),
        mutualInducedMaxIterations(60), diisHistorySize(20), predictorOrder(3), mutualInducedTargetEpsilon(1e-5), sorParameter(0.55),
        includeChargeRedistribution(true), fixedChargeO(-0.51966), fixedChargeH(0.25983), fixedChargeM(0.0),
        alphaO(0.001310), alphaH(0.000294), dampO(0.001310), dampH(0.000294), dampM(0.001310),
        tholeCC(0.4), tholeCD(0.4), tholeDD(0.055), tholeDDOH(0.626), tholeDDHH(0.055) {
}

int MBPolElectrostaticsForce::addWater(int oxygen, int hydrogen1, int hydrogen2, int msite) {
    waters.push_back(WaterInfo(oxygen, hydrogen1, hydrogen2, msite));
    return waters.size()-1;
}

void MBPolElectrostaticsForce::getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2, int& msite) const {
    ASSERT_VALID_INDEX(index, waters);
    oxygen = waters[index].oxygen;
    hydrogen1 = waters[index].hydrogen1;
    hydrogen2 = waters[index].hydrogen2;
    msite = waters[index].msite;
}

void MBPolElectrostaticsForce::setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2, int msite) {
    ASSERT_VALID_INDEX(index, waters);
    waters[index] = WaterInfo(oxygen, hydrogen1, hydrogen2, msite);
}

MBPolElectrostaticsForce::NonbondedMethod MBPolElectrostaticsForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void MBPolElectrostaticsForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("MBPolElectrostaticsForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double MBPolElectrostaticsForce::getCutoffDistance() const {
    return cutoffDistance;
}

void MBPolElectrostaticsForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

double MBPolElectrostaticsForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void MBPolElectrostaticsForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void MBPolElectrostaticsForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void MBPolElectrostaticsForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void MBPolElectrostaticsForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MBPolElectrostaticsForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

MBPolElectrostaticsForce::PolarizationType MBPolElectrostaticsForce::getPolarizationType() const {
    return polarizationType;
}

void MBPolElectrostaticsForce::setPolarizationType(PolarizationType type) {
    if (type < 0 || type > 1)
        throw OpenMMException("MBPolElectrostaticsForce: Illegal value for polarization type");
    polarizationType = type;
}

int MBPolElectrostaticsForce::getMutualInducedMaxIterations() const {
    return mutualInducedMaxIterations;
}

void MBPolElectrostaticsForce::setMutualInducedMaxIterations(int iterations) {
    mutualInducedMaxIterations = iterations;
}

double MBPolElectrostaticsForce::getMutualInducedTargetEpsilon() const {
    return mutualInducedTargetEpsilon;
}

void MBPolElectrostaticsForce::setMutualInducedTargetEpsilon(double epsilon) {
    mutualInducedTargetEpsilon = epsilon;
}

MBPolElectrostaticsForce::InducedDipoleMethod MBPolElectrostaticsForce::getInducedDipoleMethod() const {
    return inducedDipoleMethod;
}

void MBPolElectrostaticsForce::setInducedDipoleMethod(InducedDipoleMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("MBPolElectrostaticsForce: Illegal value for induced dipole method");
    inducedDipoleMethod = method;
}

int MBPolElectrostaticsForce::getDIISHistorySize() const {
    return diisHistorySize;
}

void MBPolElectrostaticsForce::setDIISHistorySize(int size) {
    if (size < 2 || size > 32)
        throw OpenMMException("MBPolElectrostaticsForce: The DIIS history size must be between 2 and 32");
    diisHistorySize = size;
}

double MBPolElectrostaticsForce::getSORParameter() const {
    return sorParameter;
}

void MBPolElectrostaticsForce::setSORParameter(double omega) {
    if (!(omega > 0.0 && omega <= 1.0))
        throw OpenMMException("MBPolElectrostaticsForce: The SOR parameter must be between 0 and 1");
    sorParameter = omega;
}

int MBPolElectrostaticsForce::getPredictorOrder() const {
    return predictorOrder;
}

void MBPolElectrostaticsForce::setPredictorOrder(int order) {
    if (order < 0 || order > 3)
        throw OpenMMException("MBPolElectrostaticsForce: The predictor order must be between 0 and 3");
    predictorOrder = order;
}

bool MBPolElectrostaticsForce::getIncludeChargeRedistribution() const {
    return includeChargeRedistribution;
}

void MBPolElectrostaticsForce::setIncludeChargeRedistribution(bool include) {
    includeChargeRedistribution = include;
}

void MBPolElectrostaticsForce::getFixedCharges(double& chargeO, double& chargeH, double& chargeM) const {
    chargeO = fixedChargeO;
    chargeH = fixedChargeH;
    chargeM = fixedChargeM;
}

void MBPolElectrostaticsForce::setFixedCharges(double chargeO, double chargeH, double chargeM) {
    fixedChargeO = chargeO;
    fixedChargeH = chargeH;
    fixedChargeM = chargeM;
}

void MBPolElectrostaticsForce::getPolarizabilities(double& alphaO, double& alphaH) const {
    alphaO = this->alphaO;
    alphaH = this->alphaH;
}

void MBPolElectrostaticsForce::setPolarizabilities(double alphaO, double alphaH) {
    this->alphaO = alphaO;
    this->alphaH = alphaH;
}

void MBPolElectrostaticsForce::getDampingFactors(double& dampO, double& dampH, double& dampM) const {
    dampO = this->dampO;
    dampH = this->dampH;
    dampM = this->dampM;
}

void MBPolElectrostaticsForce::setDampingFactors(double dampO, double dampH, double dampM) {
    this->dampO = dampO;
    this->dampH = dampH;
    this->dampM = dampM;
}

void MBPolElectrostaticsForce::getTholeParameters(double& cc, double& cd, double& dd, double& ddOH, double& ddHH) const {
    cc = tholeCC;
    cd = tholeCD;
    dd = tholeDD;
    ddOH = tholeDDOH;
    ddHH = tholeDDHH;
}

void MBPolElectrostaticsForce::setTholeParameters(double cc, double cd, double dd, double ddOH, double ddHH) {
    tholeCC = cc;
    tholeCD = cd;
    tholeDD = dd;
    tholeDDOH = ddOH;
    tholeDDHH = ddHH;
}

void MBPolElectrostaticsForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MBPolElectrostaticsForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void MBPolElectrostaticsForce::getCharges(Context& context, vector<double>& charges) {
    dynamic_cast<MBPolElectrostaticsForceImpl&>(getImplInContext(context)).getCharges(getContextImpl(context), charges);
}

void MBPolElectrostaticsForce::updateParametersInContext(Context& context) {
    dynamic_cast<MBPolElectrostaticsForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

ForceImpl* MBPolElectrostaticsForce::createImpl() const {
    return new MBPolElectrostaticsForceImpl(*this);
}
