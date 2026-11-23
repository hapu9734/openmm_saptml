/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "WaterDimerPIPForce.h"
#include "internal/WaterDimerPIPForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

WaterDimerPIPForce::WaterDimerPIPForce() : nonbondedMethod(CutoffPeriodic), cutoffDistance(0.65), switchingDistance(0.45),
        inPlane(0.0), outOfPlane(0.0), ooZero(0.18), ooFull(0.19), ohZero(0.085), ohFull(0.1), hhZero(0.05), hhFull(0.065),
        maxNeighbors(64), overflowCheckInterval(1000) {
    const double oh[4] = {0.05, 0.065, 0.14, 0.16};
    const double hh[4] = {0.085, 0.105, 0.205, 0.23};
    for (int i = 0; i < 4; i++) {
        ohWindow[i] = oh[i];
        hhWindow[i] = hh[i];
    }
}

int WaterDimerPIPForce::addWater(int oxygen, int hydrogen1, int hydrogen2) {
    waters.push_back(WaterInfo(oxygen, hydrogen1, hydrogen2));
    return waters.size()-1;
}

void WaterDimerPIPForce::getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const {
    ASSERT_VALID_INDEX(index, waters);
    oxygen = waters[index].oxygen;
    hydrogen1 = waters[index].hydrogen1;
    hydrogen2 = waters[index].hydrogen2;
}

void WaterDimerPIPForce::setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2) {
    ASSERT_VALID_INDEX(index, waters);
    waters[index] = WaterInfo(oxygen, hydrogen1, hydrogen2);
}

int WaterDimerPIPForce::addVariable(int site1, int site2, double k, double d0, VariableForm form) {
    variables.push_back(VariableInfo(site1, site2, k, d0, form));
    return variables.size()-1;
}

void WaterDimerPIPForce::getVariableParameters(int index, int& site1, int& site2, double& k, double& d0, VariableForm& form) const {
    ASSERT_VALID_INDEX(index, variables);
    site1 = variables[index].site1;
    site2 = variables[index].site2;
    k = variables[index].k;
    d0 = variables[index].d0;
    form = variables[index].form;
}

void WaterDimerPIPForce::setVariableParameters(int index, int site1, int site2, double k, double d0, VariableForm form) {
    ASSERT_VALID_INDEX(index, variables);
    variables[index] = VariableInfo(site1, site2, k, d0, form);
}

int WaterDimerPIPForce::addMonomial(int variable1, int variable2, int variable3, int variable4) {
    monomials.push_back(MonomialInfo(variable1, variable2, variable3, variable4));
    return monomials.size()-1;
}

void WaterDimerPIPForce::getMonomialParameters(int index, int& variable1, int& variable2, int& variable3, int& variable4) const {
    ASSERT_VALID_INDEX(index, monomials);
    variable1 = monomials[index].variable1;
    variable2 = monomials[index].variable2;
    variable3 = monomials[index].variable3;
    variable4 = monomials[index].variable4;
}

void WaterDimerPIPForce::setMonomialParameters(int index, int variable1, int variable2, int variable3, int variable4) {
    ASSERT_VALID_INDEX(index, monomials);
    monomials[index] = MonomialInfo(variable1, variable2, variable3, variable4);
}

int WaterDimerPIPForce::addTerm(double coefficient, int firstMonomial, int numMonomials) {
    terms.push_back(TermInfo(coefficient, firstMonomial, numMonomials));
    return terms.size()-1;
}

void WaterDimerPIPForce::getTermParameters(int index, double& coefficient, int& firstMonomial, int& numMonomials) const {
    ASSERT_VALID_INDEX(index, terms);
    coefficient = terms[index].coefficient;
    firstMonomial = terms[index].firstMonomial;
    numMonomials = terms[index].numMonomials;
}

void WaterDimerPIPForce::setTermParameters(int index, double coefficient, int firstMonomial, int numMonomials) {
    ASSERT_VALID_INDEX(index, terms);
    terms[index] = TermInfo(coefficient, firstMonomial, numMonomials);
}

WaterDimerPIPForce::NonbondedMethod WaterDimerPIPForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void WaterDimerPIPForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("WaterDimerPIPForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double WaterDimerPIPForce::getCutoffDistance() const {
    return cutoffDistance;
}

void WaterDimerPIPForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

double WaterDimerPIPForce::getSwitchingDistance() const {
    return switchingDistance;
}

void WaterDimerPIPForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

void WaterDimerPIPForce::getExtraPointParameters(double& inPlane, double& outOfPlane) const {
    inPlane = this->inPlane;
    outOfPlane = this->outOfPlane;
}

void WaterDimerPIPForce::setExtraPointParameters(double inPlane, double outOfPlane) {
    this->inPlane = inPlane;
    this->outOfPlane = outOfPlane;
}

void WaterDimerPIPForce::getOxygenOxygenSwitch(double& zeroDistance, double& fullDistance) const {
    zeroDistance = ooZero;
    fullDistance = ooFull;
}

void WaterDimerPIPForce::setOxygenOxygenSwitch(double zeroDistance, double fullDistance) {
    ooZero = zeroDistance;
    ooFull = fullDistance;
}

void WaterDimerPIPForce::getIntermolecularOHSwitch(double& zeroDistance, double& fullDistance) const {
    zeroDistance = ohZero;
    fullDistance = ohFull;
}

void WaterDimerPIPForce::setIntermolecularOHSwitch(double zeroDistance, double fullDistance) {
    ohZero = zeroDistance;
    ohFull = fullDistance;
}

void WaterDimerPIPForce::getIntermolecularHHSwitch(double& zeroDistance, double& fullDistance) const {
    zeroDistance = hhZero;
    fullDistance = hhFull;
}

void WaterDimerPIPForce::setIntermolecularHHSwitch(double zeroDistance, double fullDistance) {
    hhZero = zeroDistance;
    hhFull = fullDistance;
}

void WaterDimerPIPForce::getIntramolecularOHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const {
    lowerZero = ohWindow[0];
    lowerFull = ohWindow[1];
    upperFull = ohWindow[2];
    upperZero = ohWindow[3];
}

void WaterDimerPIPForce::setIntramolecularOHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero) {
    ohWindow[0] = lowerZero;
    ohWindow[1] = lowerFull;
    ohWindow[2] = upperFull;
    ohWindow[3] = upperZero;
}

void WaterDimerPIPForce::getIntramolecularHHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const {
    lowerZero = hhWindow[0];
    lowerFull = hhWindow[1];
    upperFull = hhWindow[2];
    upperZero = hhWindow[3];
}

void WaterDimerPIPForce::setIntramolecularHHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero) {
    hhWindow[0] = lowerZero;
    hhWindow[1] = lowerFull;
    hhWindow[2] = upperFull;
    hhWindow[3] = upperZero;
}

int WaterDimerPIPForce::getMaxNeighbors() const {
    return maxNeighbors;
}

void WaterDimerPIPForce::setMaxNeighbors(int maxNeighbors) {
    if (maxNeighbors < 2)
        throw OpenMMException("WaterDimerPIPForce: maxNeighbors must be at least 2");
    this->maxNeighbors = maxNeighbors;
}

int WaterDimerPIPForce::getOverflowCheckInterval() const {
    return overflowCheckInterval;
}

void WaterDimerPIPForce::setOverflowCheckInterval(int interval) {
    if (interval < 1)
        throw OpenMMException("WaterDimerPIPForce: The overflow check interval must be positive");
    overflowCheckInterval = interval;
}

ForceImpl* WaterDimerPIPForce::createImpl() const {
    return new WaterDimerPIPForceImpl(*this);
}
