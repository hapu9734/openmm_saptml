/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "PartridgeSchwenkeForce.h"
#include "internal/PartridgeSchwenkeForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

PartridgeSchwenkeForce::PartridgeSchwenkeForce() : usePeriodic(false) {
}

int PartridgeSchwenkeForce::addWater(int oxygen, int hydrogen1, int hydrogen2) {
    waters.push_back(WaterInfo(oxygen, hydrogen1, hydrogen2));
    return waters.size()-1;
}

void PartridgeSchwenkeForce::getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const {
    ASSERT_VALID_INDEX(index, waters);
    oxygen = waters[index].oxygen;
    hydrogen1 = waters[index].hydrogen1;
    hydrogen2 = waters[index].hydrogen2;
}

void PartridgeSchwenkeForce::setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2) {
    ASSERT_VALID_INDEX(index, waters);
    waters[index] = WaterInfo(oxygen, hydrogen1, hydrogen2);
}

void PartridgeSchwenkeForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool PartridgeSchwenkeForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

ForceImpl* PartridgeSchwenkeForce::createImpl() const {
    return new PartridgeSchwenkeForceImpl(*this);
}
