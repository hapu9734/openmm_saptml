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
