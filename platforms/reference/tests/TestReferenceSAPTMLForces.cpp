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

/**
 * This tests the Reference implementation of the SAPT-ML forces.
 */

#include "TestSAPTMLForces.h"

extern "C" OPENMM_EXPORT void registerSAPTMLReferenceKernelFactories();

int main(int argc, char* argv[]) {
    try {
        registerSAPTMLReferenceKernelFactories();
        Platform& platform = Platform::getPlatformByName("Reference");
        map<string, string> props;
        testElectrostaticsEnergies(platform, props, 1e-6);
        testPartridgeSchwenke(platform, props, 1e-8, 1e-5);
        testForces(platform, props, 5e-4);
        testDynamics(platform, props);
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
