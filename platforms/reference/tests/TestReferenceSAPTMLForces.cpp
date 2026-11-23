/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

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
