/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "ReferenceSAPTMLKernelFactory.h"
#include "ReferenceSAPTMLKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/OpenMMException.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

static void registerReferenceKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceSAPTMLKernelFactory* factory = new ReferenceSAPTMLKernelFactory();
            platform.registerKernelFactory(CalcMBPolElectrostaticsForceKernel::Name(), factory);
            platform.registerKernelFactory(CalcWaterDimerPIPForceKernel::Name(), factory);
            platform.registerKernelFactory(CalcWaterTrimerPIPForceKernel::Name(), factory);
            platform.registerKernelFactory(CalcPartridgeSchwenkeForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    registerReferenceKernelFactories();
}

extern "C" OPENMM_EXPORT void registerSAPTMLReferenceKernelFactories() {
    registerReferenceKernelFactories();
}

KernelImpl* ReferenceSAPTMLKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcMBPolElectrostaticsForceKernel::Name())
        return new ReferenceCalcMBPolElectrostaticsForceKernel(name, platform);
    if (name == CalcWaterDimerPIPForceKernel::Name())
        return new ReferenceCalcWaterDimerPIPForceKernel(name, platform);
    if (name == CalcWaterTrimerPIPForceKernel::Name())
        return new ReferenceCalcWaterTrimerPIPForceKernel(name, platform);
    if (name == CalcPartridgeSchwenkeForceKernel::Name())
        return new ReferenceCalcPartridgeSchwenkeForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
