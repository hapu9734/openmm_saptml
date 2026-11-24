/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include <exception>

#include "CudaSAPTMLKernelFactory.h"
#include "CudaSAPTMLKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

static void registerCudaKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaSAPTMLKernelFactory* factory = new CudaSAPTMLKernelFactory();
        platform.registerKernelFactory(CalcMBPolElectrostaticsForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcWaterDimerPIPForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcWaterTrimerPIPForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcPartridgeSchwenkeForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // The CUDA platform is not available.
    }
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    registerCudaKernelFactories();
}

extern "C" OPENMM_EXPORT void registerSAPTMLCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerCudaKernelFactories();
}

KernelImpl* CudaSAPTMLKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcMBPolElectrostaticsForceKernel::Name())
        return new CudaCalcMBPolElectrostaticsForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcWaterDimerPIPForceKernel::Name())
        return new CommonCalcWaterDimerPIPForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcWaterTrimerPIPForceKernel::Name())
        return new CommonCalcWaterTrimerPIPForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcPartridgeSchwenkeForceKernel::Name())
        return new CommonCalcPartridgeSchwenkeForceKernel(name, platform, cu, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
