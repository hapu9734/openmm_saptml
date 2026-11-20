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

#include "ReferenceSAPTMLKernelFactory.h"
#include "ReferenceSAPTMLKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
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
        return new ReferenceCalcMBPolElectrostaticsForceKernel(name, platform, context.getSystem());
    if (name == CalcWaterDimerPIPForceKernel::Name())
        return new ReferenceCalcWaterDimerPIPForceKernel(name, platform);
    if (name == CalcWaterTrimerPIPForceKernel::Name())
        return new ReferenceCalcWaterTrimerPIPForceKernel(name, platform);
    if (name == CalcPartridgeSchwenkeForceKernel::Name())
        return new ReferenceCalcPartridgeSchwenkeForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
