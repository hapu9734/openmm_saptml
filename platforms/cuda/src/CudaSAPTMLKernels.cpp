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

#include "CudaSAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcMBPolElectrostaticsForceKernel::~CudaCalcMBPolElectrostaticsForceKernel() {
#ifndef SAPTML_USE_COMPUTE_FFT
    ContextSelector selector(cc);
    if (hasInitializedFFT)
        cufftDestroy(fft);
#endif
}

void CudaCalcMBPolElectrostaticsForceKernel::initialize(const System& system, const MBPolElectrostaticsForce& force) {
    CommonCalcMBPolElectrostaticsForceKernel::initialize(system, force);
    if (usePME) {
        ContextSelector selector(cc);
#ifdef SAPTML_USE_COMPUTE_FFT
        fft = cc.createFFT(gridSizeX, gridSizeY, gridSizeZ, false);
#else
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ, cc.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cc.intToString(result));
#endif
        hasInitializedFFT = true;
    }
}

void CudaCalcMBPolElectrostaticsForceKernel::computeFFT(bool forward) {
#ifdef SAPTML_USE_COMPUTE_FFT
    // The platform FFT uses its input as workspace: grid1 is rewritten by the inverse transform, grid2 by the next
    // forward transform, so neither is needed afterwards.
    if (forward)
        fft->execFFT(pmeGrid1, pmeGrid2, true);
    else
        fft->execFFT(pmeGrid2, pmeGrid1, false);
#else
    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
    if (forward) {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer(), CUFFT_FORWARD);
    }
    else {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid2.getDevicePointer(), (double2*) grid1.getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) grid2.getDevicePointer(), (float2*) grid1.getDevicePointer(), CUFFT_INVERSE);
    }
#endif
}
