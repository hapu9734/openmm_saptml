/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "CudaSAPTMLKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"

using namespace SAPTMLPlugin;
using namespace OpenMM;

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
