#ifndef CUDA_SAPTML_KERNELS_H_
#define CUDA_SAPTML_KERNELS_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "CommonSAPTMLKernels.h"
#include "openmm/cuda/CudaContext.h"
#ifdef SAPTML_USE_COMPUTE_FFT
#include "openmm/common/FFT3D.h"
#else
#include <cufft.h>
#endif

namespace SAPTMLPlugin {

class CudaCalcMBPolElectrostaticsForceKernel : public CommonCalcMBPolElectrostaticsForceKernel {
public:
    CudaCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CommonCalcMBPolElectrostaticsForceKernel(name, platform, cu, system), hasInitializedFFT(false) {
    }
    ~CudaCalcMBPolElectrostaticsForceKernel();
    void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force);
    void computeFFT(bool forward);
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
private:
    bool hasInitializedFFT;
#ifdef SAPTML_USE_COMPUTE_FFT
    OpenMM::FFT3D fft;
#else
    cufftHandle fft;
#endif
};

} // namespace SAPTMLPlugin

#endif /*CUDA_SAPTML_KERNELS_H_*/
