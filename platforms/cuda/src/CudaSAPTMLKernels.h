#ifndef CUDA_SAPTML_KERNELS_H_
#define CUDA_SAPTML_KERNELS_H_

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

#include "CommonSAPTMLKernels.h"
#include "openmm/cuda/CudaContext.h"
#ifdef SAPTML_USE_COMPUTE_FFT
#include "openmm/common/FFT3D.h"
#else
#include <cufft.h>
#endif

namespace SAPTMLPlugin {

/**
 * This kernel is invoked by MBPolElectrostaticsForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMBPolElectrostaticsForceKernel : public CommonCalcMBPolElectrostaticsForceKernel {
public:
    CudaCalcMBPolElectrostaticsForceKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CommonCalcMBPolElectrostaticsForceKernel(name, platform, cu, system), hasInitializedFFT(false) {
    }
    ~CudaCalcMBPolElectrostaticsForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MBPolElectrostaticsForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MBPolElectrostaticsForce& force);
    /**
     * Compute the FFT of the PME grid.
     */
    void computeFFT(bool forward);
    /**
     * Get whether to use fixed point charge spreading.
     */
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
