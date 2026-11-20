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
 * This tests the CUDA implementation of the SAPT-ML forces: agreement of the energies and forces with the Reference
 * platform, the induced-dipole solvers and a few dynamics steps.  The finite-difference check of TestSAPTMLForces.h is
 * not run here: the device kernels keep the cluster geometry of the two-body polynomial in single precision, so the
 * energy is not smooth on the scale of the finite-difference step, while the analytic forces agree with the Reference
 * platform to better than 1e-5 kJ/mol/nm in double precision.
 */

#include "TestSAPTMLForces.h"

extern "C" OPENMM_EXPORT void registerSAPTMLCudaKernelFactories();
extern "C" OPENMM_EXPORT void registerSAPTMLReferenceKernelFactories();

static map<string, string> props;
static double tolE = 1e-4, tolF = 1e-3;

void compare(ForceKind kind, const char* label, const double xyz[][3], int nwat, double box, bool redistribute = true,
        MBPolElectrostaticsForce::PolarizationType pol = MBPolElectrostaticsForce::Mutual) {
    Platform& reference = Platform::getPlatformByName("Reference");
    Platform& cuda = Platform::getPlatformByName("CUDA");
    vector<Vec3> fr, fc;
    double er = evaluate(kind, reference, xyz, nwat, box, fr, map<string, string>(), redistribute, pol, 1e-9);
    double ec = evaluate(kind, cuda, xyz, nwat, box, fc, props, redistribute, pol, 1e-9);
    double maxF = 0, normF = 0;
    for (int i = 0; i < (int) fr.size(); i++) {
        Vec3 d = fr[i]-fc[i];
        maxF = max(maxF, sqrt(d.dot(d)));
        normF = max(normF, sqrt(fr[i].dot(fr[i])));
    }
    printf("%-26s %-28s E_ref %16.8f  E_cuda %16.8f  dE %10.2e  |dF|max %9.2e (relative %8.2e)\n",
           kindName(kind), label, er, ec, ec-er, maxF, maxF/normF);
    ASSERT_EQUAL_TOL(er, ec, tolE*max(1.0, fabs(er)));
    ASSERT(maxF/normF < tolF);
}

void testSolvers() {
    // DIIS and SOR, with and without the predictor, must converge to the same dipoles.
    Platform& cuda = Platform::getPlatformByName("CUDA");
    double energies[4];
    for (int i = 0; i < 4; i++) {
        System system;
        vector<Vec3> positions;
        buildWaters(system, W32_XYZ, 32, positions, TEST_BOX_NM);
        MBPolElectrostaticsForce* force = addElectrostatics(system, 32, TEST_BOX_NM, true, MBPolElectrostaticsForce::Mutual, 1e-8);
        force->setInducedDipoleMethod(i < 2 ? MBPolElectrostaticsForce::DIIS : MBPolElectrostaticsForce::SOR);
        force->setPredictorOrder(i%2 == 0 ? 0 : 3);
        force->setDIISHistorySize(8);
        force->setSORParameter(0.6);
        VerletIntegrator integrator(0.0002);
        Context context(system, integrator, cuda, props);
        context.setPositions(positions);
        context.computeVirtualSites();
        integrator.step(5);
        context.setPositions(positions);
        context.computeVirtualSites();
        energies[i] = context.getState(State::Energy).getPotentialEnergy();
    }
    printf("solvers: DIIS %.6f, DIIS + predictor %.6f, SOR %.6f, SOR + predictor %.6f kJ/mol\n", energies[0], energies[1], energies[2], energies[3]);
    for (int i = 1; i < 4; i++)
        ASSERT_EQUAL_TOL(energies[0], energies[i], 1e-5*fabs(energies[0]));
}

int main(int argc, char* argv[]) {
    try {
        registerSAPTMLReferenceKernelFactories();
        registerSAPTMLCudaKernelFactories();
        string precision = (argc > 1 ? argv[1] : "single");
        props["Precision"] = precision;
        if (precision == "double") {
            tolE = 1e-7;
            tolF = 1e-5;
        }
        else if (precision == "mixed") {
            tolE = 1e-4;
            tolF = 1e-3;
        }
        printf("CUDA precision: %s\n", precision.c_str());
        compare(Electrostatics, "water trimer, mutual", WATER3_XYZ, 3, 0.0);
        compare(Electrostatics, "water trimer, direct", WATER3_XYZ, 3, 0.0, true, MBPolElectrostaticsForce::Direct);
        compare(Electrostatics, "water trimer, fixed charges", WATER3_XYZ, 3, 0.0, false);
        compare(Electrostatics, "32 waters, cluster", W32_XYZ, 32, 0.0);
        compare(Electrostatics, "32 waters, PME", W32_XYZ, 32, TEST_BOX_NM);
        compare(Electrostatics, "32 waters, PME, direct", W32_XYZ, 32, TEST_BOX_NM, true, MBPolElectrostaticsForce::Direct);
        compare(DimerPIP, "32 waters, cluster", W32_XYZ, 32, 0.0);
        compare(DimerPIP, "32 waters, periodic", W32_XYZ, 32, TEST_BOX_NM);
        compare(TrimerPIP, "32 waters, cluster", W32_XYZ, 32, 0.0);
        compare(TrimerPIP, "32 waters, periodic", W32_XYZ, 32, TEST_BOX_NM);
        compare(PartridgeSchwenke, "32 waters, cluster", W32_XYZ, 32, 0.0);
        compare(PartridgeSchwenke, "32 waters, periodic", W32_XYZ, 32, TEST_BOX_NM);
        compare(PartridgeSchwenke, "stretched and bent waters", PS_XYZ, PS_NUM_WATERS, 0.0);
        Platform& cuda = Platform::getPlatformByName("CUDA");
        if (precision == "double")
            testPartridgeSchwenke(cuda, props, 1e-8, 1e-4);
        testSolvers();
        testDynamics(cuda, props);
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
