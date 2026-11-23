#ifndef SAPTML_TEST_FORCES_H_
#define SAPTML_TEST_FORCES_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

/**
 * Tests shared by the platforms: electrostatic energies of a water trimer and a 32-water cluster against reference
 * values, Partridge-Schwenke energies, finite-difference forces of all four forces, and a short MD run.  The
 * polynomial forces use small synthetic polynomials that exercise every variable.
 */

#include "MBPolElectrostaticsForce.h"
#include "PartridgeSchwenkeForce.h"
#include "WaterDimerPIPForce.h"
#include "WaterTrimerPIPForce.h"
#include "TestWaters.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/VirtualSite.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace SAPTMLPlugin;
using namespace OpenMM;
using namespace std;

static const double VS_O = 0.573293118, VS_H = 0.213353441;

/**
 * Build a System of waters (O, H1, H2, M particle order) and their positions.
 */
static void buildWaters(System& system, const double xyz[][3], int nwat, vector<Vec3>& positions, double box) {
    positions.clear();
    for (int w = 0; w < nwat; w++) {
        int j = 4*w;
        system.addParticle(15.99943);
        system.addParticle(1.007947);
        system.addParticle(1.007947);
        system.addParticle(0.0);
        system.setVirtualSite(j+3, new ThreeParticleAverageSite(j, j+1, j+2, VS_O, VS_H, VS_H));
        Vec3 o(xyz[3*w][0], xyz[3*w][1], xyz[3*w][2]);
        Vec3 h1(xyz[3*w+1][0], xyz[3*w+1][1], xyz[3*w+1][2]);
        Vec3 h2(xyz[3*w+2][0], xyz[3*w+2][1], xyz[3*w+2][2]);
        positions.push_back(o);
        positions.push_back(h1);
        positions.push_back(h2);
        positions.push_back(o*VS_O+h1*VS_H+h2*VS_H);
    }
    if (box > 0)
        system.setDefaultPeriodicBoxVectors(Vec3(box, 0, 0), Vec3(0, box, 0), Vec3(0, 0, box));
}

static MBPolElectrostaticsForce* addElectrostatics(System& system, int nwat, double box, bool redistribute,
        MBPolElectrostaticsForce::PolarizationType pol, double eps) {
    MBPolElectrostaticsForce* force = new MBPolElectrostaticsForce();
    for (int w = 0; w < nwat; w++)
        force->addWater(4*w, 4*w+1, 4*w+2, 4*w+3);
    if (box > 0) {
        force->setNonbondedMethod(MBPolElectrostaticsForce::PME);
        force->setCutoffDistance(0.9);
        force->setPMEParameters(TEST_ALPHA_NM, TEST_GRID, TEST_GRID, TEST_GRID);
    }
    force->setIncludeChargeRedistribution(redistribute);
    force->setPolarizationType(pol);
    force->setMutualInducedTargetEpsilon(eps);
    force->setMutualInducedMaxIterations(300);
    system.addForce(force);
    return force;
}

/**
 * A synthetic two-body polynomial on the 31 MB-pol site pairs (sites 0-4: O, H1, H2, X1, X2 of the first water, 5-9
 * of the second), with a linear term in every variable and terms of degree 2 to 4.
 */
static WaterDimerPIPForce* addDimerPIP(System& system, int nwat, double box) {
    WaterDimerPIPForce* force = new WaterDimerPIPForce();
    for (int w = 0; w < nwat; w++)
        force->addWater(4*w, 4*w+1, 4*w+2);
    force->setNonbondedMethod(box > 0 ? WaterDimerPIPForce::CutoffPeriodic : WaterDimerPIPForce::CutoffNonPeriodic);
    force->setExtraPointParameters(-0.0972148691, 0.9859272078);
    const int pairs[31][2] = {{1, 2}, {6, 7}, {0, 1}, {0, 2}, {5, 6}, {5, 7}, {1, 6}, {1, 7}, {2, 6}, {2, 7}, {0, 6}, {0, 7}, {5, 1}, {5, 2}, {0, 5},
                              {3, 6}, {3, 7}, {4, 6}, {4, 7}, {8, 1}, {8, 2}, {9, 1}, {9, 2}, {0, 8}, {0, 9}, {5, 3}, {5, 4}, {3, 8}, {3, 9}, {4, 8}, {4, 9}};
    for (int i = 0; i < 31; i++) {
        bool coulomb = (i >= 6 && i <= 14);
        force->addVariable(pairs[i][0], pairs[i][1], coulomb ? 0.3 : 0.6, coulomb ? 4.0 : 1.0,
                           coulomb ? WaterDimerPIPForce::Coulomb : WaterDimerPIPForce::Exponential);
    }
    int m = 0;
    for (int i = 0; i < 31; i++) {
        force->addMonomial(i, -1, -1, -1);
        force->addTerm(0.5*(i%3+1), m++, 1);
    }
    for (int i = 0; i < 31; i += 2) {
        force->addMonomial(i, (i+5)%31, -1, -1);
        force->addMonomial((i+5)%31, (i+9)%31, -1, -1);
        force->addTerm(-0.7*(i%4+1), m, 2);
        m += 2;
    }
    for (int i = 0; i < 31; i += 3) {
        force->addMonomial(i, (i+7)%31, (i+11)%31, -1);
        force->addTerm(0.9, m++, 1);
    }
    for (int i = 0; i < 31; i += 5) {
        force->addMonomial(i, (i+3)%31, (i+13)%31, (i+17)%31);
        force->addTerm(-1.3, m++, 1);
    }
    system.addForce(force);
    return force;
}

/**
 * A synthetic three-body polynomial: the 36 MB-pol variables (all site pairs of a trimer) with terms of every degree.
 */
static WaterTrimerPIPForce* addTrimerPIP(System& system, int nwat, double box) {
    WaterTrimerPIPForce* force = new WaterTrimerPIPForce();
    for (int w = 0; w < nwat; w++)
        force->addWater(4*w, 4*w+1, 4*w+2);
    force->setNonbondedMethod(box > 0 ? WaterTrimerPIPForce::CutoffPeriodic : WaterTrimerPIPForce::CutoffNonPeriodic);
    // MB-pol variable order: intramolecular H-H (3), O-H (6), intermolecular H-H (12), O-H (12), O-O (3)
    const int pairs[36][2] = {{1, 2}, {4, 5}, {7, 8}, {0, 1}, {0, 2}, {3, 4}, {3, 5}, {6, 7}, {6, 8},
                              {1, 4}, {1, 5}, {2, 4}, {2, 5}, {1, 7}, {1, 8}, {2, 7}, {2, 8}, {4, 7}, {4, 8}, {5, 7}, {5, 8},
                              {0, 4}, {0, 5}, {3, 1}, {3, 2}, {0, 7}, {0, 8}, {6, 1}, {6, 2}, {3, 7}, {3, 8}, {6, 4}, {6, 5},
                              {0, 3}, {0, 6}, {3, 6}};
    for (int i = 0; i < 36; i++)
        force->addVariable(pairs[i][0], pairs[i][1], 0.5, 1.5, 0.05*(i%5));
    int m = 0;
    force->addMonomial(-1, -1, -1, -1);
    force->addTerm(0.25, m++, 1);
    for (int i = 0; i < 36; i++) {
        force->addMonomial(i, -1, -1, -1);
        force->addTerm(0.4*(i%3+1), m++, 1);
    }
    for (int i = 0; i < 36; i += 2) {
        force->addMonomial(i, (i+5)%36, -1, -1);
        force->addMonomial((i+5)%36, (i+9)%36, -1, -1);
        force->addTerm(-0.6*(i%4+1), m, 2);
        m += 2;
    }
    for (int i = 0; i < 36; i += 3) {
        force->addMonomial(i, (i+7)%36, (i+11)%36, -1);
        force->addTerm(0.8, m++, 1);
    }
    for (int i = 0; i < 36; i += 5) {
        force->addMonomial(i, (i+3)%36, (i+13)%36, (i+17)%36);
        force->addTerm(-1.1, m++, 1);
    }
    system.addForce(force);
    return force;
}

/**
 * Single waters for PartridgeSchwenkeForce, 0.5 nm apart, with energies (kJ/mol) from an independent numpy evaluator
 * of the MB-pol one-body potential: the minimum region, stretched and dissociating bonds, a nearly linear and a
 * strongly bent molecule, and a compressed bond.
 */
static const int PS_NUM_WATERS = 7;
static const double PS_XYZ[][3] = {
    {0, 0, 0},
    {0.09572, 0, 0},
    {-0.023998720840903407, 0.092662720648599498, 0},
    {0.5, 0, 0},
    {0.60030000000000006, 0, 0},
    {0.4887758334723859, 0.091413500566165762, 0},
    {1, 0, 0},
    {1.25, 0, 0},
    {0.97600629351781809, 0.092643359444953735, 0},
    {1.5, 0, 0},
    {1.8999999999999999, 0, 0},
    {1.4672686722837336, 0.089928583809211426, 0},
    {2, 0, 0},
    {2.0956999999999999, 0, 0},
    {1.9038329653373514, 0.002518222427217423, 0},
    {2.5, 0, 0},
    {2.6150000000000002, 0, 0},
    {2.5610313031621659, 0.11478318707166049, 0},
    {3, 0, 0},
    {3.0699999999999998, 0, 0},
    {2.9525000000000001, 0.082272413359521679, 0},
};
// r1, r2 (nm), theta (deg): 0.09572 0.09572 104.52; 0.1003 0.0921 97; 0.25 0.0957 104.52; 0.4 0.0957 110; 0.0957 0.0962 178.5; 0.115 0.13 62; 0.07 0.095 120
static const double PS_REF_ENERGY[] = {0.002343607917, 12.103650065961, 485.683680705786, 505.145383025074, 136.759602805939, 262.802878592652, 334.699852630272};

static PartridgeSchwenkeForce* addPartridgeSchwenke(System& system, int nwat) {
    PartridgeSchwenkeForce* force = new PartridgeSchwenkeForce();
    for (int w = 0; w < nwat; w++)
        force->addWater(4*w, 4*w+1, 4*w+2);
    system.addForce(force);
    return force;
}

enum ForceKind {Electrostatics, DimerPIP, TrimerPIP, PartridgeSchwenke};

static void buildSystem(ForceKind kind, System& system, const double xyz[][3], int nwat, double box, vector<Vec3>& positions,
        bool redistribute = true, MBPolElectrostaticsForce::PolarizationType pol = MBPolElectrostaticsForce::Mutual, double eps = 1e-9) {
    buildWaters(system, xyz, nwat, positions, box);
    if (kind == Electrostatics)
        addElectrostatics(system, nwat, box, redistribute, pol, eps);
    else if (kind == DimerPIP)
        addDimerPIP(system, nwat, box);
    else if (kind == TrimerPIP)
        addTrimerPIP(system, nwat, box);
    else
        addPartridgeSchwenke(system, nwat);
}

/**
 * Energy and forces of a system on a platform.
 */
static double evaluate(ForceKind kind, Platform& platform, const double xyz[][3], int nwat, double box, vector<Vec3>& forces,
        const map<string, string>& props = map<string, string>(), bool redistribute = true,
        MBPolElectrostaticsForce::PolarizationType pol = MBPolElectrostaticsForce::Mutual, double eps = 1e-9,
        vector<Vec3>* induced = NULL, vector<double>* charges = NULL) {
    System system;
    vector<Vec3> positions;
    buildSystem(kind, system, xyz, nwat, box, positions, redistribute, pol, eps);
    VerletIntegrator integrator(0.0005);
    Context context(system, integrator, platform, props);
    context.setPositions(positions);
    context.computeVirtualSites();
    State state = context.getState(State::Energy | State::Forces);
    forces = state.getForces();
    if (kind == Electrostatics) {
        MBPolElectrostaticsForce& force = dynamic_cast<MBPolElectrostaticsForce&>(system.getForce(0));
        if (induced != NULL)
            force.getInducedDipoles(context, *induced);
        if (charges != NULL)
            force.getCharges(context, *charges);
    }
    return state.getPotentialEnergy();
}

/**
 * Largest error of the analytic forces on the given atoms against central finite differences, relative to
 * max(1, |F|).  Uses SOR without the predictor: DIIS solves for its coefficients in single precision, and the
 * resulting ~1e-7 kJ/mol energy noise would be amplified by the finite differences.
 */
static double finiteDifferenceCheck(ForceKind kind, Platform& platform, const double xyz[][3], int nwat, double box, const vector<int>& atoms,
        const map<string, string>& props = map<string, string>(), double h = 2e-5) {
    System system;
    vector<Vec3> positions;
    buildSystem(kind, system, xyz, nwat, box, positions);
    if (kind == Electrostatics) {
        MBPolElectrostaticsForce& force = dynamic_cast<MBPolElectrostaticsForce&>(system.getForce(0));
        force.setInducedDipoleMethod(MBPolElectrostaticsForce::SOR);
        force.setPredictorOrder(0);
    }
    VerletIntegrator integrator(0.0005);
    Context context(system, integrator, platform, props);
    context.setPositions(positions);
    context.computeVirtualSites();
    vector<Vec3> f = context.getState(State::Forces).getForces();
    double maxErr = 0;
    for (int a : atoms) {
        for (int k = 0; k < 3; k++) {
            vector<Vec3> p = positions;
            p[a][k] += h;
            context.setPositions(p);
            context.computeVirtualSites();
            double ep = context.getState(State::Energy).getPotentialEnergy();
            p[a][k] -= 2*h;
            context.setPositions(p);
            context.computeVirtualSites();
            double em = context.getState(State::Energy).getPotentialEnergy();
            double fd = -(ep-em)/(2*h);
            double err = fabs(fd-f[a][k]);
            double scale = max(1.0, fabs(f[a][k]));
            maxErr = max(maxErr, err/scale);
        }
    }
    return maxErr;
}

static const char* kindName(ForceKind kind) {
    return (kind == Electrostatics ? "MBPolElectrostaticsForce" : kind == DimerPIP ? "WaterDimerPIPForce" :
            kind == TrimerPIP ? "WaterTrimerPIPForce" : "PartridgeSchwenkeForce");
}

/**
 * Finite-difference forces of all four forces for 32 waters, as a cluster and in a periodic box.
 */
static void testForces(Platform& platform, const map<string, string>& props, double tolerance) {
    vector<int> atoms = {0, 1, 13, 50};
    for (int kind = Electrostatics; kind <= PartridgeSchwenke; kind++) {
        vector<Vec3> f;
        double e = evaluate((ForceKind) kind, platform, W32_XYZ, 32, 0.0, f, props);
        double err = finiteDifferenceCheck((ForceKind) kind, platform, W32_XYZ, 32, 0.0, atoms, props);
        printf("%-26s 32 waters, cluster: E = %14.6f kJ/mol, max relative FD force error %.2e\n", kindName((ForceKind) kind), e, err);
        ASSERT(err < tolerance);
        e = evaluate((ForceKind) kind, platform, W32_XYZ, 32, TEST_BOX_NM, f, props);
        err = finiteDifferenceCheck((ForceKind) kind, platform, W32_XYZ, 32, TEST_BOX_NM, atoms, props);
        printf("%-26s 32 waters, periodic: E = %14.6f kJ/mol, max relative FD force error %.2e\n", kindName((ForceKind) kind), e, err);
        ASSERT(err < tolerance);
    }
}

/**
 * Energies of the one-body potential against the reference values, and finite-difference forces away from the minimum.
 */
static void testPartridgeSchwenke(Platform& platform, const map<string, string>& props, double tolerance, double forceTolerance) {
    vector<Vec3> f;
    double total = 0;
    for (int w = 0; w < PS_NUM_WATERS; w++) {
        double e = evaluate(PartridgeSchwenke, platform, &PS_XYZ[3*w], 1, 0.0, f, props);
        printf("PartridgeSchwenkeForce water %d: E = %14.8f kJ/mol (reference %14.8f)\n", w, e, PS_REF_ENERGY[w]);
        ASSERT_EQUAL_TOL(PS_REF_ENERGY[w], e, tolerance);
        total += PS_REF_ENERGY[w];
    }
    double e = evaluate(PartridgeSchwenke, platform, PS_XYZ, PS_NUM_WATERS, 0.0, f, props);
    ASSERT_EQUAL_TOL(total, e, tolerance);
    vector<int> atoms;
    for (int i = 0; i < 4*PS_NUM_WATERS; i++)
        if (i%4 != 3)
            atoms.push_back(i);
    // Smaller step: on the steep stretched and compressed bonds the default 2e-5 nm gives a truncation error of 4e-4.
    double err = finiteDifferenceCheck(PartridgeSchwenke, platform, PS_XYZ, PS_NUM_WATERS, 0.0, atoms, props, 2e-6);
    printf("PartridgeSchwenkeForce %d waters: max relative FD force error %.2e\n", PS_NUM_WATERS, err);
    ASSERT(err < forceTolerance);
}

static void testElectrostaticsEnergies(Platform& platform, const map<string, string>& props, double tolerance) {
    vector<Vec3> f;
    vector<double> q;
    double e = evaluate(Electrostatics, platform, WATER3_XYZ, 3, 0.0, f, props, true, MBPolElectrostaticsForce::Mutual, 1e-10, NULL, &q);
    printf("water trimer: E = %.8f kJ/mol (reference %.8f)\n", e, REF_WATER3_CLUSTER);
    ASSERT_EQUAL_TOL(REF_WATER3_CLUSTER, e, tolerance);
    double qsum = 0;
    for (double qi : q)
        qsum += qi;
    ASSERT_EQUAL_TOL(0.0, qsum, 1e-6);
    e = evaluate(Electrostatics, platform, WATER3_XYZ, 3, 0.0, f, props, false, MBPolElectrostaticsForce::Mutual, 1e-10);
    printf("water trimer, fixed charges: E = %.8f kJ/mol (reference %.8f)\n", e, REF_WATER3_CLUSTER_FIXEDQ);
    ASSERT_EQUAL_TOL(REF_WATER3_CLUSTER_FIXEDQ, e, tolerance);
    e = evaluate(Electrostatics, platform, W32_XYZ, 32, 0.0, f, props, true, MBPolElectrostaticsForce::Mutual, 1e-10);
    printf("32 waters, cluster: E = %.8f kJ/mol (reference %.8f)\n", e, REF_W32_CLUSTER);
    ASSERT_EQUAL_TOL(REF_W32_CLUSTER, e, 10*tolerance);
}

static void testDynamics(Platform& platform, const map<string, string>& props) {
    System system;
    vector<Vec3> positions;
    buildWaters(system, W32_XYZ, 32, positions, TEST_BOX_NM);
    addElectrostatics(system, 32, TEST_BOX_NM, true, MBPolElectrostaticsForce::Mutual, 1e-5);
    addDimerPIP(system, 32, TEST_BOX_NM);
    addTrimerPIP(system, 32, TEST_BOX_NM);
    addPartridgeSchwenke(system, 32);
    VerletIntegrator integrator(0.0002);
    Context context(system, integrator, platform, props);
    context.setPositions(positions);
    context.computeVirtualSites();
    context.setVelocitiesToTemperature(300.0);
    State s0 = context.getState(State::Energy | State::Forces);
    for (int rep = 0; rep < 3; rep++) {
        State s = context.getState(State::Energy | State::Forces);
        ASSERT_EQUAL_TOL(s0.getPotentialEnergy(), s.getPotentialEnergy(), 1e-6);
    }
    integrator.step(20);
    State s = context.getState(State::Energy);
    ASSERT(!std::isnan(s.getPotentialEnergy()));
    printf("20 MD steps OK, E = %.6f kJ/mol\n", s.getPotentialEnergy());
}

#endif /*SAPTML_TEST_FORCES_H_*/
