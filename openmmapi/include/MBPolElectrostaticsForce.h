#ifndef OPENMM_MBPOLELECTROSTATICSFORCE_H_
#define OPENMM_MBPOLELECTROSTATICSFORCE_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportSAPTML.h"
#include <vector>

namespace SAPTMLPlugin {

/**
 * This class implements the MB-pol (TTM4-F) electrostatics of water: geometry-dependent point charges on the
 * hydrogens and the M site from the Partridge-Schwenke dipole moment surface, isotropic induced point dipoles on the
 * oxygen and the hydrogens, and TTM4-F Thole damping, a function of a u^4 with u = r/(A_i A_j)^(1/6) and separate
 * parameters a for charge-charge, charge-dipole, and intermolecular and intramolecular dipole-dipole interactions.
 * Permanent charges neither interact with nor polarize their own molecule.
 *
 * To use this class, create an MBPolElectrostaticsForce and call addWater() once for each water.  Every particle of
 * the System must belong to exactly one water, and each M site must be a virtual site: a ThreeParticleAverageSite of
 * O, H1, H2 with weights 0.573293118, 0.213353441, 0.213353441.  Use NoCutoff for clusters and PME for periodic
 * systems.
 */

class OPENMM_EXPORT_SAPTML MBPolElectrostaticsForce : public OpenMM::Force {
public:
    /**
     * Methods for handling long-range electrostatics.
     */
    enum NonbondedMethod {
        /**
         * No cutoff; for clusters.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions with particle-mesh Ewald summation.
         */
        PME = 1
    };
    /**
     * Ways of computing the induced dipoles.
     */
    enum PolarizationType {
        /**
         * Polarized by the permanent charges and by each other, iterated to self-consistency (the default).
         */
        Mutual = 0,
        /**
         * Polarized by the permanent charges only.  The forces are not the exact gradient of the energy, so use
         * this for energies only.
         */
        Direct = 1
    };
    /**
     * Iterative solvers for the mutual induction.
     */
    enum InducedDipoleMethod {
        /**
         * Direct inversion in the iterative subspace over the last getDIISHistorySize() iterations (the default).
         */
        DIIS = 0,
        /**
         * Successive over-relaxation with the parameter getSORParameter().
         */
        SOR = 1
    };
    /**
     * Create an MBPolElectrostaticsForce.
     */
    MBPolElectrostaticsForce();
    /**
     * Get the number of waters that have been added to the force.
     */
    int getNumWaters() const {
        return waters.size();
    }
    /**
     * Add a water to the force.
     *
     * @param oxygen     the index of the oxygen particle
     * @param hydrogen1  the index of the first hydrogen particle
     * @param hydrogen2  the index of the second hydrogen particle
     * @param msite      the index of the M site, which must be a virtual site
     * @return the index of the water that was added
     */
    int addWater(int oxygen, int hydrogen1, int hydrogen2, int msite);
    /**
     * Get the particles of a water, as passed to addWater().
     */
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2, int& msite) const;
    /**
     * Set the particles of a water, as for addWater().
     */
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2, int msite);
    /**
     * Get the method used for handling long-range electrostatics.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long-range electrostatics.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) for the real-space part of PME.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) for the real-space part of PME.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the error tolerance for Ewald summation.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation, used to select the PME parameters when they are not set
     * explicitly.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the PME separation parameter alpha (in 1/nm) and the number of grid points along each axis, as set with
     * setPMEParameters().
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the PME separation parameter alpha (in 1/nm) and the number of grid points along each axis.  If alpha is
     * 0 (the default), they are chosen from the Ewald error tolerance instead.
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the PME parameters actually used in a Context, which may differ from those requested because of platform
     * restrictions on the grid size.
     */
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the polarization type.
     */
    PolarizationType getPolarizationType() const;
    /**
     * Set the polarization type.
     */
    void setPolarizationType(PolarizationType type);
    /**
     * Get the maximum number of iterations for the mutual induced dipoles.
     */
    int getMutualInducedMaxIterations() const;
    /**
     * Set the maximum number of iterations for the mutual induced dipoles.
     */
    void setMutualInducedMaxIterations(int iterations);
    /**
     * Get the convergence target (in Debye) for the RMS change of the induced dipoles between iterations.
     */
    double getMutualInducedTargetEpsilon() const;
    /**
     * Set the convergence target (in Debye) for the RMS change of the induced dipoles between iterations.
     */
    void setMutualInducedTargetEpsilon(double epsilon);
    /**
     * Get the iterative solver for the mutual induction.
     */
    InducedDipoleMethod getInducedDipoleMethod() const;
    /**
     * Set the iterative solver for the mutual induction.  The Reference platform always uses SOR.
     */
    void setInducedDipoleMethod(InducedDipoleMethod method);
    /**
     * Get the number of previous iterations the DIIS solver extrapolates from.
     */
    int getDIISHistorySize() const;
    /**
     * Set the number of previous iterations the DIIS solver extrapolates from (2 to 32).
     */
    void setDIISHistorySize(int size);
    /**
     * Get the relaxation parameter of the SOR solver.
     */
    double getSORParameter() const;
    /**
     * Set the relaxation parameter of the SOR solver (0 < omega <= 1; 1 is a plain Jacobi iteration).
     */
    void setSORParameter(double omega);
    /**
     * Get the order of the extrapolated starting guess for the induced dipoles.
     */
    int getPredictorOrder() const;
    /**
     * Set the order (0 to 3) of the starting guess for the induced dipoles: 0 starts from the direct dipoles, 1 from
     * the converged dipoles of the previous force evaluation, and 2 or 3 from a linear or quadratic extrapolation of
     * previous ones.  Energy-only evaluations do not enter the history.  The Reference platform ignores this setting.
     */
    void setPredictorOrder(int order);
    /**
     * Get whether the charges depend on the geometry of the water (TTM4-F charge redistribution) or are fixed.
     */
    bool getIncludeChargeRedistribution() const;
    /**
     * Set whether the charges depend on the geometry of the water (TTM4-F charge redistribution) or are fixed.
     */
    void setIncludeChargeRedistribution(bool include);
    /**
     * Get the charges (in e) of the oxygen, each hydrogen and the M site used when charge redistribution is off.
     */
    void getFixedCharges(double& chargeO, double& chargeH, double& chargeM) const;
    /**
     * Set the charges (in e) of the oxygen, each hydrogen and the M site used when charge redistribution is off.
     */
    void setFixedCharges(double chargeO, double chargeH, double chargeM);
    /**
     * Get the polarizabilities (in nm^3) of the oxygen and each hydrogen.
     */
    void getPolarizabilities(double& alphaO, double& alphaH) const;
    /**
     * Set the polarizabilities (in nm^3) of the oxygen and each hydrogen.  The M site is not polarizable.
     */
    void setPolarizabilities(double alphaO, double alphaH);
    /**
     * Get the Thole damping factors A_i (in nm^3) of the oxygen, each hydrogen and the M site.
     */
    void getDampingFactors(double& dampO, double& dampH, double& dampM) const;
    /**
     * Set the Thole damping factors A_i (in nm^3) of the oxygen, each hydrogen and the M site.
     */
    void setDampingFactors(double dampO, double dampH, double dampM);
    /**
     * Get the Thole parameters a, as passed to setTholeParameters().
     */
    void getTholeParameters(double& cc, double& cd, double& dd, double& ddOH, double& ddHH) const;
    /**
     * Set the Thole parameters a.
     *
     * @param cc     charge-charge interactions
     * @param cd     charge-dipole interactions
     * @param dd     intermolecular dipole-dipole interactions
     * @param ddOH   intramolecular dipole-dipole interactions between the oxygen and a hydrogen
     * @param ddHH   intramolecular dipole-dipole interactions between the two hydrogens
     */
    void setTholeParameters(double cc, double cd, double dd, double ddOH, double ddHH);
    /**
     * Get the induced dipole moments (in e*nm) of all particles in a Context, indexed by particle.
     */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the current charges (in e) of all particles in a Context, indexed by particle.
     */
    void getCharges(OpenMM::Context& context, std::vector<double>& charges);
    /**
     * Update the water particles, fixed charges, polarizabilities and damping factors in an existing Context
     * without reinitializing it.  Other settings require reinitializing the Context.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Get whether this force uses periodic boundary conditions (true for PME).
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == PME;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class WaterInfo;
    NonbondedMethod nonbondedMethod;
    PolarizationType polarizationType;
    InducedDipoleMethod inducedDipoleMethod;
    double cutoffDistance, ewaldErrorTol, alpha;
    int nx, ny, nz;
    int mutualInducedMaxIterations, diisHistorySize, predictorOrder;
    double mutualInducedTargetEpsilon, sorParameter;
    bool includeChargeRedistribution;
    double fixedChargeO, fixedChargeH, fixedChargeM;
    double alphaO, alphaH;
    double dampO, dampH, dampM;
    double tholeCC, tholeCD, tholeDD, tholeDDOH, tholeDDHH;
    std::vector<WaterInfo> waters;
};

/**
 * Internal record of a water.
 * @private
 */
class MBPolElectrostaticsForce::WaterInfo {
public:
    int oxygen, hydrogen1, hydrogen2, msite;
    WaterInfo() : oxygen(-1), hydrogen1(-1), hydrogen2(-1), msite(-1) {
    }
    WaterInfo(int oxygen, int hydrogen1, int hydrogen2, int msite) :
        oxygen(oxygen), hydrogen1(hydrogen1), hydrogen2(hydrogen2), msite(msite) {
    }
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_MBPOLELECTROSTATICSFORCE_H_*/
