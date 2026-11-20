#ifndef OPENMM_MBPOLELECTROSTATICSFORCE_H_
#define OPENMM_MBPOLELECTROSTATICSFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportSAPTML.h"
#include <vector>

namespace SAPTMLPlugin {

/**
 * This class implements the MB-pol (TTM4-F) electrostatics of water: geometry-dependent point charges on the
 * hydrogens and on the M site, taken from the Partridge-Schwenke dipole moment surface, together with the forces
 * arising from their dependence on the atomic positions; isotropic point dipoles on the oxygen and the hydrogens
 * that are polarized by the permanent charges and by each other (mutual induction); and Thole damping of the
 * form 1-exp(-a u^4), u = r/(A_i A_j)^(1/6), with separate parameters for charge-charge, charge-dipole,
 * intermolecular dipole-dipole and intramolecular dipole-dipole interactions.  There are no intramolecular
 * interactions between permanent charges, and the permanent charges of a molecule do not polarize its own dipoles.
 *
 * To use this class, create an MBPolElectrostaticsForce object, then call addWater() once for each water molecule
 * in the System.  Every particle of the System must belong to exactly one water, and the M site of every water
 * must be a virtual site (a ThreeParticleAverageSite with the TTM4-F weights 0.573293118, 0.213353441, 0.213353441).
 * Clusters use NoCutoff; periodic systems use PME.
 *
 * The induced dipoles are determined iteratively until the RMS change of the dipoles falls below
 * getMutualInducedTargetEpsilon() (in Debye).  The iteration is accelerated either by DIIS extrapolation over the
 * last getDIISHistorySize() iterations (the default) or by successive over-relaxation with the parameter
 * getSORParameter().  On GPU platforms the starting guess of each iteration can be extrapolated from the converged
 * dipoles of the previous force evaluations (setPredictorOrder()).
 */

class OPENMM_EXPORT_SAPTML MBPolElectrostaticsForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range electrostatic interactions.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to interactions.  The full set of interactions is computed exactly.  This is appropriate
         * for clusters.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the
         * interaction of each particle with all periodic copies of every other particle.
         */
        PME = 1
    };
    /**
     * This is an enumeration of the different ways the induced dipoles can be computed.
     */
    enum PolarizationType {
        /**
         * The induced dipoles are polarized by the permanent charges and by each other, iterated to self-consistency.
         */
        Mutual = 0,
        /**
         * The induced dipoles are polarized by the permanent charges only.  The forces are not the exact gradient
         * of the energy, so this type should be used for energy evaluations only.
         */
        Direct = 1
    };
    /**
     * This is an enumeration of the iterative solvers available for the mutual induction.
     */
    enum InducedDipoleMethod {
        /**
         * Direct inversion in the iterative subspace: every iteration extrapolates the dipoles from the previous
         * iterations (up to getDIISHistorySize() of them).
         */
        DIIS = 0,
        /**
         * Successive over-relaxation: every iteration mixes the new dipoles with the old ones with the weight
         * getSORParameter().
         */
        SOR = 1
    };
    /**
     * Create an MBPolElectrostaticsForce.
     */
    MBPolElectrostaticsForce();
    /**
     * Get the number of water molecules for which force field parameters have been defined.
     */
    int getNumWaters() const {
        return waters.size();
    }
    /**
     * Add a water molecule to the force.
     *
     * @param oxygen     the index within the System of the oxygen
     * @param hydrogen1  the index within the System of the first hydrogen
     * @param hydrogen2  the index within the System of the second hydrogen
     * @param msite      the index within the System of the M site, which must be a virtual site
     * @return the index of the water that was added
     */
    int addWater(int oxygen, int hydrogen1, int hydrogen2, int msite);
    /**
     * Get the particles of a water molecule.
     *
     * @param index           the index of the water for which to get parameters
     * @param[out] oxygen     the index within the System of the oxygen
     * @param[out] hydrogen1  the index within the System of the first hydrogen
     * @param[out] hydrogen2  the index within the System of the second hydrogen
     * @param[out] msite      the index within the System of the M site
     */
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2, int& msite) const;
    /**
     * Set the particles of a water molecule.
     *
     * @param index      the index of the water for which to set parameters
     * @param oxygen     the index within the System of the oxygen
     * @param hydrogen1  the index within the System of the first hydrogen
     * @param hydrogen2  the index within the System of the second hydrogen
     * @param msite      the index within the System of the M site, which must be a virtual site
     */
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2, int msite);
    /**
     * Get the method used for handling long range electrostatic interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range electrostatic interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for the real space part of PME.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for the real space part of PME.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  It is used to select the PME parameters when they are not set explicitly.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  It is used to select the PME parameters when they are not set explicitly.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are ignored
     * and instead calculated based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are ignored
     * and instead calculated based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions on
     * the allowed grid sizes, the values that are actually used may be slightly different from those specified
     * with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
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
     * Get the maximum number of iterations to be used in calculating the mutual induced dipoles.
     */
    int getMutualInducedMaxIterations() const;
    /**
     * Set the maximum number of iterations to be used in calculating the mutual induced dipoles.
     */
    void setMutualInducedMaxIterations(int iterations);
    /**
     * Get the target epsilon (the RMS change of the induced dipoles between iterations, in Debye) to be used in
     * calculating the mutual induced dipoles.
     */
    double getMutualInducedTargetEpsilon() const;
    /**
     * Set the target epsilon (the RMS change of the induced dipoles between iterations, in Debye) to be used in
     * calculating the mutual induced dipoles.
     */
    void setMutualInducedTargetEpsilon(double epsilon);
    /**
     * Get the iterative solver used for the mutual induction.  The Reference platform always uses SOR.
     */
    InducedDipoleMethod getInducedDipoleMethod() const;
    /**
     * Set the iterative solver used for the mutual induction.  The Reference platform always uses SOR.
     */
    void setInducedDipoleMethod(InducedDipoleMethod method);
    /**
     * Get the number of previous iterations the DIIS solver extrapolates from.
     */
    int getDIISHistorySize() const;
    /**
     * Set the number of previous iterations the DIIS solver extrapolates from (between 2 and 32).
     */
    void setDIISHistorySize(int size);
    /**
     * Get the relaxation parameter of the SOR solver.
     */
    double getSORParameter() const;
    /**
     * Set the relaxation parameter of the SOR solver (between 0 and 1; 1 is a plain Jacobi iteration).
     */
    void setSORParameter(double omega);
    /**
     * Get the order of the extrapolation used for the starting guess of the mutual induction.
     */
    int getPredictorOrder() const;
    /**
     * Set the order of the extrapolation used for the starting guess of the mutual induction: 0 starts from the
     * dipoles induced by the permanent charges alone, 1 from the converged dipoles of the previous force
     * evaluation, and 2 or 3 from a linear or quadratic extrapolation of the converged dipoles of the previous
     * force evaluations.  Evaluations of the energy alone do not enter the history, and the convergence test is
     * not affected.  The Reference platform ignores this setting.
     */
    void setPredictorOrder(int order);
    /**
     * Get whether the charges depend on the geometry of the water (the TTM4-F charge model) or are fixed.
     */
    bool getIncludeChargeRedistribution() const;
    /**
     * Set whether the charges depend on the geometry of the water (the TTM4-F charge model) or are fixed.
     */
    void setIncludeChargeRedistribution(bool include);
    /**
     * Get the charges (in units of the proton charge) used when charge redistribution is off.
     *
     * @param[out] chargeO   the charge of the oxygen
     * @param[out] chargeH   the charge of each hydrogen
     * @param[out] chargeM   the charge of the M site
     */
    void getFixedCharges(double& chargeO, double& chargeH, double& chargeM) const;
    /**
     * Set the charges (in units of the proton charge) used when charge redistribution is off.
     *
     * @param chargeO   the charge of the oxygen
     * @param chargeH   the charge of each hydrogen
     * @param chargeM   the charge of the M site
     */
    void setFixedCharges(double chargeO, double chargeH, double chargeM);
    /**
     * Get the polarizabilities (in nm^3) of the oxygen and of each hydrogen.  The M site is not polarizable.
     */
    void getPolarizabilities(double& alphaO, double& alphaH) const;
    /**
     * Set the polarizabilities (in nm^3) of the oxygen and of each hydrogen.  The M site is not polarizable.
     */
    void setPolarizabilities(double alphaO, double alphaH);
    /**
     * Get the Thole damping factors A_i (in nm^3) of the oxygen, of each hydrogen and of the M site.  The damped
     * distance of a pair is u = r/(A_i A_j)^(1/6).
     */
    void getDampingFactors(double& dampO, double& dampH, double& dampM) const;
    /**
     * Set the Thole damping factors A_i (in nm^3) of the oxygen, of each hydrogen and of the M site.  The damped
     * distance of a pair is u = r/(A_i A_j)^(1/6).
     */
    void setDampingFactors(double dampO, double dampH, double dampM);
    /**
     * Get the Thole damping parameters a of the damping function 1-exp(-a u^4).
     *
     * @param[out] cc     charge-charge interactions
     * @param[out] cd     charge-dipole interactions
     * @param[out] dd     dipole-dipole interactions between molecules
     * @param[out] ddOH   dipole-dipole interactions between the oxygen and a hydrogen of one molecule
     * @param[out] ddHH   dipole-dipole interactions between the two hydrogens of one molecule
     */
    void getTholeParameters(double& cc, double& cd, double& dd, double& ddOH, double& ddHH) const;
    /**
     * Set the Thole damping parameters a of the damping function 1-exp(-a u^4).
     *
     * @param cc     charge-charge interactions
     * @param cd     charge-dipole interactions
     * @param dd     dipole-dipole interactions between molecules
     * @param ddOH   dipole-dipole interactions between the oxygen and a hydrogen of one molecule
     * @param ddHH   dipole-dipole interactions between the two hydrogens of one molecule
     */
    void setTholeParameters(double cc, double cd, double dd, double ddOH, double ddHH);
    /**
     * Get the induced dipole moments of all particles (in e*nm).
     *
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(OpenMM::Context& context, std::vector<OpenMM::Vec3>& dipoles);
    /**
     * Get the current (geometry-dependent) charges of all particles (in units of the proton charge).
     *
     * @param context    the Context for which to get the charges
     * @param charges    the charge of particle i is stored into the i'th element
     */
    void getCharges(OpenMM::Context& context, std::vector<double>& charges);
    /**
     * Update the parameters in a Context to match those stored in this Force object.  This method provides an
     * efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setWaterParameters() or the setters of the model parameters to modify this object's parameters,
     * then call updateParametersInContext() to copy them over to the Context.  The nonbonded method, the cutoff,
     * the PME parameters and the induction settings cannot be changed this way.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
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
 * This is an internal class used to record information about a water.
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
