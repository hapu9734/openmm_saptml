#ifndef OPENMM_WATERTRIMERPIPFORCE_H_
#define OPENMM_WATERTRIMERPIPFORCE_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "openmm/Force.h"
#include "internal/windowsExportSAPTML.h"
#include <vector>

namespace SAPTMLPlugin {

/**
 * This class implements a three-body correction for water in the form of the MB-pol three-body polynomial: a
 * permutationally invariant polynomial in exactly 36 variables, each a function of the distance between two of the
 * nine atoms of a triple of waters, multiplied by the MB-pol three-body switching function of the oxygen-oxygen
 * distances.
 *
 * To use this class, create a WaterTrimerPIPForce, define the polynomial with addVariable(), addMonomial() and
 * addTerm(), and call addWater() once for each water.  The sites of a triple are numbered 0 = O, 1 = H1, 2 = H2 for
 * the first water, 3-5 for the second and 6-8 for the third.
 *
 * The interaction is also switched off smoothly at short intermolecular O-O, O-H and H-H distances and when an
 * intramolecular O-H or H-H distance leaves its window; the default switches and windows are those of the SAPT-ML
 * model.  These switches assume the MB-pol variable order: variables 0-2 are the intramolecular H-H distances, 3-8
 * the intramolecular O-H, 9-20 the intermolecular H-H, 21-32 the intermolecular O-H, and 33-35 the O-O distances.
 */

class OPENMM_EXPORT_SAPTML WaterTrimerPIPForce : public OpenMM::Force {
public:
    /**
     * Methods for handling the cutoff.
     */
    enum NonbondedMethod {
        /**
         * Triples beyond the cutoff do not interact.
         */
        CutoffNonPeriodic = 0,
        /**
         * Periodic boundary conditions with the minimum image convention; triples beyond the cutoff do not interact.
         */
        CutoffPeriodic = 1
    };
    /**
     * Create a WaterTrimerPIPForce.
     */
    WaterTrimerPIPForce();
    /**
     * Get the number of waters that have been added to the force.
     */
    int getNumWaters() const {
        return waters.size();
    }
    /**
     * Get the number of variables of the polynomial.
     */
    int getNumVariables() const {
        return variables.size();
    }
    /**
     * Get the number of monomials of the polynomial.
     */
    int getNumMonomials() const {
        return monomials.size();
    }
    /**
     * Get the number of terms of the polynomial.
     */
    int getNumTerms() const {
        return terms.size();
    }
    /**
     * Add a water to the force.
     *
     * @param oxygen     the index of the oxygen particle
     * @param hydrogen1  the index of the first hydrogen particle
     * @param hydrogen2  the index of the second hydrogen particle
     * @return the index of the water that was added
     */
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Get the particles of a water, as passed to addWater().
     */
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    /**
     * Set the particles of a water, as for addWater().
     */
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Add a variable exp(-k (r - d0)) - center, with r the distance in Angstrom.  Centering the variables at their
     * typical values improves the accuracy in single precision.
     *
     * @param site1   the first site (0-8) of the distance
     * @param site2   the second site (0-8) of the distance
     * @param k       the exponent (in 1/Angstrom)
     * @param d0      the offset (in Angstrom)
     * @param center  the value subtracted from the exponential
     * @return the index of the variable that was added
     */
    int addVariable(int site1, int site2, double k, double d0, double center=0.0);
    /**
     * Get the parameters of a variable, as passed to addVariable().
     */
    void getVariableParameters(int index, int& site1, int& site2, double& k, double& d0, double& center) const;
    /**
     * Set the parameters of a variable, as for addVariable().
     */
    void setVariableParameters(int index, int site1, int site2, double k, double d0, double center);
    /**
     * Add a monomial, the product of up to four variables given by index; pass -1 for unused trailing variables.
     *
     * @return the index of the monomial that was added
     */
    int addMonomial(int variable1, int variable2, int variable3, int variable4);
    /**
     * Get the variables of a monomial, as passed to addMonomial().
     */
    void getMonomialParameters(int index, int& variable1, int& variable2, int& variable3, int& variable4) const;
    /**
     * Set the variables of a monomial, as for addMonomial().
     */
    void setMonomialParameters(int index, int variable1, int variable2, int variable3, int variable4);
    /**
     * Add a term: a coefficient times the sum of a contiguous range of monomials of the same degree (typically the
     * orbit of one monomial under the permutations of the trimer).
     *
     * @param coefficient     the coefficient (in kJ/mol)
     * @param firstMonomial   the index of the first monomial of the range
     * @param numMonomials    the number of monomials in the range
     * @return the index of the term that was added
     */
    int addTerm(double coefficient, int firstMonomial, int numMonomials);
    /**
     * Get the parameters of a term, as passed to addTerm().
     */
    void getTermParameters(int index, double& coefficient, int& firstMonomial, int& numMonomials) const;
    /**
     * Set the parameters of a term, as for addTerm().
     */
    void setTermParameters(int index, double coefficient, int firstMonomial, int numMonomials);
    /**
     * Get the method used for handling the cutoff.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling the cutoff.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff (in nm) of the oxygen-oxygen distances.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff (in nm) of the oxygen-oxygen distances: a triple interacts only when at least two of its
     * oxygen-oxygen distances are below it.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the oxygen-oxygen distance (in nm) at which the switching function starts to reduce the interaction.
     */
    double getSwitchingDistance() const;
    /**
     * Set the oxygen-oxygen distance (in nm, 0 <= switch < cutoff) at which the switching function starts to reduce
     * the interaction.  Each oxygen-oxygen pair contributes a factor s = (1 + cos(pi (r - switch)/(cutoff - switch)))/2
     * between the switching distance and the cutoff, and the interaction is multiplied by s_AB s_AC + s_AB s_BC +
     * s_AC s_BC.
     */
    void setSwitchingDistance(double distance);
    /**
     * Get the oxygen-oxygen distances (in nm) of the short-range switch.
     */
    void getOxygenOxygenSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the oxygen-oxygen distances (in nm) of the short-range switch: the interaction is zero below zeroDistance
     * and complete above fullDistance (0 < zeroDistance < fullDistance < cutoff).
     */
    void setOxygenOxygenSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the intermolecular oxygen-hydrogen distances (in nm) of the short-range switch.
     */
    void getIntermolecularOHSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the intermolecular oxygen-hydrogen distances (in nm) of the short-range switch, as for
     * setOxygenOxygenSwitch().  A fullDistance of 0 disables it.
     */
    void setIntermolecularOHSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the intermolecular hydrogen-hydrogen distances (in nm) of the short-range switch.
     */
    void getIntermolecularHHSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the intermolecular hydrogen-hydrogen distances (in nm) of the short-range switch, as for
     * setOxygenOxygenSwitch().  A fullDistance of 0 disables it.
     */
    void setIntermolecularHHSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the window (in nm) of intramolecular oxygen-hydrogen distances.
     */
    void getIntramolecularOHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    /**
     * Set the window (in nm) of intramolecular oxygen-hydrogen distances: the interaction is zero below lowerZero
     * and above upperZero, and complete between lowerFull and upperFull.
     */
    void setIntramolecularOHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    /**
     * Get the window (in nm) of intramolecular hydrogen-hydrogen distances.
     */
    void getIntramolecularHHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    /**
     * Set the window (in nm) of intramolecular hydrogen-hydrogen distances, as for setIntramolecularOHWindow().
     */
    void setIntramolecularHHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    /**
     * Get the maximum number of waters within the cutoff of any one water.
     */
    int getMaxNeighbors() const;
    /**
     * Set the maximum number of waters within the cutoff of any one water, which sizes the neighbor lists.  An
     * exception is thrown if it is exceeded.
     */
    void setMaxNeighbors(int maxNeighbors);
    /**
     * Get the number of force evaluations between neighbor-list overflow checks.
     */
    int getOverflowCheckInterval() const;
    /**
     * Set the number of force evaluations between neighbor-list overflow checks.  Each check synchronizes with the
     * GPU; the Reference platform checks every evaluation.
     */
    void setOverflowCheckInterval(int interval);
    /**
     * Get whether this force uses periodic boundary conditions (true for CutoffPeriodic).
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CutoffPeriodic;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class WaterInfo;
    class VariableInfo;
    class MonomialInfo;
    class TermInfo;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance, switchingDistance;
    double ooZero, ooFull, ohZero, ohFull, hhZero, hhFull;
    double ohWindow[4], hhWindow[4];
    int maxNeighbors, overflowCheckInterval;
    std::vector<WaterInfo> waters;
    std::vector<VariableInfo> variables;
    std::vector<MonomialInfo> monomials;
    std::vector<TermInfo> terms;
};

/**
 * Internal record of a water.
 * @private
 */
class WaterTrimerPIPForce::WaterInfo {
public:
    int oxygen, hydrogen1, hydrogen2;
    WaterInfo() : oxygen(-1), hydrogen1(-1), hydrogen2(-1) {
    }
    WaterInfo(int oxygen, int hydrogen1, int hydrogen2) : oxygen(oxygen), hydrogen1(hydrogen1), hydrogen2(hydrogen2) {
    }
};

/**
 * Internal record of a variable.
 * @private
 */
class WaterTrimerPIPForce::VariableInfo {
public:
    int site1, site2;
    double k, d0, center;
    VariableInfo() : site1(-1), site2(-1), k(0.0), d0(0.0), center(0.0) {
    }
    VariableInfo(int site1, int site2, double k, double d0, double center) : site1(site1), site2(site2), k(k), d0(d0), center(center) {
    }
};

/**
 * Internal record of a monomial.
 * @private
 */
class WaterTrimerPIPForce::MonomialInfo {
public:
    int variable1, variable2, variable3, variable4;
    MonomialInfo() : variable1(-1), variable2(-1), variable3(-1), variable4(-1) {
    }
    MonomialInfo(int variable1, int variable2, int variable3, int variable4) :
        variable1(variable1), variable2(variable2), variable3(variable3), variable4(variable4) {
    }
};

/**
 * Internal record of a term.
 * @private
 */
class WaterTrimerPIPForce::TermInfo {
public:
    double coefficient;
    int firstMonomial, numMonomials;
    TermInfo() : coefficient(0.0), firstMonomial(0), numMonomials(0) {
    }
    TermInfo(double coefficient, int firstMonomial, int numMonomials) :
        coefficient(coefficient), firstMonomial(firstMonomial), numMonomials(numMonomials) {
    }
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_WATERTRIMERPIPFORCE_H_*/
