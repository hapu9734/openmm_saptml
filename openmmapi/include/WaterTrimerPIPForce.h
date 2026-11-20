#ifndef OPENMM_WATERTRIMERPIPFORCE_H_
#define OPENMM_WATERTRIMERPIPFORCE_H_

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

#include "openmm/Force.h"
#include "internal/windowsExportSAPTML.h"
#include <vector>

namespace SAPTMLPlugin {

/**
 * This class implements a three-body correction for water in the form of the MB-pol three-body polynomial: a
 * permutationally invariant polynomial in 36 exponential variables of the distances between the three atoms of
 * each water of a triple, switched off smoothly as any of the three oxygen-oxygen distances approaches the
 * cutoff distance.
 *
 * To use this class, create a WaterTrimerPIPForce object, define the polynomial with addVariable(), addMonomial()
 * and addTerm(), and call addWater() once for each water molecule in the System.  The sites of a triple are
 * numbered 0 = O, 1 = H1, 2 = H2 of the first water, 3-5 of the second and 6-8 of the third water.  A variable
 * is exp(-k (r - d0)) - center, with r the distance (in Angstrom) between two sites; the center (0 by default)
 * lets a polynomial be expressed in variables shifted by their typical values, which keeps the evaluation in
 * single precision accurate.  A monomial is a product of up to four variables, and a term is a coefficient
 * multiplying the sum of a contiguous range of monomials (the orbit of one monomial under the permutations of
 * the trimer).
 *
 * The polynomial is a short-range correction fitted in the normal range of molecular geometries.  It is switched
 * off smoothly when two oxygens come closer than the oxygen-oxygen switch, when an intermolecular O-H or H-H
 * distance comes closer than the corresponding switch, and when an intramolecular O-H or H-H distance leaves its
 * window.  The defaults are those of the SAPT-ML water model.
 */

class OPENMM_EXPORT_SAPTML WaterTrimerPIPForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling the cutoff.
     */
    enum NonbondedMethod {
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 0,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic
         * copy of every other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 1
    };
    /**
     * Create a WaterTrimerPIPForce.
     */
    WaterTrimerPIPForce();
    /**
     * Get the number of water molecules for which force field parameters have been defined.
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
     * Add a water molecule to the force.
     *
     * @param oxygen     the index within the System of the oxygen
     * @param hydrogen1  the index within the System of the first hydrogen
     * @param hydrogen2  the index within the System of the second hydrogen
     * @return the index of the water that was added
     */
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Get the particles of a water molecule.
     *
     * @param index           the index of the water for which to get parameters
     * @param[out] oxygen     the index within the System of the oxygen
     * @param[out] hydrogen1  the index within the System of the first hydrogen
     * @param[out] hydrogen2  the index within the System of the second hydrogen
     */
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    /**
     * Set the particles of a water molecule.
     *
     * @param index      the index of the water for which to set parameters
     * @param oxygen     the index within the System of the oxygen
     * @param hydrogen1  the index within the System of the first hydrogen
     * @param hydrogen2  the index within the System of the second hydrogen
     */
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Add a variable to the polynomial.
     *
     * @param site1   the first site (0-8) of the pair whose distance the variable depends on
     * @param site2   the second site (0-8) of the pair whose distance the variable depends on
     * @param k       the exponent k (in 1/Angstrom)
     * @param d0      the offset d0 (in Angstrom)
     * @param center  the value subtracted from exp(-k (r - d0))
     * @return the index of the variable that was added
     */
    int addVariable(int site1, int site2, double k, double d0, double center=0.0);
    /**
     * Get the parameters of a variable.
     *
     * @param index        the index of the variable for which to get parameters
     * @param[out] site1   the first site (0-8) of the pair whose distance the variable depends on
     * @param[out] site2   the second site (0-8) of the pair whose distance the variable depends on
     * @param[out] k       the exponent k (in 1/Angstrom)
     * @param[out] d0      the offset d0 (in Angstrom)
     * @param[out] center  the value subtracted from exp(-k (r - d0))
     */
    void getVariableParameters(int index, int& site1, int& site2, double& k, double& d0, double& center) const;
    /**
     * Set the parameters of a variable.
     *
     * @param index   the index of the variable for which to set parameters
     * @param site1   the first site (0-8) of the pair whose distance the variable depends on
     * @param site2   the second site (0-8) of the pair whose distance the variable depends on
     * @param k       the exponent k (in 1/Angstrom)
     * @param d0      the offset d0 (in Angstrom)
     * @param center  the value subtracted from exp(-k (r - d0))
     */
    void setVariableParameters(int index, int site1, int site2, double k, double d0, double center);
    /**
     * Add a monomial to the polynomial.
     *
     * @param variable1   the index of the first variable
     * @param variable2   the index of the second variable, or -1 for a monomial of lower degree
     * @param variable3   the index of the third variable, or -1 for a monomial of lower degree
     * @param variable4   the index of the fourth variable, or -1 for a monomial of lower degree
     * @return the index of the monomial that was added
     */
    int addMonomial(int variable1, int variable2, int variable3, int variable4);
    /**
     * Get the variables of a monomial.
     *
     * @param index            the index of the monomial for which to get parameters
     * @param[out] variable1   the index of the first variable
     * @param[out] variable2   the index of the second variable, or -1 for a monomial of lower degree
     * @param[out] variable3   the index of the third variable, or -1 for a monomial of lower degree
     * @param[out] variable4   the index of the fourth variable, or -1 for a monomial of lower degree
     */
    void getMonomialParameters(int index, int& variable1, int& variable2, int& variable3, int& variable4) const;
    /**
     * Set the variables of a monomial.
     *
     * @param index       the index of the monomial for which to set parameters
     * @param variable1   the index of the first variable
     * @param variable2   the index of the second variable, or -1 for a monomial of lower degree
     * @param variable3   the index of the third variable, or -1 for a monomial of lower degree
     * @param variable4   the index of the fourth variable, or -1 for a monomial of lower degree
     */
    void setMonomialParameters(int index, int variable1, int variable2, int variable3, int variable4);
    /**
     * Add a term to the polynomial: a coefficient multiplying the sum of a contiguous range of monomials, all of
     * the same degree.
     *
     * @param coefficient     the coefficient (in kJ/mol)
     * @param firstMonomial   the index of the first monomial of the range
     * @param numMonomials    the number of monomials in the range
     * @return the index of the term that was added
     */
    int addTerm(double coefficient, int firstMonomial, int numMonomials);
    /**
     * Get the parameters of a term.
     *
     * @param index                the index of the term for which to get parameters
     * @param[out] coefficient     the coefficient (in kJ/mol)
     * @param[out] firstMonomial   the index of the first monomial of the range
     * @param[out] numMonomials    the number of monomials in the range
     */
    void getTermParameters(int index, double& coefficient, int& firstMonomial, int& numMonomials) const;
    /**
     * Set the parameters of a term.
     *
     * @param index           the index of the term for which to set parameters
     * @param coefficient     the coefficient (in kJ/mol)
     * @param firstMonomial   the index of the first monomial of the range
     * @param numMonomials    the number of monomials in the range
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
     * Get the cutoff distance (in nm): a triple interacts only when at least two of its oxygen-oxygen distances
     * are below it.
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm): a triple interacts only when at least two of its oxygen-oxygen distances
     * are below it.
     */
    void setCutoffDistance(double distance);
    /**
     * Get the oxygen-oxygen distance (in nm) at which the switching function starts to reduce the interaction.
     */
    double getSwitchingDistance() const;
    /**
     * Set the oxygen-oxygen distance (in nm) at which the switching function starts to reduce the interaction.
     * Each oxygen-oxygen pair of a triple contributes a factor (1 + cos(pi (r - switch)/(cutoff - switch)))/2
     * between the switching distance and the cutoff (1 below the switching distance, 0 beyond the cutoff), and
     * the interaction is multiplied by the sum over the three choices of the pair that is left out of the product
     * of the other two factors, as in MB-pol.
     */
    void setSwitchingDistance(double distance);
    /**
     * Get the oxygen-oxygen distances (in nm) below which the interaction is switched off: it is zero below
     * zeroDistance and complete above fullDistance.
     */
    void getOxygenOxygenSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the oxygen-oxygen distances (in nm) below which the interaction is switched off: it is zero below
     * zeroDistance and complete above fullDistance.
     */
    void setOxygenOxygenSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the intermolecular oxygen-hydrogen distances (in nm) below which the interaction is switched off.
     * Both zero means no switch.
     */
    void getIntermolecularOHSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the intermolecular oxygen-hydrogen distances (in nm) below which the interaction is switched off.
     * Both zero means no switch.
     */
    void setIntermolecularOHSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the intermolecular hydrogen-hydrogen distances (in nm) below which the interaction is switched off.
     * Both zero means no switch.
     */
    void getIntermolecularHHSwitch(double& zeroDistance, double& fullDistance) const;
    /**
     * Set the intermolecular hydrogen-hydrogen distances (in nm) below which the interaction is switched off.
     * Both zero means no switch.
     */
    void setIntermolecularHHSwitch(double zeroDistance, double fullDistance);
    /**
     * Get the window (in nm) of intramolecular oxygen-hydrogen distances in which the interaction is complete:
     * it is zero below lowerZero and above upperZero, and complete between lowerFull and upperFull.
     */
    void getIntramolecularOHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    /**
     * Set the window (in nm) of intramolecular oxygen-hydrogen distances in which the interaction is complete:
     * it is zero below lowerZero and above upperZero, and complete between lowerFull and upperFull.
     */
    void setIntramolecularOHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    /**
     * Get the window (in nm) of intramolecular hydrogen-hydrogen distances in which the interaction is complete.
     */
    void getIntramolecularHHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    /**
     * Set the window (in nm) of intramolecular hydrogen-hydrogen distances in which the interaction is complete.
     */
    void setIntramolecularHHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    /**
     * Get the maximum number of waters that may lie within the cutoff distance of any one water.
     */
    int getMaxNeighbors() const;
    /**
     * Set the maximum number of waters that may lie within the cutoff distance of any one water.  This sizes the
     * neighbor lists; an exception is thrown if it is exceeded.
     */
    void setMaxNeighbors(int maxNeighbors);
    /**
     * Get the number of force evaluations between checks of the neighbor list capacity on GPU platforms.
     */
    int getOverflowCheckInterval() const;
    /**
     * Set the number of force evaluations between checks of the neighbor list capacity on GPU platforms.  Every
     * check synchronizes with the device.
     */
    void setOverflowCheckInterval(int interval);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     *
     * @returns true if nonbondedMethod uses PBC and false otherwise
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
 * This is an internal class used to record information about a water.
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
 * This is an internal class used to record information about a variable.
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
 * This is an internal class used to record information about a monomial.
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
 * This is an internal class used to record information about a term.
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
