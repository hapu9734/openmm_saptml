#ifndef OPENMM_PARTRIDGESCHWENKEFORCE_H_
#define OPENMM_PARTRIDGESCHWENKEFORCE_H_

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
 * This class implements the one-body (monomer) potential of water used by MB-pol: the Partridge-Schwenke potential
 * energy surface (J. Chem. Phys. 106, 4618 (1997)) with the MB-pol scaling of its components (J. Chem. Theory
 * Comput. 9, 5395 (2013)).  It is evaluated analytically, so it is defined for every geometry of the molecule: a
 * Morse term for each O-H bond that rises smoothly to the dissociation energy, an H-H repulsion, and a polynomial in
 * the bond lengths and the bond angle that is damped by a Gaussian of the bond length displacements.  The energy is
 * zero at the minimum of the surface.
 *
 * To use this class, create a PartridgeSchwenkeForce object and call addWater() once for each water molecule in the
 * System.  The force has no adjustable parameters.
 */

class OPENMM_EXPORT_SAPTML PartridgeSchwenkeForce : public OpenMM::Force {
public:
    /**
     * Create a PartridgeSchwenkeForce.
     */
    PartridgeSchwenkeForce();
    /**
     * Get the number of waters for which the potential is computed.
     */
    int getNumWaters() const {
        return waters.size();
    }
    /**
     * Add a water molecule.
     *
     * @param oxygen      the index of the oxygen particle
     * @param hydrogen1   the index of the first hydrogen particle
     * @param hydrogen2   the index of the second hydrogen particle
     * @return the index of the water that was added
     */
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Get the particles of a water molecule.
     *
     * @param index            the index of the water for which to get the particles
     * @param[out] oxygen      the index of the oxygen particle
     * @param[out] hydrogen1   the index of the first hydrogen particle
     * @param[out] hydrogen2   the index of the second hydrogen particle
     */
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    /**
     * Set the particles of a water molecule.
     *
     * @param index       the index of the water for which to set the particles
     * @param oxygen      the index of the oxygen particle
     * @param hydrogen1   the index of the first hydrogen particle
     * @param hydrogen2   the index of the second hydrogen particle
     */
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    /**
     * Set whether this force should apply periodic boundary conditions when calculating the intramolecular
     * displacements.  Usually this is not appropriate for bonded forces, but there are situations when it can be
     * useful.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Returns whether or not this force makes use of periodic boundary conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class WaterInfo;
    bool usePeriodic;
    std::vector<WaterInfo> waters;
};

/**
 * This is an internal class used to record information about a water.
 * @private
 */
class PartridgeSchwenkeForce::WaterInfo {
public:
    int oxygen, hydrogen1, hydrogen2;
    WaterInfo() : oxygen(-1), hydrogen1(-1), hydrogen2(-1) {
    }
    WaterInfo(int oxygen, int hydrogen1, int hydrogen2) : oxygen(oxygen), hydrogen1(hydrogen1), hydrogen2(hydrogen2) {
    }
};

} // namespace SAPTMLPlugin

#endif /*OPENMM_PARTRIDGESCHWENKEFORCE_H_*/
