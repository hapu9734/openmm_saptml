#ifndef OPENMM_PARTRIDGESCHWENKEFORCE_H_
#define OPENMM_PARTRIDGESCHWENKEFORCE_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "openmm/Force.h"
#include "internal/windowsExportSAPTML.h"
#include <vector>

namespace SAPTMLPlugin {

/**
 * This class implements the one-body (monomer) potential of water used by MB-pol: the Partridge-Schwenke potential
 * energy surface (J. Chem. Phys. 106, 4618 (1997)) with the MB-pol scaling of its components (J. Chem. Theory
 * Comput. 9, 5395 (2013)), evaluated analytically.  It consists of a Morse term for each O-H bond, an H-H repulsion,
 * and a polynomial in the bond lengths and angle damped by a Gaussian of the bond length displacements.  The energy
 * is zero at the minimum.
 *
 * To use this class, create a PartridgeSchwenkeForce and call addWater() once for each water.  The force has no
 * adjustable parameters.
 */

class OPENMM_EXPORT_SAPTML PartridgeSchwenkeForce : public OpenMM::Force {
public:
    /**
     * Create a PartridgeSchwenkeForce.
     */
    PartridgeSchwenkeForce();
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
     * Set whether to apply periodic boundary conditions to the intramolecular displacements (off by default).
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get whether this force uses periodic boundary conditions.
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
 * Internal record of a water.
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
