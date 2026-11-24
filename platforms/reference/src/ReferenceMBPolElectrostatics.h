/* Portions copyright (c) 2006 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __ReferenceMBPolElectrostatics_H__
#define __ReferenceMBPolElectrostatics_H__

#include <vector>
#include "openmm/Vec3.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "fftpack.h"

using std::vector;

class int2 {
public:
    int2() {
        data[0] = data[1] = 0;
    }
    int2(int x, int y) {
        data[0] = x;
        data[1] = y;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 2);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 2);
        return data[index];
    }

private:
    int data[2];
};

class IntVec {
public:
    IntVec() {
        data[0] = data[1] = data[2] = 0;
    }
    IntVec(int x, int y, int z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 3);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 3);
        return data[index];
    }

private:
    int data[3];
};

namespace SAPTMLPlugin {

using namespace OpenMM;

enum TholeIndices { TCC, TCD, TDD, TDDOH, TDDHH };

/**
 * MB-pol-style water electrostatics: point charges (geometry-dependent when charge redistribution
 * is on) and Thole-damped induced dipoles. Adapted from the OpenMM AMOEBA reference implementation.
 * Units: nm, e, kJ/mol. ReferencePmeMBPolElectrostatics adds PME.
 */
class ReferenceMBPolElectrostatics {

public:

    enum PolarizationType {
        Mutual = 0,
        Direct = 1   // induced dipoles from the permanent field only, no iteration
    };

    enum ChargeDerivativesIndicesFinal { vsH1f, vsH2f, vsMf };

    ReferenceMBPolElectrostatics( );

    virtual ~ReferenceMBPolElectrostatics( ){};

    PolarizationType getPolarizationType( void ) const;

    void setPolarizationType( PolarizationType polarizationType );

    /** Nonzero if the last induced-dipole iteration converged. */
    int getMutualInducedDipoleConverged( void ) const;

    int getMutualInducedDipoleIterations( void ) const;

    void setIncludeChargeRedistribution( bool includeChargeRedistribution );

    /** Register the waters: site type per particle (0=O,1=H,2=M) and the (o,h1,h2,m) lists. */
    void setWaters( const std::vector<int>& siteTypes, const std::vector< std::vector<int> >& waters );

    const std::vector<OpenMM::Vec3>& getInducedDipoles( void ) const { return _inducedDipole; }
    const std::vector<double>& getCurrentCharges( void ) const { return _currentCharges; }

    bool getIncludeChargeRedistribution( void ) const;

    /** Final convergence measure: RMS per-site induced-dipole update (Debye). */
    double getMutualInducedDipoleEpsilon( void ) const;

    /** Convergence target, same measure as getMutualInducedDipoleEpsilon (Debye). */
    void setMutualInducedDipoleTargetEpsilon( double targetEpsilon );

    /** Set the over-relaxation parameter of the induced-dipole iteration (default 0.55; 1 = plain Jacobi). */
    void setPolarSOR( double polarSOR );

    double getMutualInducedDipoleTargetEpsilon( void ) const;

    void setMaximumMutualInducedDipoleIterations( int maximumMutualInducedDipoleIterations );

    int getMaximumMutualInducedDipoleIterations( void ) const;

    /**
     * Add forces to `forces` and return the energy (kJ/mol).
     * tholes: 5 per particle in TholeIndices order. multipoleAtomZs/Xs/Ys: the other three sites of
     * each particle's water (identify intramolecular pairs).
     */
    double calculateForceAndEnergy( const std::vector<OpenMM::Vec3>& particlePositions,
                                        const std::vector<double>& charges,
                                        const std::vector<double>& tholes,
                                        const std::vector<double>& dampingFactors,
                                        const std::vector<double>& polarity,
                                        const std::vector<int>& multipoleAtomZs,
                                        const std::vector<int>& multipoleAtomXs,
                                        const std::vector<int>& multipoleAtomYs,
                                        std::vector<OpenMM::Vec3>& forces );

protected:

    class ElectrostaticsParticleData {
        public:
            unsigned int particleIndex;
            Vec3 position;
            double charge;
            Vec3 chargeDerivatives[3];       // d q(otherSiteIndex[s]) / d position (e/nm), s = vsH1f, vsH2f, vsMf
            unsigned int otherSiteIndex[3];  // H1, H2, M site of this water
            double thole[5];                 // TholeIndices order
            double dampingFactor;
            double polarity;
            int multipoleAtomZs;             // Zs/Ys/Xs: the other three sites of this water
            int multipoleAtomYs;
            int multipoleAtomXs;
            int siteType;      // 0 = O, 1 = H, 2 = M
    };

    /* One induced-dipole set: its fixed field, its dipoles, and the field due to the induced dipoles. */
    struct UpdateInducedDipoleFieldStruct {
            UpdateInducedDipoleFieldStruct( std::vector<OpenMM::Vec3>* inputFixed_E_Field, std::vector<OpenMM::Vec3>* inputInducedDipoles );
            std::vector<OpenMM::Vec3>* fixedElectrostaticsField;
            std::vector<OpenMM::Vec3>* inducedDipoles;
            std::vector<OpenMM::Vec3> inducedDipoleField;
    };

    unsigned int _numParticles;

    PolarizationType _polarizationType;
    bool _includeChargeRedistribution;
    double _electric;

    // AMOEBA's d/p sets; no scaling is applied, so the two hold the same values (up to round-off with PME)
    std::vector<Vec3> _fixedElectrostaticsField;
    std::vector<Vec3> _fixedElectrostaticsFieldPolar;
    std::vector<Vec3> _inducedDipole;
    std::vector<Vec3> _inducedDipolePolar;

    int _mutualInducedDipoleConverged;
    int _mutualInducedDipoleIterations;
    int _maximumMutualInducedDipoleIterations;
    double  _mutualInducedDipoleEpsilon;
    double  _mutualInducedDipoleTargetEpsilon;
    double  _polarSOR;
    double  _debye;

    // water bookkeeping (set with setWaters)
    std::vector<int> _siteTypes;
    std::vector< std::vector<int> > _waters;      // o, h1, h2, m
    std::vector<double> _currentCharges;

    void loadParticleData( const std::vector<OpenMM::Vec3>& particlePositions,
                           const std::vector<double>& charges,
                           const std::vector<double>& tholes,
                           const std::vector<double>& dampingFactors,
                           const std::vector<double>& polarity,
                           const std::vector<int>& multipoleAtomZs,
                           const std::vector<int>& multipoleAtomXs,
                           const std::vector<int>& multipoleAtomYs,
                           std::vector<ElectrostaticsParticleData>& particleData ) const;

    void computeWaterCharge(ElectrostaticsParticleData& particleO, ElectrostaticsParticleData& particleH1,
                   ElectrostaticsParticleData& particleH2,ElectrostaticsParticleData& particleM);

    virtual void calculateFixedElectrostaticsField( const vector<ElectrostaticsParticleData>& particleData );

    void setMutualInducedDipoleConverged( int converged );

    void setMutualInducedDipoleIterations( int iterations );

    void setMutualInducedDipoleEpsilon( double epsilon );

    /**
     * Thole-damped (n-2)!!/r^n for n = interactionOrder (1, 3, 5, 7); with justScale, only the damping
     * factor. interactionType is a TholeIndices value.
     */
    double getAndScaleInverseRs(  const ElectrostaticsParticleData& particleI,
                                                                        const ElectrostaticsParticleData& particleK,
                                                              double r, bool justScale, int interactionOrder, int interactionType) const;

    void zeroFixedElectrostaticsFields( void );

    virtual void calculateFixedElectrostaticsFieldPairIxn( const ElectrostaticsParticleData& particleI, const ElectrostaticsParticleData& particleJ );

    virtual void initializeInducedDipoles( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields );

    /** field[I] += rr3*mu_J + rr5*(mu_J.delta)*delta and symmetrically for J; delta = r_J - r_I. */
    void calculateInducedDipolePairIxn( unsigned int particleI, unsigned int particleJ,
                                        double rr3, double rr5, const Vec3& delta,
                                        const std::vector<Vec3>& inducedDipole,
                                        std::vector<Vec3>& field ) const;

    void calculateInducedDipolePairIxns( const ElectrostaticsParticleData& particleI, const ElectrostaticsParticleData& particleJ,
                                         std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields );

    virtual void calculateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                               std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    void convergeInduceDipoles( const std::vector<ElectrostaticsParticleData>& particleData,
                                std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField );

    /** One SOR sweep over all dipole sets; returns the largest per-set sum of squared dipole changes. */
    double updateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                          std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);

    /**
     * SOR update toward fixedElectrostaticsField + polarity*inducedDipoleField (the fixed field already
     * includes the polarity); returns the sum of squared changes.
     */
    double updateInducedDipole( const std::vector<ElectrostaticsParticleData>& particleI,
                                    const std::vector<Vec3>& fixedElectrostaticsField,
                                    const std::vector<Vec3>& inducedDipoleField,
                                    std::vector<Vec3>& inducedDipoles);

    void calculateInducedDipoles( const std::vector<ElectrostaticsParticleData>& particleData );

    /** Load particle data, compute water charges (if enabled) and induced dipoles; throws if not converged. */
    void setup( const std::vector<OpenMM::Vec3>& particlePositions,
                const std::vector<double>& charges,
                const std::vector<double>& tholes,
                const std::vector<double>& dampingFactors,
                const std::vector<double>& polarity,
                const std::vector<int>& multipoleAtomZs,
                const std::vector<int>& multipoleAtomXs,
                const std::vector<int>& multipoleAtomYs,
                std::vector<ElectrostaticsParticleData>& particleData );

    double calculateElectrostaticPairIxn(
            const std::vector<ElectrostaticsParticleData>& particleData,
                                                                                    unsigned int iIndex,
                                                                                    unsigned int kIndex,
                                                                                    std::vector<OpenMM::Vec3>& forces) const;

    virtual double calculateElectrostatic( const std::vector<ElectrostaticsParticleData>& particleData,
                                               std::vector<OpenMM::Vec3>& forces );

    /** Resize to numParticles and zero. */
    void initializeVec3Vector( vector<Vec3>& vectorToInitialize ) const;
};

class ReferencePmeMBPolElectrostatics : public ReferenceMBPolElectrostatics {

public:

    ReferencePmeMBPolElectrostatics( void );

    ~ReferencePmeMBPolElectrostatics( );

    void setCutoffDistance( double cutoffDistance );

    /** Ewald alpha (nm^-1). */
    void setAlphaEwald( double alphaEwald );

    void setPmeGridDimensions( std::vector<int>& pmeGridDimensions );

    /** Rectangular box edge lengths (nm). */
     void setPeriodicBoxSize( Vec3& boxSize );

protected:

     void resizePmeArrays( void );

     double calculatePmeDirectElectrostaticPairIxn( const std::vector<ElectrostaticsParticleData>& particleData,
                            unsigned int iIndex, unsigned int jIndex,
                                                        std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential ) const;

private:

    static const int MBPOL_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;
    double _cutoffDistanceSquared;

    Vec3 _invPeriodicBoxSize;
    Vec3 _periodicBoxSize;

    int _totalGridSize;
    IntVec _pmeGridDimensions;

    fftpack_t   _fftplan;

    unsigned int _pmeGridSize;
    t_complex* _pmeGrid;

    std::vector<double> _pmeBsplineModuli[3];
    std::vector<Vec3> _thetai[3];
    std::vector<IntVec> _iGrid;
    std::vector<double> _phi;
    std::vector<double> _phid;
    std::vector<double> _phip;
    std::vector<double> _phidp;
    std::vector<int> _pmeAtomRange;
    std::vector<int2> _pmeAtomGridIndex;

    void initializePmeGrid( void );

    /** Apply minimum-image PBC to deltaR in place. */
    void getPeriodicDelta( Vec3& deltaR ) const;

    /** Grid points per nm along each axis. */
    void getPmeScale( Vec3& scale ) const;

    void initializeBSplineModuli( void );

    /** Direct-space (erfc-screened) fixed field at I due to J and vice versa. */
    void calculateFixedElectrostaticsFieldPairIxn( const ElectrostaticsParticleData& particleI, const ElectrostaticsParticleData& particleJ );

    void calculateFixedElectrostaticsField( const vector<ElectrostaticsParticleData>& particleData );

    /**
     * B-spline coefficients for one atom along one axis; w = offset from the grid point.
     * thetai[i] = (value, 1st, 2nd derivative).
     */
    void computeBSplinePoint(  std::vector<Vec3>& thetai, double w  );

    void computeMBPolBsplines( const std::vector<ElectrostaticsParticleData>& particleData );

    /** For each grid point, find the range of sorted atoms associated with that point. */
    void findMBPolAtomRangeForGrid( const vector<ElectrostaticsParticleData>& particleData );

    void getGridPointGivenGridIndex( int gridIndex, IntVec& gridPoint ) const;

    /** Charge contribution to gridPoint from the atoms in the particleGridIndices range (x/y offsets ix, iy). */
     double computeFixedElectrostaticssGridValue( const vector<ElectrostaticsParticleData>& particleData,
                                                 const int2& particleGridIndices, int ix, int iy, const IntVec& gridPoint ) const;

    void spreadFixedElectrostaticssOntoGrid( const vector<ElectrostaticsParticleData>& particleData );

    void performMBPolReciprocalConvolution( void );

    /** Reciprocal potential of the fixed charges and its first and second derivatives at each site (_phi, 10 per site). */
    void computeFixedPotentialFromGrid(void );

    /**
     * Reciprocal potential derivatives of the induced dipoles at each site: _phid (d set) and _phip (p set),
     * 10 per site; _phidp (d + p), potential and first derivatives, 4 per site.
     */
    void computeInducedPotentialFromGrid( void );

    double computeReciprocalSpaceFixedElectrostaticsForceAndEnergy( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                   std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential) const;

    /** Reciprocal fixed field from _phi; overwrites _fixedElectrostaticsField. */
    void recordFixedElectrostaticsField( void );

    void calculateReciprocalSpaceInducedDipoleField( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields );

    /** field[I] += preFactor1*mu_J + preFactor2*(mu_J.delta)*delta and symmetrically for J. */
    void calculateDirectInducedDipolePairIxn( unsigned int iIndex, unsigned int jIndex,
                                              double preFactor1, double preFactor2, const Vec3& delta,
                                              const std::vector<Vec3>& inducedDipole,
                                              std::vector<Vec3>& field ) const;

    void calculateDirectInducedDipolePairIxns( const ElectrostaticsParticleData& particleI,
                                               const ElectrostaticsParticleData& particleJ,
                                               std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields );

    void initializeInducedDipoles( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields );

    /** Grid value at gridPoint from the atoms in the atomIndices range: d set in re, p set in im. */
    t_complex computeInducedDipoleGridValue( const int2& atomIndices, const Vec3& scale, int ix, int iy, const IntVec& gridPoint,
                                             const std::vector<Vec3>& inputInducedDipole,
                                             const std::vector<Vec3>& inputInducedDipolePolar ) const;

    /** Spread the d set into the real part and the p set into the imaginary part of the grid. */
    void spreadInducedDipolesOnGrid( const std::vector<Vec3>& inputInducedDipole,
                                     const std::vector<Vec3>& inputInducedDipolePolar );

    void calculateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                       std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /** Add the reciprocal induced-dipole fields (from _phid, _phip) to field and fieldPolar. */
    void recordInducedDipoleField( vector<Vec3>& field, vector<Vec3>& fieldPolar );

    double calculatePmeSelfEnergy( const std::vector<ElectrostaticsParticleData>& particleData, std::vector<double>& electrostaticPotential ) const;

    /** Direct polarization omits the induced-induced reciprocal force terms. */
     double computeReciprocalSpaceInducedDipoleForceAndEnergy( ReferenceMBPolElectrostatics::PolarizationType polarizationType,
                                                                   const std::vector<ElectrostaticsParticleData>& particleData,
                                                                   std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential) const;

    double calculateElectrostatic( const std::vector<ElectrostaticsParticleData>& particleData,
                                       std::vector<OpenMM::Vec3>& forces );

};

} // namespace SAPTMLPlugin

#endif // __ReferenceMBPolElectrostatics_H__
