
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

#include "ReferenceMBPolElectrostatics.h"
#include "openmm/OpenMMException.h"
#include <sstream>
#include <algorithm>
#include <iostream>

// erf()/erfc() for old MSVC
#include "openmm/internal/MSVC_erfc.h"

// incomplete gamma function (from MB-pol)
#include "gammq.h"

using std::vector;
using OpenMM::Vec3;

#undef MBPOL_DEBUG

namespace SAPTMLPlugin {

ReferenceMBPolElectrostatics::ReferenceMBPolElectrostatics( ) :
                                                   _nonbondedMethod(NoCutoff),
                                                   _numParticles(0),
                                                   _electric(ONE_4PI_EPS0),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _polarSOR(0.55),
                                                   _debye(48.033324),
                                                   _includeChargeRedistribution(true)
{
    initialize();
}

ReferenceMBPolElectrostatics::ReferenceMBPolElectrostatics( NonbondedMethod nonbondedMethod ) :
                                                   _nonbondedMethod(NoCutoff),
                                                   _numParticles(0),
                                                   _electric(ONE_4PI_EPS0),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _polarSOR(0.55),
                                                   _debye(48.033324),
                                                   _includeChargeRedistribution(true)
{
    initialize();
}

void ReferenceMBPolElectrostatics::initialize( void )
{
    return;
}

ReferenceMBPolElectrostatics::NonbondedMethod ReferenceMBPolElectrostatics::getNonbondedMethod( void ) const
{
    return _nonbondedMethod;
}

void ReferenceMBPolElectrostatics::setNonbondedMethod( ReferenceMBPolElectrostatics::NonbondedMethod nonbondedMethod )
{
    _nonbondedMethod = nonbondedMethod;
}

ReferenceMBPolElectrostatics::PolarizationType ReferenceMBPolElectrostatics::getPolarizationType( void ) const
{
    return _polarizationType;
}

void ReferenceMBPolElectrostatics::setPolarizationType( ReferenceMBPolElectrostatics::PolarizationType polarizationType )
{
    _polarizationType = polarizationType;
}

void ReferenceMBPolElectrostatics::setWaters( const std::vector<int>& siteTypes, const std::vector<int>& waterIndices, const std::vector< std::vector<int> >& waters ) {
    _siteTypes = siteTypes;
    _waterIndices = waterIndices;
    _waters = waters;
}

void ReferenceMBPolElectrostatics::setIncludeChargeRedistribution( bool includeChargeRedistribution ) {
    _includeChargeRedistribution = includeChargeRedistribution;
}

bool ReferenceMBPolElectrostatics::getIncludeChargeRedistribution( void ) const
{
    return _includeChargeRedistribution;
}

int ReferenceMBPolElectrostatics::getMutualInducedDipoleConverged( void ) const
{
    return _mutualInducedDipoleConverged;
}

void ReferenceMBPolElectrostatics::setMutualInducedDipoleConverged( int mutualInducedDipoleConverged )
{
    _mutualInducedDipoleConverged = mutualInducedDipoleConverged;
}

int ReferenceMBPolElectrostatics::getMutualInducedDipoleIterations( void ) const
{
    return _mutualInducedDipoleIterations;
}

void ReferenceMBPolElectrostatics::setMutualInducedDipoleIterations( int mutualInducedDipoleIterations )
{
    _mutualInducedDipoleIterations = mutualInducedDipoleIterations;
}

double ReferenceMBPolElectrostatics::getMutualInducedDipoleEpsilon( void ) const
{
    return _mutualInducedDipoleEpsilon;
}

void ReferenceMBPolElectrostatics::setMutualInducedDipoleEpsilon( double mutualInducedDipoleEpsilon )
{
    _mutualInducedDipoleEpsilon = mutualInducedDipoleEpsilon;
}

int ReferenceMBPolElectrostatics::getMaximumMutualInducedDipoleIterations( void ) const
{
    return _maximumMutualInducedDipoleIterations;
}

void ReferenceMBPolElectrostatics::setMaximumMutualInducedDipoleIterations( int maximumMutualInducedDipoleIterations )
{
    _maximumMutualInducedDipoleIterations = maximumMutualInducedDipoleIterations;
}

double ReferenceMBPolElectrostatics::getMutualInducedDipoleTargetEpsilon( void ) const
{
    return _mutualInducedDipoleTargetEpsilon;
}

void ReferenceMBPolElectrostatics::setMutualInducedDipoleTargetEpsilon( double mutualInducedDipoleTargetEpsilon )
{
    _mutualInducedDipoleTargetEpsilon = mutualInducedDipoleTargetEpsilon;
}

void ReferenceMBPolElectrostatics::setPolarSOR( double polarSOR )
{
    _polarSOR = polarSOR;
}

double ReferenceMBPolElectrostatics::getPolarSOR( void ) const
{
    return _polarSOR;
}

double ReferenceMBPolElectrostatics::normalizeVec3( Vec3& vectorToNormalize ) const
{
    double norm = SQRT( vectorToNormalize.dot( vectorToNormalize ) );
    if( norm > 0.0 ){
        vectorToNormalize *= (1.0/norm);
    }
    return norm;
}

void ReferenceMBPolElectrostatics::initializeDoubleVector( vector<double>& vectorToInitialize ) const
{
    double zero = 0.0;
    vectorToInitialize.resize( _numParticles );
    std::fill( vectorToInitialize.begin(), vectorToInitialize.end(), zero );
}

void ReferenceMBPolElectrostatics::initializeVec3Vector( vector<Vec3>& vectorToInitialize ) const
{
    vectorToInitialize.resize( _numParticles );
    Vec3 zeroVec( 0.0, 0.0, 0.0 );
    std::fill( vectorToInitialize.begin(), vectorToInitialize.end(), zeroVec );
}

void ReferenceMBPolElectrostatics::copyVec3Vector( const std::vector<OpenMM::Vec3>& inputVector, std::vector<OpenMM::Vec3>& outputVector ) const
{
    outputVector.resize( inputVector.size() );
    for( unsigned int ii = 0; ii < inputVector.size(); ii++ ){
        outputVector[ii] = inputVector[ii];
    }
    return;
}

void ReferenceMBPolElectrostatics::loadParticleData( const std::vector<Vec3>& particlePositions,
                                                      const std::vector<double>& charges,
                                                      const std::vector<double>& tholes,
                                                      const std::vector<double>& dampingFactors,
                                                      const std::vector<double>& polarity,
                                                      const std::vector<int>& multipoleAtomZs,
                                                      const std::vector<int>& multipoleAtomXs,
                                                      const std::vector<int>& multipoleAtomYs,
                                                      std::vector<ElectrostaticsParticleData>& particleData ) const
{

    particleData.resize( _numParticles );
    for( unsigned int ii = 0; ii < _numParticles; ii++ ){

        particleData[ii].particleIndex        = ii;

        particleData[ii].position             = particlePositions[ii];
        particleData[ii].charge               = charges[ii];

        particleData[ii].thole[TCC]           = tholes[5*ii+0];
        particleData[ii].thole[TCD]           = tholes[5*ii+1];
        particleData[ii].thole[TDD]          = tholes[5*ii+2];
        particleData[ii].thole[TDDOH]         = tholes[5*ii+3];
        particleData[ii].thole[TDDHH]         = tholes[5*ii+4];
        particleData[ii].dampingFactor        = dampingFactors[ii];
        particleData[ii].polarity             = polarity[ii];

        particleData[ii].multipoleAtomZs = multipoleAtomZs[ii];
        particleData[ii].multipoleAtomXs = multipoleAtomXs[ii];
        particleData[ii].multipoleAtomYs = multipoleAtomYs[ii];
        particleData[ii].siteType        = (ii < _siteTypes.size() ? _siteTypes[ii] : 0);
        particleData[ii].waterIndex      = (ii < _waterIndices.size() ? _waterIndices[ii] : (int) (ii/4));

    }
}

void ReferenceMBPolElectrostatics::zeroFixedElectrostaticsFields( void )
{
    initializeVec3Vector( _fixedElectrostaticsField );
    initializeVec3Vector( _fixedElectrostaticsFieldPolar );
}

double ReferencePmeMBPolElectrostatics::ewaldScalingReal (  double r, int interactionOrder) const
{
    // real-space error function terms bn0..bn3

    double ralpha = _alphaEwald*r;
    double bn0    = erfc(ralpha)/r;

    double r2 = r*r;

    double alsq2  = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n = 0.0;
    if( _alphaEwald > 0.0 ){
        alsq2n = 1.0/(SQRT_PI*_alphaEwald);
    }
    double exp2a  = EXP(-(ralpha*ralpha));

    alsq2n           *= alsq2;
    double bn1    = (bn0+alsq2n*exp2a)/r2;

    alsq2n           *= alsq2;
    double bn2    = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n           *= alsq2;
    double bn3    = (5.0*bn2+alsq2n*exp2a)/r2;

    switch (interactionOrder) {

        case 1:
            return bn0;

        case 3:
            return bn1;

        case 5:
            return bn2;

        case 7:
            return bn3;

    }
    return bn0;
}

double ReferenceMBPolElectrostatics::getAndScaleInverseRs(  const ElectrostaticsParticleData& particleI,
                                                                    const ElectrostaticsParticleData& particleK,
                                                          double r, bool justScale, int interactionOrder, int interactionType) const
{

    bool isSameWater = (particleI.multipoleAtomZs == particleK.particleIndex) or
                (particleI.multipoleAtomYs == particleK.particleIndex) or
                (particleI.multipoleAtomXs == particleK.particleIndex);

    // rrI = (n-2)!!/r^n for n = interactionOrder (1 if justScale); Thole a from interactionType

    double rrI;
    if (justScale) {
        rrI         = 1.;
    } else {
        double rI             =  1.0/r;
        double r2I            =  rI*rI;
        rrI                    = rI;
        for( unsigned int ii  = 3; ii <= interactionOrder; ii=ii+2 ){
           rrI *= (ii-2)*r2I;
        }
    }

    double pgamma = std::min(particleI.thole[interactionType], particleK.thole[interactionType]);

    if (interactionType == TDD) {
        // dipole-dipole a: TDD between waters; within a water TDDOH if one site is O, else TDDHH

        if (isSameWater) {
            bool oneIsOxygen = (particleI.siteType == 0) or (particleK.siteType == 0);
            if (oneIsOxygen) {
                pgamma = std::min(particleI.thole[TDDOH],particleK.thole[TDDOH]);
            } else {
                pgamma = std::min(particleI.thole[TDDHH],particleK.thole[TDDHH]);
            }
        } else {
            pgamma = std::min(particleI.thole[TDD],particleK.thole[TDD]);
        }
    }

    // MB-pol Thole damping: A = (d_i d_j)^(1/6), u = r/A, s3 = 1 - exp(-a u^4);
    // s1 adds a^(1/4) u Gamma(3/4, a u^4) (upper incomplete gamma = Gamma(3/4)*gammq)
    double damp      = pow(particleI.dampingFactor*particleK.dampingFactor, 1/6.);

    if( (damp != 0.0) and ( damp > -50.0 ) ) {

        double ratio       = pow(r/damp, 4);
        double dampForExp = -1 * pgamma * ratio;

        switch (interactionOrder) {

        case 1:
            return rrI * (1.0 - EXP(dampForExp) + pow(pgamma, 1.0/4.0)*(r/damp)*EXP(ttm::gammln(3.0/4.0))*ttm::gammq(3.0/4.0, -dampForExp));

        case 3:
            return rrI * ( 1.0 - EXP(dampForExp) );

        case 5:
            return rrI * ((1.0 - EXP(dampForExp)) - (4./3.) * pgamma * EXP(dampForExp) * ratio);

        case 7:
            return rrI * (((1.0 - EXP(dampForExp)) - (4./3.) * pgamma * EXP(dampForExp) * ratio) -
                                (4./15.) * pgamma * (4. * pgamma * ratio - 1.) * EXP(dampForExp) / pow(damp, 4) * pow(r, 4));
        }
    }

    return rrI;
}

void ReferenceMBPolElectrostatics::calculateFixedElectrostaticsFieldPairIxn( const ElectrostaticsParticleData& particleI,
                                                                         const ElectrostaticsParticleData& particleJ,
                                                                         double dScale, double pScale )
{

    if( particleI.particleIndex == particleJ.particleIndex )return;

    // no permanent field from sites of the same water (multipoleAtomZs/Ys/Xs = the other three sites)

    bool isSameWater = (particleI.multipoleAtomZs == particleJ.particleIndex) or
            (particleI.multipoleAtomYs == particleJ.particleIndex) or
            (particleI.multipoleAtomXs == particleJ.particleIndex);
    if( isSameWater )return;

    Vec3 deltaR    = particleJ.position - particleI.position;
    double r      = SQRT( deltaR.dot( deltaR ) );

    // field of a charge: charge-dipole damping
    double rr3 = getAndScaleInverseRs( particleI, particleJ,r,false,3,TCD);

    // field at I due to J

    double factor                           = rr3*particleJ.charge;

    Vec3 field                               = deltaR*factor;

    unsigned int particleIndex                  = particleI.particleIndex;
    _fixedElectrostaticsField[particleIndex]        -= field*dScale;
    _fixedElectrostaticsFieldPolar[particleIndex]   -= field*pScale;

    // field at J due to I

    factor                                      = rr3*particleI.charge;

    field                                       = deltaR*factor;
    particleIndex                               = particleJ.particleIndex;
    _fixedElectrostaticsField[particleIndex]        += field*dScale;
    _fixedElectrostaticsFieldPolar[particleIndex]   += field*pScale;

    return;
}

void ReferenceMBPolElectrostatics::calculateFixedElectrostaticsField( const vector<ElectrostaticsParticleData>& particleData )
{

    // jj starts at ii; the pair routines skip ii == jj

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){
        for( unsigned int jj = ii; jj < _numParticles; jj++ ){

            double dScale, pScale;
                dScale = pScale = 1.0;
            calculateFixedElectrostaticsFieldPairIxn( particleData[ii], particleData[jj], dScale, pScale );
        }
    }
    return;
}

void ReferenceMBPolElectrostatics::initializeInducedDipoles( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields )
{

    _inducedDipole.resize( _numParticles );
    _inducedDipolePolar.resize( _numParticles );

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){
        _inducedDipole[ii]       = _fixedElectrostaticsField[ii];
        _inducedDipolePolar[ii]  = _fixedElectrostaticsFieldPolar[ii];
    }

    return;
}

void ReferenceMBPolElectrostatics::calculateInducedDipolePairIxn( unsigned int particleI,
                                                                   unsigned int particleJ,
                                                                   double rr3,
                                                                   double rr5,
                                                                   const Vec3& deltaR,
                                                                   const std::vector<Vec3>& inducedDipole,
                                                                   std::vector<Vec3>& field ) const
{

    double dDotDelta            = rr5*(inducedDipole[particleJ].dot( deltaR ) );
    field[particleI]               += inducedDipole[particleJ]*rr3 + deltaR*dDotDelta;
    dDotDelta                       = rr5*(inducedDipole[particleI].dot( deltaR ) );
    field[particleJ]               += inducedDipole[particleI]*rr3 + deltaR*dDotDelta;

    return;
}

void ReferenceMBPolElectrostatics::calculateInducedDipolePairIxns( const ElectrostaticsParticleData& particleI,
                                                                    const ElectrostaticsParticleData& particleJ,
                                                                    std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields )
{

   if( particleI.particleIndex == particleJ.particleIndex )return;

    Vec3 deltaR       = particleJ.position - particleI.position;
    double r         =  SQRT( deltaR.dot( deltaR ) );

    double scale3 = getAndScaleInverseRs( particleI, particleJ, r, false, 3, TDD);
    double scale5 = getAndScaleInverseRs( particleI, particleJ, r, false, 5, TDD);

    for( unsigned int ii = 0; ii < updateInducedDipoleFields.size(); ii++ ){
        calculateInducedDipolePairIxn( particleI.particleIndex, particleJ.particleIndex, -scale3, scale5, deltaR,
                                       *(updateInducedDipoleFields[ii].inducedDipoles), updateInducedDipoleFields[ii].inducedDipoleField );
    }
    return;

}

void ReferenceMBPolElectrostatics::calculateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                  std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        for( unsigned int jj = ii; jj < particleData.size(); jj++ ){
            calculateInducedDipolePairIxns( particleData[ii], particleData[jj], updateInducedDipoleFields );
        }
    }
    return;
}

double ReferenceMBPolElectrostatics::updateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                     std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    Vec3 zeroVec( 0.0, 0.0, 0.0 );
    for( unsigned int ii = 0; ii < updateInducedDipoleFields.size(); ii++ ){
        std::fill( updateInducedDipoleFields[ii].inducedDipoleField.begin(), updateInducedDipoleFields[ii].inducedDipoleField.end(), zeroVec );
    }

    calculateInducedDipoleFields( particleData, updateInducedDipoleFields);

    double maxEpsilon = 0.0;
    for( unsigned int kk = 0; kk < updateInducedDipoleFields.size(); kk++ ){
        double epsilon = updateInducedDipole( particleData,
                                                  *(updateInducedDipoleFields[kk].fixedElectrostaticsField),
                                                    updateInducedDipoleFields[kk].inducedDipoleField,
                                                  *(updateInducedDipoleFields[kk].inducedDipoles) );

        maxEpsilon = epsilon > maxEpsilon ? epsilon : maxEpsilon;
    }

    return maxEpsilon;
}

double ReferenceMBPolElectrostatics::updateInducedDipole( const std::vector<ElectrostaticsParticleData>& particleData,
                                                               const std::vector<Vec3>& fixedElectrostaticsField,
                                                               const std::vector<Vec3>& inducedDipoleField,
                                                               std::vector<Vec3>& inducedDipole )
{

    double epsilon                    = 0.0;
    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        Vec3    oldValue               = inducedDipole[ii];
        Vec3    newValue               = fixedElectrostaticsField[ii] + inducedDipoleField[ii]*particleData[ii].polarity;
        Vec3    delta                  = newValue - oldValue;
        inducedDipole[ii]                 = oldValue + delta*_polarSOR;
        epsilon                          += delta.dot( delta );
    }
    return epsilon;
}

void ReferenceMBPolElectrostatics::convergeInduceDipoles( const std::vector<ElectrostaticsParticleData>& particleData,
                                                           std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField)
{

    bool done                 = false;
    setMutualInducedDipoleConverged( false );
    int iteration             = 0;
    double currentEpsilon = 1.0e+50;

    // iterate until converged or max iterations; epsilon = RMS per-site dipole update in Debye (_debye: e*nm -> D)

    while( !done ){

        double epsilon = updateInducedDipoleFields( particleData, updateInducedDipoleField);
                   epsilon = _polarSOR*_debye*SQRT( epsilon/( static_cast<double>(_numParticles) ) );

        if( epsilon < getMutualInducedDipoleTargetEpsilon() ){
            setMutualInducedDipoleConverged( true );
            done = true;
        } else if( iteration >= getMaximumMutualInducedDipoleIterations() ){
            done = true;
        }
        // an increase of epsilon is not treated as divergence: near round-off it can fluctuate

        currentEpsilon = epsilon;
        iteration++;
    }
    setMutualInducedDipoleEpsilon( currentEpsilon );
    setMutualInducedDipoleIterations( iteration );

    return;
}

void ReferenceMBPolElectrostatics::calculateInducedDipoles( const std::vector<ElectrostaticsParticleData>& particleData )
{

    zeroFixedElectrostaticsFields();
    calculateFixedElectrostaticsField( particleData );

    // fixed fields -> direct induced dipoles alpha*E; Direct polarization stops there, Mutual iterates

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){
        _fixedElectrostaticsField[ii]      *= particleData[ii].polarity;
        _fixedElectrostaticsFieldPolar[ii] *= particleData[ii].polarity;
    }

    _inducedDipole.resize( _numParticles );
    _inducedDipolePolar.resize( _numParticles );
    std::vector<UpdateInducedDipoleFieldStruct> updateInducedDipoleField;
    updateInducedDipoleField.push_back( UpdateInducedDipoleFieldStruct( &_fixedElectrostaticsField,       &_inducedDipole ) );
    updateInducedDipoleField.push_back( UpdateInducedDipoleFieldStruct( &_fixedElectrostaticsFieldPolar,  &_inducedDipolePolar ) );

    initializeInducedDipoles( updateInducedDipoleField );

    if( getPolarizationType() == ReferenceMBPolElectrostatics::Direct ){
        setMutualInducedDipoleConverged( true );
        return;
    }

    convergeInduceDipoles( particleData, updateInducedDipoleField );

    return;
}

double ReferenceMBPolElectrostatics::calculateElectrostaticPairIxn( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                         unsigned int iIndex,
                                                                         unsigned int kIndex,
                                                                         std::vector<Vec3>& forces) const
{
    double temp3,temp5,temp7;
    double gl[9],gli[7],glip[7];
    double sc[10],sci[8],scip[8];
    double gf[7],gfi[6],gti[6];

    ElectrostaticsParticleData particleI = particleData[iIndex];
    ElectrostaticsParticleData particleK = particleData[kIndex];

    Vec3 delta       = particleK.position - particleI.position;
    double r2       = delta.dot( delta );

    double f        = _electric/_dielectric;

    double pdi      = particleI.dampingFactor;

    // undamped (n-2)!!/r^n; damping is applied below via the scale factors
    double r        = SQRT(r2);
    double rr1      = 1.0/r;
    double rr3      = rr1/r2;
    double rr5      = 3.0*rr3/r2;
    double rr7      = 5.0*rr5/r2;

    // scalar products of the induced dipoles (sci) and of the d/p sets (scip)

    sci[1] = _inducedDipole[iIndex][0]*_inducedDipole[kIndex][0]
       + _inducedDipole[iIndex][1]*_inducedDipole[kIndex][1]
       + _inducedDipole[iIndex][2]*_inducedDipole[kIndex][2];
    sci[2] = _inducedDipole[iIndex][0]*delta[0]
       + _inducedDipole[iIndex][1]*delta[1]
       + _inducedDipole[iIndex][2]*delta[2];
    sci[3] = _inducedDipole[kIndex][0]*delta[0]
       + _inducedDipole[kIndex][1]*delta[1]
       + _inducedDipole[kIndex][2]*delta[2];

    scip[1] = _inducedDipole[iIndex][0]*_inducedDipolePolar[kIndex][0]
        + _inducedDipole[iIndex][1]*_inducedDipolePolar[kIndex][1]
        + _inducedDipole[iIndex][2]*_inducedDipolePolar[kIndex][2]
        + _inducedDipolePolar[iIndex][0]*_inducedDipole[kIndex][0]
        + _inducedDipolePolar[iIndex][1]*_inducedDipole[kIndex][1]
        + _inducedDipolePolar[iIndex][2]*_inducedDipole[kIndex][2];
    scip[2] = _inducedDipolePolar[iIndex][0]*delta[0]
        + _inducedDipolePolar[iIndex][1]*delta[1]
        + _inducedDipolePolar[iIndex][2]*delta[2];
    scip[3] = _inducedDipolePolar[kIndex][0]*delta[0]
        + _inducedDipolePolar[kIndex][1]*delta[1]
        + _inducedDipolePolar[kIndex][2]*delta[2];

    // gl: permanent (charge-charge); gli/glip: charge-induced dipole

    gl[0] = particleI.charge*particleK.charge;

    gli[0] = particleK.charge*sci[2] - particleI.charge*sci[3];

    glip[0] = particleK.charge*scip[2] - particleI.charge*scip[3];

    bool isSameWater = (particleI.multipoleAtomZs == particleK.particleIndex) or
                       (particleI.multipoleAtomYs == particleK.particleIndex) or
                  (particleI.multipoleAtomXs == particleK.particleIndex);
    // same-water sites have no charge-charge and no charge-induced dipole interaction
    if( isSameWater ) {
        gl[0] = 0.;
        gli[0] = 0.;
        glip[0] = 0.;
    }

    double scale1CC = getAndScaleInverseRs( particleI, particleK, r, true, 1, TCC);
    double scale3CD = getAndScaleInverseRs( particleI, particleK, r, true, 3, TCD);

    if( !isSameWater ){
        _sitePotential[iIndex] += particleK.charge*rr1*scale1CC - sci[3]*rr3*scale3CD;
        _sitePotential[kIndex] += particleI.charge*rr1*scale1CC + sci[2]*rr3*scale3CD;
    }

    double energy =       rr1* gl[0]*scale1CC  ; // charge-charge
    energy           += 0.5*( rr3*gli[0]*scale3CD ); // charge - induced dipole
    energy           *= f;

    double scale3CC = getAndScaleInverseRs( particleI, particleK, r, true, 3, TCC);
    double scale5CD = getAndScaleInverseRs( particleI, particleK, r, true, 5, TCD);
    double scale5DD = getAndScaleInverseRs( particleI, particleK, r, true, 5, TDD);
    double scale7DD = getAndScaleInverseRs( particleI, particleK, r, true, 7, TDD);

    gf[0] = rr3*gl[0]*scale3CC ; // charge-charge

    gfi[0] = 0.5 * rr5 *  gli[0]*scale5CD + // charge - induced dipole
             0.5 * rr5 * glip[0]*scale5CD + // charge - induced dipole
             0.5 * rr5 * scip[1]*scale5DD + // induced dipole - induced dipole
           - 0.5 * rr7 * (sci[2]*scip[3] + scip[2]*sci[3])*scale7DD; // induced dipole - induced dipole

    // ftm2: permanent force, ftm2i: induced force

    Vec3 ftm2 = delta*gf[0];

    Vec3 ftm2i  = delta*gfi[0];

    ftm2i += ( _inducedDipolePolar[iIndex] *  sci[3] + // iPdipole_i * idipole_k
                    _inducedDipole[iIndex] * scip[3] +
               _inducedDipolePolar[kIndex] *  sci[2] + // iPdipole_k * idipole_i
                _inducedDipole[kIndex] * scip[2]  ) * 0.5 * rr5 * scale5DD;

    // same-water sites have no charge-induced dipole interaction
    if (not( isSameWater )) {
    ftm2i += (
                   ( _inducedDipole[iIndex] +
             _inducedDipolePolar[iIndex] )*-particleK.charge +
                   ( _inducedDipole[kIndex] +
             _inducedDipolePolar[kIndex] )* particleI.charge
                 ) * 0.5 * rr3 * scale3CD;
    }

    // forces from the geometry dependence of the charges (charge redistribution)

    if (getIncludeChargeRedistribution() and (not (isSameWater))){

        double distanceK, distanceI,
           scale1I, scale1K, scale3I, scale3K,
           inducedDipoleI, inducedDipoleK;
    Vec3 deltaI, deltaK;

        for (size_t s = 0; s < 3; ++s) {

            // s = vsH1f, vsH2f, vsMf: charge of site otherSiteIndex[s]

            deltaI = particleData[particleI.otherSiteIndex[s]].position
           - particleK.position;
            distanceI = SQRT(deltaI.dot(deltaI));
            deltaK = particleData[particleK.otherSiteIndex[s]].position
           - particleI.position;
            distanceK = SQRT(deltaK.dot(deltaK));

            scale1I = getAndScaleInverseRs( particleData[particleI.otherSiteIndex[s]], particleK, distanceI, true, 1, TCC );
            scale3I = getAndScaleInverseRs( particleData[particleI.otherSiteIndex[s]], particleK, distanceI, true, 3, TCC );

            scale1K = getAndScaleInverseRs( particleData[particleK.otherSiteIndex[s]], particleI, distanceK, true, 1, TCC );
            scale3K = getAndScaleInverseRs( particleData[particleK.otherSiteIndex[s]], particleI, distanceK, true, 3, TCC );

            inducedDipoleI = _inducedDipole[kIndex].dot(deltaI);
            inducedDipoleK = _inducedDipole[iIndex].dot(deltaK);

            for (size_t i = 0; i < 3; ++i) {

                ftm2[i] +=  scale1I * (1.0/distanceI) * particleI.chargeDerivatives[s][i] * particleK.charge; // charge - charge
                ftm2[i] -=  scale1K * (1.0/distanceK) * particleK.chargeDerivatives[s][i] * particleI.charge; // charge - charge

                ftm2i[i] += scale3I * pow(1.0/distanceI,3) * particleI.chargeDerivatives[s][i] * inducedDipoleI; // charge - induced dipole
                ftm2i[i] -= scale3K * pow(1.0/distanceK,3) * particleK.chargeDerivatives[s][i] * inducedDipoleK; // charge - induced dipole

            }

        }
    }

    Vec3 force   = ftm2 + ftm2i;
            force  *= f;

    forces[iIndex] -= force;
    forces[kIndex] += force;

    return energy;
}

double ReferenceMBPolElectrostatics::calculateElectrostatic( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                  std::vector<Vec3>& forces )
{

    double energy = 0.0;

    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        for( unsigned int jj = ii+1; jj < particleData.size(); jj++ ){

            energy += calculateElectrostaticPairIxn( particleData, ii, jj, forces);

        }
    }

    return energy;
}

void ReferenceMBPolElectrostatics::setup( const std::vector<Vec3>& particlePositions,
                                           const std::vector<double>& charges,
                                           const std::vector<double>& tholes,
                                           const std::vector<double>& dampingFactors,
                                           const std::vector<double>& polarity,
                                           const std::vector<int>& axisTypes,
                                           const std::vector<int>& multipoleAtomZs,
                                           const std::vector<int>& multipoleAtomXs,
                                           const std::vector<int>& multipoleAtomYs,
                                           const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                           std::vector<ElectrostaticsParticleData>& particleData )
{

    _numParticles = particlePositions.size();
    loadParticleData( particlePositions, charges,
                      tholes, dampingFactors, polarity, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, particleData );

    if (getIncludeChargeRedistribution())
    {
        for( unsigned int w = 0; w < _waters.size(); w++ ){
            computeWaterCharge(particleData[_waters[w][0]], particleData[_waters[w][1]], particleData[_waters[w][2]], particleData[_waters[w][3]]);
        }
    }
    _currentCharges.resize( _numParticles );
    for( unsigned int ii = 0; ii < _numParticles; ii++ )
        _currentCharges[ii] = particleData[ii].charge;
    _sitePotential.assign( _numParticles, 0.0 );

    calculateInducedDipoles( particleData );

    if( !getMutualInducedDipoleConverged() ){
        std::stringstream message;
        message << "Induced dipoles did not converge: ";
        message << " iterations="      << getMutualInducedDipoleIterations();
        message << " eps="             << getMutualInducedDipoleEpsilon();
        throw OpenMMException(message.str());
    }

    return;
}

double ReferenceMBPolElectrostatics::calculateForceAndEnergy( const std::vector<Vec3>& particlePositions,
                                                                   const std::vector<double>& charges,
                                                                   const std::vector<double>& tholes,
                                                                   const std::vector<double>& dampingFactors,
                                                                   const std::vector<double>& polarity,
                                                                   const std::vector<int>& axisTypes,
                                                                   const std::vector<int>& multipoleAtomZs,
                                                                   const std::vector<int>& multipoleAtomXs,
                                                                   const std::vector<int>& multipoleAtomYs,
                                                                   const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                                                   std::vector<Vec3>& forces )
{

    std::vector<ElectrostaticsParticleData> particleData;
    setup( particlePositions, charges, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
            multipoleAtomCovalentInfo, particleData );

    double energy = calculateElectrostatic( particleData, forces );

    return energy;
}

void ReferenceMBPolElectrostatics::calculateMBPolSystemElectrostaticsMoments( const std::vector<double>& masses,
                                                                           const std::vector<Vec3>& particlePositions,
                                                                           const std::vector<double>& charges,
                                                                           const std::vector<double>& tholes,
                                                                           const std::vector<double>& dampingFactors,
                                                                           const std::vector<double>& polarity,
                                                                           const std::vector<int>& axisTypes,
                                                                           const std::vector<int>& multipoleAtomZs,
                                                                           const std::vector<int>& multipoleAtomXs,
                                                                           const std::vector<int>& multipoleAtomYs,
                                                                           const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                                                           std::vector<double>& outputElectrostaticsMoments )
{

    std::vector<ElectrostaticsParticleData> particleData;
    setup( particlePositions, charges, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData );

    double totalMass  = 0.0;
    Vec3 centerOfMass  = Vec3( 0.0, 0.0, 0.0 );
    for( unsigned int ii  = 0; ii < _numParticles; ii++ ){
        double mass   = masses[ii];
        totalMass        += mass;
        centerOfMass     += particleData[ii].position*mass;
    }
    vector<Vec3> localPositions( _numParticles );
    if( totalMass > 0.0 ){
        centerOfMass  *= 1.0/totalMass;
    }
    for( unsigned int ii  = 0; ii < _numParticles; ii++ ){
        localPositions[ii] = particleData[ii].position - centerOfMass;
    }

    double netchg  = 0.0;

    Vec3 dpl        = Vec3( 0.0, 0.0, 0.0 );

    double xxqdp   = 0.0;
    double xyqdp   = 0.0;
    double xzqdp   = 0.0;

    double yyqdp   = 0.0;
    double yzqdp   = 0.0;

    double zzqdp   = 0.0;

    for( unsigned int ii  = 0; ii < _numParticles; ii++ ){

        double charge         = particleData[ii].charge;
        Vec3 position          = localPositions[ii];
        netchg                   += charge;

        Vec3 netDipole         = (_inducedDipole[ii]);

        dpl                      += position*charge + netDipole;

        xxqdp                    += position[0]*position[0]*charge + 2.0*position[0]*netDipole[0];
        xyqdp                    += position[0]*position[1]*charge + position[0]*netDipole[1] + position[1]*netDipole[0];
        xzqdp                    += position[0]*position[2]*charge + position[0]*netDipole[2] + position[2]*netDipole[0];

        yyqdp                    += position[1]*position[1]*charge + 2.0*position[1]*netDipole[1];
        yzqdp                    += position[1]*position[2]*charge + position[1]*netDipole[2] + position[2]*netDipole[1];

        zzqdp                    += position[2]*position[2]*charge + 2.0*position[2]*netDipole[2];

    }

    outputElectrostaticsMoments.resize( 13 );
    double qave                  = (xxqdp + yyqdp + zzqdp)/3.0;
    outputElectrostaticsMoments[4]        = 0.5*(xxqdp-qave);
    outputElectrostaticsMoments[5]        = 0.5*xyqdp;
    outputElectrostaticsMoments[6]        = 0.5*xzqdp;
    outputElectrostaticsMoments[8]        = 0.5*(yyqdp-qave);
    outputElectrostaticsMoments[9]        = 0.5*yzqdp;
    outputElectrostaticsMoments[12]       = 0.5*(zzqdp-qave);

    outputElectrostaticsMoments[7]  = outputElectrostaticsMoments[5];
    outputElectrostaticsMoments[10] = outputElectrostaticsMoments[6];
    outputElectrostaticsMoments[11] = outputElectrostaticsMoments[9];

    double debye           = 4.80321;

    outputElectrostaticsMoments[0]  = netchg;

    dpl                       *= 10.0*debye;
    outputElectrostaticsMoments[1]  = dpl[0];
    outputElectrostaticsMoments[2]  = dpl[1];
    outputElectrostaticsMoments[3]  = dpl[2];

    debye *= 3.0;
    for( unsigned int ii = 4; ii < 13; ii++ ){
        outputElectrostaticsMoments[ii] *= 100.0*debye;
    }

    return;
}

double ReferenceMBPolElectrostatics::calculateElectrostaticPotentialForParticleGridPoint( const ElectrostaticsParticleData& particleI, const Vec3& gridPoint ) const
{

    Vec3 deltaR           = particleI.position - gridPoint;

    getPeriodicDelta( deltaR );

    double r2            = deltaR.dot( deltaR );
    double r             = SQRT( r2 );

    double rr1           = 1.0/r;
    double potential     = particleI.charge*rr1;

    double rr2           = rr1*rr1;
    double rr3           = rr1*rr2;

    double scu           = _inducedDipole[particleI.particleIndex].dot( deltaR );
    potential               -= (scu)*rr3;

    double rr5           = 3.0*rr3*rr2;

    return potential;

}

void ReferenceMBPolElectrostatics::calculateElectrostaticPotential( const std::vector<Vec3>& particlePositions,
                                                                     const std::vector<double>& charges,
                                                                     const std::vector<double>& tholes,
                                                                     const std::vector<double>& dampingFactors,
                                                                     const std::vector<double>& polarity,
                                                                     const std::vector<int>& axisTypes,
                                                                     const std::vector<int>& multipoleAtomZs,
                                                                     const std::vector<int>& multipoleAtomXs,
                                                                     const std::vector<int>& multipoleAtomYs,
                                                                     const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                                                     const std::vector<Vec3>& grid,
                                                                     std::vector<double>& potential )
{

    std::vector<ElectrostaticsParticleData> particleData;
    setup( particlePositions, charges, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData );

    potential.resize( grid.size() );
    for( unsigned int ii = 0; ii < grid.size(); ii++ ){
        potential[ii] = 0.0;
    }

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){
        for( unsigned int jj = 0; jj < grid.size(); jj++ ){
            potential[jj] += calculateElectrostaticPotentialForParticleGridPoint( particleData[ii], grid[jj]  );
        }
    }

    double term = _electric/_dielectric;
    for( unsigned int ii = 0; ii < grid.size(); ii++ ){
        potential[ii] *= term;
    }

    return;
}

ReferenceMBPolElectrostatics::UpdateInducedDipoleFieldStruct::UpdateInducedDipoleFieldStruct( std::vector<OpenMM::Vec3>* inputFixed_E_Field, std::vector<OpenMM::Vec3>* inputInducedDipoles )
{
    fixedElectrostaticsField  = inputFixed_E_Field;
    inducedDipoles = inputInducedDipoles;
    inducedDipoleField.resize( fixedElectrostaticsField->size() );
}

const int ReferencePmeMBPolElectrostatics::MBPOL_PME_ORDER = 5;

const double ReferencePmeMBPolElectrostatics::SQRT_PI = 1.77245385091;

ReferencePmeMBPolElectrostatics::ReferencePmeMBPolElectrostatics( void ) :
               ReferenceMBPolElectrostatics(PME),
               _cutoffDistance(0.9), _cutoffDistanceSquared(0.81),
               _pmeGridSize(0), _totalGridSize(0), _alphaEwald(0.0)
{

    _fftplan = NULL;
    _pmeGrid = NULL;
    _pmeGridDimensions = IntVec( -1, -1, -1 );
}

ReferencePmeMBPolElectrostatics::~ReferencePmeMBPolElectrostatics( )
{
    if( _fftplan ){
        fftpack_destroy( _fftplan );
    }
    if( _pmeGrid ){
        delete[] _pmeGrid;
    }
};

double ReferencePmeMBPolElectrostatics::getCutoffDistance( void ) const
{
     return _cutoffDistance;
};

void ReferencePmeMBPolElectrostatics::setCutoffDistance( double cutoffDistance )
{
     _cutoffDistance        = cutoffDistance;
     _cutoffDistanceSquared = cutoffDistance*cutoffDistance;
};

double ReferencePmeMBPolElectrostatics::getAlphaEwald( void ) const
{
     return _alphaEwald;
};

void ReferencePmeMBPolElectrostatics::setAlphaEwald( double alphaEwald )
{
     _alphaEwald = alphaEwald;
};

void ReferencePmeMBPolElectrostatics::getPmeGridDimensions( std::vector<int>& pmeGridDimensions ) const
{

    pmeGridDimensions.resize( 3 );

    pmeGridDimensions[0] = _pmeGridDimensions[0];
    pmeGridDimensions[1] = _pmeGridDimensions[1];
    pmeGridDimensions[2] = _pmeGridDimensions[2];

    return;
};

void ReferencePmeMBPolElectrostatics::setPmeGridDimensions( std::vector<int>& pmeGridDimensions )
{

    if( (pmeGridDimensions[0] == _pmeGridDimensions[0]) &&
        (pmeGridDimensions[1] == _pmeGridDimensions[1]) &&
        (pmeGridDimensions[2] == _pmeGridDimensions[2]) )return;

    if( _fftplan ){
        fftpack_destroy(_fftplan);
    }
    fftpack_init_3d(&_fftplan,pmeGridDimensions[0], pmeGridDimensions[1], pmeGridDimensions[2]);

    _pmeGridDimensions[0] = pmeGridDimensions[0];
    _pmeGridDimensions[1] = pmeGridDimensions[1];
    _pmeGridDimensions[2] = pmeGridDimensions[2];

    initializeBSplineModuli( );
};

void ReferencePmeMBPolElectrostatics::setPeriodicBoxSize( Vec3& boxSize )
{

    if( boxSize[0] == 0.0 ||  boxSize[1] == 0.0 ||  boxSize[2] == 0.0 ){
        std::stringstream message;
        message << "Box size of zero is invalid.";
        throw OpenMMException(message.str());
    }

    _periodicBoxSize       = boxSize;

    _invPeriodicBoxSize[0] = 1.0/boxSize[0];
    _invPeriodicBoxSize[1] = 1.0/boxSize[1];
    _invPeriodicBoxSize[2] = 1.0/boxSize[2];

    return;
};

int compareInt2( const int2& v1, const int2& v2 )
{
    return v1[1] < v2[1];
}

void ReferencePmeMBPolElectrostatics::resizePmeArrays( void )
{

    _totalGridSize = _pmeGridDimensions[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2];
    if( _pmeGridSize < _totalGridSize ){
        if( _pmeGrid ){
            delete[] _pmeGrid;
        }
        _pmeGrid      = new t_complex[_totalGridSize];
        _pmeGridSize  = _totalGridSize;
    }

    for( unsigned int ii = 0; ii < 3; ii++ ){
       _pmeBsplineModuli[ii].resize( _pmeGridDimensions[ii] );
       _thetai[ii].resize( MBPOL_PME_ORDER*_numParticles );
    }

    _iGrid.resize( _numParticles );
    _phi.resize( 20*_numParticles );
    _phid.resize( 10*_numParticles );
    _phip.resize( 10*_numParticles );
    _phidp.resize( 20*_numParticles );
    _pmeAtomRange.resize( _totalGridSize + 1);
    _pmeAtomGridIndex.resize( _numParticles );

    return;
}

void ReferencePmeMBPolElectrostatics::initializePmeGrid( void )
{
    if( _pmeGrid == NULL )return;

    for (int jj = 0; jj < _totalGridSize; jj++){
        _pmeGrid[jj].re = _pmeGrid[jj].im = 0.0;
    }
    return;
}

void ReferencePmeMBPolElectrostatics::getPeriodicDelta( Vec3& deltaR ) const
{
    deltaR[0]  -= FLOOR(deltaR[0]*_invPeriodicBoxSize[0]+0.5)*_periodicBoxSize[0];
    deltaR[1]  -= FLOOR(deltaR[1]*_invPeriodicBoxSize[1]+0.5)*_periodicBoxSize[1];
    deltaR[2]  -= FLOOR(deltaR[2]*_invPeriodicBoxSize[2]+0.5)*_periodicBoxSize[2];
}

void ReferencePmeMBPolElectrostatics::getPmeScale( Vec3& scale ) const
{
    scale[0] = static_cast<double>(_pmeGridDimensions[0])*_invPeriodicBoxSize[0];
    scale[1] = static_cast<double>(_pmeGridDimensions[1])*_invPeriodicBoxSize[1];
    scale[2] = static_cast<double>(_pmeGridDimensions[2])*_invPeriodicBoxSize[2];
}

void ReferencePmeMBPolElectrostatics::initializeBSplineModuli( void )
{

    int maxSize = -1;
    for( unsigned int ii = 0; ii < 3; ii++ ){
       _pmeBsplineModuli[ii].resize( _pmeGridDimensions[ii] );
        maxSize = maxSize  > _pmeGridDimensions[ii] ? maxSize : _pmeGridDimensions[ii];
    }

    double array[MBPOL_PME_ORDER];
    double x = 0.0;
    array[0]     = 1.0 - x;
    array[1]     = x;
    for( int k = 2; k < MBPOL_PME_ORDER; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++){
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }

    vector<double> bsarray(maxSize+1, 0.0);
    for( int i = 2; i <= MBPOL_PME_ORDER+1; i++){
        bsarray[i] = array[i-2];
    }
    for( int dim = 0; dim < 3; dim++) {

        int size = _pmeGridDimensions[dim];

        // get the modulus of the discrete Fourier transform

        double factor = 2.0*M_PI/size;
        for (int i = 0; i < size; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 1; j <= size; j++) {
                double arg = factor*i*(j-1);
                sum1          += bsarray[j]*COS(arg);
                sum2          += bsarray[j]*SIN(arg);
            }
            _pmeBsplineModuli[dim][i] = (sum1*sum1 + sum2*sum2);
        }

        // fix for exponential Euler spline interpolation failure

        double eps = 1.0e-7;
        if (_pmeBsplineModuli[dim][0] < eps){
            _pmeBsplineModuli[dim][0] = 0.5*_pmeBsplineModuli[dim][1];
        }
        for (int i = 1; i < size-1; i++){
            if (_pmeBsplineModuli[dim][i] < eps){
                _pmeBsplineModuli[dim][i] = 0.5*(_pmeBsplineModuli[dim][i-1]+_pmeBsplineModuli[dim][i+1]);
            }
        }
        if (_pmeBsplineModuli[dim][size-1] < eps){
            _pmeBsplineModuli[dim][size-1] = 0.5*_pmeBsplineModuli[dim][size-2];
        }

        // compute and apply the optimal zeta coefficient

        int jcut = 50;
        for (int i = 1; i <= size; i++) {
            int k = i - 1;
            if (i > size/2)
                k = k - size;
            double zeta;
            if (k == 0){
                zeta = 1.0;
            } else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                factor          = M_PI*k/size;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor/(factor+M_PI*j);
                    sum1           = sum1 + POW(arg,   MBPOL_PME_ORDER);
                    sum2           = sum2 + POW(arg, 2*MBPOL_PME_ORDER);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg  = factor/(factor-M_PI*j);
                    sum1           += POW(arg,   MBPOL_PME_ORDER);
                    sum2           += POW(arg, 2*MBPOL_PME_ORDER);
                }
                zeta = sum2/sum1;
            }
            _pmeBsplineModuli[dim][i-1] = _pmeBsplineModuli[dim][i-1]*(zeta*zeta);
        }
    }

    return;
}

void ReferencePmeMBPolElectrostatics::calculateFixedElectrostaticsFieldPairIxn( const ElectrostaticsParticleData& particleI,
                                                                            const ElectrostaticsParticleData& particleJ,
                                                                            double dscale, double pscale )
{

    // compute the real space portion of the Ewald summation

    if( particleI.particleIndex == particleJ.particleIndex )return;

    bool isSameWater = (particleI.multipoleAtomZs == particleJ.particleIndex) or
                       (particleI.multipoleAtomYs == particleJ.particleIndex) or
                       (particleI.multipoleAtomXs == particleJ.particleIndex);

    Vec3 deltaR    = particleJ.position - particleI.position;
    getPeriodicDelta( deltaR );
    double r2     = deltaR.dot( deltaR );

    if( r2 > _cutoffDistanceSquared )return;

    double r           = SQRT(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = EXP(-(ralpha*ralpha));
    alsq2n                *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    Vec3 fim            = - deltaR * bn1 * particleJ.charge;
    Vec3 fjm            = + deltaR * bn1 * particleI.charge;

    // field of a charge: charge-dipole damping; subtract the (1 - s3) part from the erfc-screened field
    double s3 = getAndScaleInverseRs( particleI, particleJ, r, true, 3,TCD);

    // a same-water pair has no permanent field at all: subtract the full real-space term
    if( isSameWater ){
        s3 = 0.;
    }
    double rr3 = (1. - s3)/(r2*r);

    Vec3 fid            = - deltaR * rr3 * particleJ.charge;
    Vec3 fjd            = + deltaR * rr3 * particleI.charge;

    Vec3 fip            = - deltaR * rr3 * particleJ.charge;
    Vec3 fjp            = + deltaR * rr3 * particleI.charge;

    unsigned int iIndex    = particleI.particleIndex;
    unsigned int jIndex    = particleJ.particleIndex;

    _fixedElectrostaticsField[iIndex]      += fim - fid;
    _fixedElectrostaticsField[jIndex]      += fjm - fjd;

    _fixedElectrostaticsFieldPolar[iIndex] += fim - fip;
    _fixedElectrostaticsFieldPolar[jIndex] += fjm - fjp;

    return;
}

void ReferencePmeMBPolElectrostatics::calculateFixedElectrostaticsField( const vector<ElectrostaticsParticleData>& particleData )
{

    // reciprocal space fixed field

    resizePmeArrays();
    computeMBPolBsplines( particleData );
    sort( _pmeAtomGridIndex.begin(), _pmeAtomGridIndex.end(), compareInt2 );
    findMBPolAtomRangeForGrid( particleData );
    initializePmeGrid();
    spreadFixedElectrostaticssOntoGrid( particleData );
    fftpack_exec_3d( _fftplan, FFTPACK_FORWARD, _pmeGrid, _pmeGrid);
    performMBPolReciprocalConvolution();
    fftpack_exec_3d( _fftplan, FFTPACK_BACKWARD, _pmeGrid, _pmeGrid);
    computeFixedPotentialFromGrid();
    recordFixedElectrostaticsField();

    // polar field starts as a copy (point charges have no self-field; term is unused)

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for( unsigned int jj = 0; jj < _numParticles; jj++ ){
        _fixedElectrostaticsFieldPolar[jj]  = _fixedElectrostaticsField[jj];
    }

    // direct space fixed field

    this->ReferenceMBPolElectrostatics::calculateFixedElectrostaticsField( particleData );

    return;
}

#define ARRAY(x,y) array[(x)-1+((y)-1)*MBPOL_PME_ORDER]

void ReferencePmeMBPolElectrostatics::computeBSplinePoint( std::vector<double4>& thetai, double w  )
{

    double array[MBPOL_PME_ORDER*MBPOL_PME_ORDER];

    // initialization to get to 2nd order recursion

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1.0 - w;

    // perform one pass to get to 3rd order recursion

    ARRAY(3,3) = 0.5 * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5 * ((1.0+w)*ARRAY(2,1)+(2.0-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5 * (1.0-w) * ARRAY(2,1);

    // compute standard B-spline recursion to desired order

    for( int i = 4; i <= MBPOL_PME_ORDER; i++){
        int k = i - 1;
        double denom = 1.0 / k;
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++){
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        }
        ARRAY(i,1) = denom * (1.0-w) * ARRAY(k,1);
    }

    // get coefficients for the B-spline first derivative

    int k = MBPOL_PME_ORDER - 1;
    ARRAY(k,MBPOL_PME_ORDER) = ARRAY(k,MBPOL_PME_ORDER-1);
    for (int i = MBPOL_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline second derivative

    k = MBPOL_PME_ORDER - 2;
    ARRAY(k,MBPOL_PME_ORDER-1) = ARRAY(k,MBPOL_PME_ORDER-2);
    for (int i = MBPOL_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MBPOL_PME_ORDER) = ARRAY(k,MBPOL_PME_ORDER-1);
    for (int i = MBPOL_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline third derivative

    k = MBPOL_PME_ORDER - 3;
    ARRAY(k,MBPOL_PME_ORDER-2) = ARRAY(k,MBPOL_PME_ORDER-3);
    for (int i = MBPOL_PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MBPOL_PME_ORDER-1) = ARRAY(k,MBPOL_PME_ORDER-2);
    for (int i = MBPOL_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MBPOL_PME_ORDER) = ARRAY(k,MBPOL_PME_ORDER-1);
    for (int i = MBPOL_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    for (int i = 1; i <= MBPOL_PME_ORDER; i++){
        thetai[i-1] = double4(ARRAY(MBPOL_PME_ORDER,i), ARRAY(MBPOL_PME_ORDER-1,i), ARRAY(MBPOL_PME_ORDER-2,i), ARRAY(MBPOL_PME_ORDER-3,i));
    }

    return;
}

void ReferencePmeMBPolElectrostatics::computeMBPolBsplines( const std::vector<ElectrostaticsParticleData>& particleData )
{

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){
        Vec3 position  = particleData[ii].position;
        getPeriodicDelta( position );
        IntVec igrid;
        for( unsigned int jj = 0; jj < 3; jj++ ){

            double w  = position[jj]*_invPeriodicBoxSize[jj];
            double fr = _pmeGridDimensions[jj]*(w-(int)(w+0.5)+0.5);
            int ifr       = static_cast<int>(fr);
            w             = fr - ifr;
            igrid[jj]     = ifr - MBPOL_PME_ORDER + 1;
            igrid[jj]    += igrid[jj] < 0 ? _pmeGridDimensions[jj] : 0;
            std::vector<double4> thetaiTemp(MBPOL_PME_ORDER);
            computeBSplinePoint( thetaiTemp, w);
            for( unsigned int kk = 0; kk < MBPOL_PME_ORDER; kk++ ){
                _thetai[jj][ii*MBPOL_PME_ORDER+kk] = thetaiTemp[kk];
            }
        }

        _iGrid[ii]               = igrid;
        _pmeAtomGridIndex[ii]    = int2( ii, igrid[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2] + igrid[1]*_pmeGridDimensions[2] + igrid[2] );

    }

    return;
}

void ReferencePmeMBPolElectrostatics::findMBPolAtomRangeForGrid( const vector<ElectrostaticsParticleData>& particleData )
{

    int last;
    int start = 0;
    last = (start == 0 ? -1 : _pmeAtomGridIndex[start-1][1]);
    for( unsigned int ii = start; ii < _numParticles; ++ii) {
        int2 atomData = _pmeAtomGridIndex[ii];
        int gridIndex = atomData[1];
        if (gridIndex != last)
        {
            for (int jj = last+1; jj <= gridIndex; ++jj){
                _pmeAtomRange[jj] = ii;
            }
            last = gridIndex;
        }
    }

    for (int j = last+1; j <= _totalGridSize; ++j){
        _pmeAtomRange[j] = _numParticles;
    }

    // the grid index is not needed again: reuse that component to hold the z index for spreading

    for( unsigned int ii = 0; ii < _numParticles; ii++) {

        double posz           = particleData[_pmeAtomGridIndex[ii][0]].position[2];
        posz                     -= FLOOR(posz*_invPeriodicBoxSize[2])*_periodicBoxSize[2];
        double w              = posz*_invPeriodicBoxSize[2];
        double fr             = _pmeGridDimensions[2]*(w-(int)(w+0.5)+0.5);
        int z                     = (static_cast<int>(fr)) - MBPOL_PME_ORDER + 1;
        _pmeAtomGridIndex[ii][1]  = z;
    }

    return;
}

void ReferencePmeMBPolElectrostatics::getGridPointGivenGridIndex( int gridIndex, IntVec& gridPoint ) const
{

    gridPoint[0]  = gridIndex/(_pmeGridDimensions[1]*_pmeGridDimensions[2]);
    int remainder = gridIndex-gridPoint[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2];
    gridPoint[1]  = remainder/_pmeGridDimensions[2];
    gridPoint[2]  = remainder-gridPoint[1]*_pmeGridDimensions[2];

    return;
}

double ReferencePmeMBPolElectrostatics::computeFixedElectrostaticssGridValue( const vector<ElectrostaticsParticleData>& particleData,
                                                                              const int2& particleGridIndices, const Vec3& scale,
                                                                              int ix, int iy, const IntVec& gridPoint ) const
{

    double gridValue = 0.0;
    for (int i = _pmeAtomRange[particleGridIndices[0]]; i < _pmeAtomRange[particleGridIndices[1]+1]; ++i) {
        int2 atomData = _pmeAtomGridIndex[i];
        int atomIndex = atomData[0];
        int z = atomData[1];
        int iz = gridPoint[2]-z+(gridPoint[2] >= z ? 0 : _pmeGridDimensions[2]);
        if( iz >= _pmeGridDimensions[2] ){
            iz -= _pmeGridDimensions[2];
        }

        double atomCharge       = particleData[atomIndex].charge;

        double4 t = _thetai[0][atomIndex*MBPOL_PME_ORDER+ix];
        double4 u = _thetai[1][atomIndex*MBPOL_PME_ORDER+iy];
        double4 v = _thetai[2][atomIndex*MBPOL_PME_ORDER+iz];
        double term0 = atomCharge*u[0]*v[0];
        gridValue += term0*t[0];
    }
    return gridValue;
}

void ReferencePmeMBPolElectrostatics::spreadFixedElectrostaticssOntoGrid( const vector<ElectrostaticsParticleData>& particleData )
{

    Vec3 scale;
    getPmeScale( scale );

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++ ){

        IntVec gridPoint;
        getGridPointGivenGridIndex( gridIndex, gridPoint );

        double result = 0.0;
        for (int ix = 0; ix < MBPOL_PME_ORDER; ++ix)
        {
            int x = gridPoint[0]-ix+(gridPoint[0] >= ix ? 0 : _pmeGridDimensions[0]);
            for (int iy = 0; iy < MBPOL_PME_ORDER; ++iy)
            {
                int y  = gridPoint[1]-iy+(gridPoint[1] >= iy ? 0 : _pmeGridDimensions[1]);
                int z1 = gridPoint[2]-MBPOL_PME_ORDER+1;
                z1    += (z1 >= 0 ? 0 : _pmeGridDimensions[2]);
                int z2 = (z1 < gridPoint[2] ? gridPoint[2] : _pmeGridDimensions[2]-1);

                int2 particleGridIndices;
                particleGridIndices[0]  = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z1;
                particleGridIndices[1]  = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z2;
                result                 += computeFixedElectrostaticssGridValue( particleData, particleGridIndices, scale, ix, iy, gridPoint );

                if (z1 > gridPoint[2]){

                    particleGridIndices[0]  = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2];
                    particleGridIndices[1]  = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+gridPoint[2];
                    result                 += computeFixedElectrostaticssGridValue( particleData, particleGridIndices, scale, ix, iy, gridPoint );
                }
            }
        }
        _pmeGrid[gridIndex].re = result;
    }
    return;
}

void ReferencePmeMBPolElectrostatics::performMBPolReciprocalConvolution( void )
{

    double expFactor   = (M_PI*M_PI)/(_alphaEwald*_alphaEwald);
    double scaleFactor = 1.0/(M_PI*_periodicBoxSize[0]*_periodicBoxSize[1]*_periodicBoxSize[2]);

    for (int index = 0; index < _totalGridSize; index++)
    {
        int kx = index/(_pmeGridDimensions[1]*_pmeGridDimensions[2]);
        int remainder = index-kx*_pmeGridDimensions[1]*_pmeGridDimensions[2];
        int ky = remainder/_pmeGridDimensions[2];
        int kz = remainder-ky*_pmeGridDimensions[2];

        if (kx == 0 && ky == 0 && kz == 0){
            _pmeGrid[index].re = _pmeGrid[index].im = 0.0;
            continue;
        }

        int mx = (kx < (_pmeGridDimensions[0]+1)/2) ? kx : (kx-_pmeGridDimensions[0]);
        int my = (ky < (_pmeGridDimensions[1]+1)/2) ? ky : (ky-_pmeGridDimensions[1]);
        int mz = (kz < (_pmeGridDimensions[2]+1)/2) ? kz : (kz-_pmeGridDimensions[2]);

        double mhx = mx*_invPeriodicBoxSize[0];
        double mhy = my*_invPeriodicBoxSize[1];
        double mhz = mz*_invPeriodicBoxSize[2];

        double bx = _pmeBsplineModuli[0][kx];
        double by = _pmeBsplineModuli[1][ky];
        double bz = _pmeBsplineModuli[2][kz];

        double m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        double denom = m2*bx*by*bz;
        double eterm = scaleFactor*EXP(-expFactor*m2)/denom;

        _pmeGrid[index].re *= eterm;
        _pmeGrid[index].im *= eterm;
    }
}

void ReferencePmeMBPolElectrostatics::computeFixedPotentialFromGrid( void )
{
    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        for (int iz = 0; iz < MBPOL_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*MBPOL_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            for (int iy = 0; iy < MBPOL_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*MBPOL_PME_ORDER+iy];
                double4 t = double4(0.0, 0.0, 0.0, 0.0);
                for (int ix = 0; ix < MBPOL_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].re;
                    double4 tadd = _thetai[0][m*MBPOL_PME_ORDER+ix];
                    t[0] += tq*tadd[0];
                    t[1] += tq*tadd[1];
                    t[2] += tq*tadd[2];
                    t[3] += tq*tadd[3];
                }
                tu00 += t[0]*u[0];
                tu10 += t[1]*u[0];
                tu01 += t[0]*u[1];
                tu20 += t[2]*u[0];
                tu11 += t[1]*u[1];
                tu02 += t[0]*u[2];
                tu30 += t[3]*u[0];
                tu21 += t[2]*u[1];
                tu12 += t[1]*u[2];
                tu03 += t[0]*u[3];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
        }
        _phi[20*m] = tuv000;
        _phi[20*m+1] = tuv100;
        _phi[20*m+2] = tuv010;
        _phi[20*m+3] = tuv001;
        _phi[20*m+4] = tuv200;
        _phi[20*m+5] = tuv020;
        _phi[20*m+6] = tuv002;
        _phi[20*m+7] = tuv110;
        _phi[20*m+8] = tuv101;
        _phi[20*m+9] = tuv011;
        _phi[20*m+10] = tuv300;
        _phi[20*m+11] = tuv030;
        _phi[20*m+12] = tuv003;
        _phi[20*m+13] = tuv210;
        _phi[20*m+14] = tuv201;
        _phi[20*m+15] = tuv120;
        _phi[20*m+16] = tuv021;
        _phi[20*m+17] = tuv102;
        _phi[20*m+18] = tuv012;
        _phi[20*m+19] = tuv111;
    }
}

t_complex ReferencePmeMBPolElectrostatics::computeInducedDipoleGridValue( const int2& particleGridIndices, const Vec3& scale, int ix, int iy,
                                                                           const IntVec& gridPoint,
                                                                           const std::vector<Vec3>& inputInducedDipole,
                                                                           const std::vector<Vec3>& inputInducedDipolePolar ) const
{

    t_complex gridValue;
    gridValue.re = gridValue.im = 0.0;

    for (int i = _pmeAtomRange[particleGridIndices[0]]; i < _pmeAtomRange[particleGridIndices[1]+1]; ++i){
        int2 atomData = _pmeAtomGridIndex[i];
        int atomIndex = atomData[0];
        int z = atomData[1];
        int iz = gridPoint[2]-z+(gridPoint[2] >= z ? 0 : _pmeGridDimensions[2]);
        if( iz >= _pmeGridDimensions[2] ){
            iz -= _pmeGridDimensions[2];
        }
        Vec3 inducedDipole       = Vec3( scale[0]*inputInducedDipole[atomIndex][0],
                                               scale[1]*inputInducedDipole[atomIndex][1],
                                               scale[2]*inputInducedDipole[atomIndex][2] );

        Vec3 inducedDipolePolar  = Vec3( scale[0]*inputInducedDipolePolar[atomIndex][0],
                                               scale[1]*inputInducedDipolePolar[atomIndex][1],
                                               scale[2]*inputInducedDipolePolar[atomIndex][2] );

        double4 t = _thetai[0][atomIndex*MBPOL_PME_ORDER+ix];
        double4 u = _thetai[1][atomIndex*MBPOL_PME_ORDER+iy];
        double4 v = _thetai[2][atomIndex*MBPOL_PME_ORDER+iz];

        double term01 = inducedDipole[1]*u[1]*v[0] + inducedDipole[2]*u[0]*v[1];
        double term11 = inducedDipole[0]*u[0]*v[0];
        double term02 = inducedDipolePolar[1]*u[1]*v[0] + inducedDipolePolar[2]*u[0]*v[1];
        double term12 = inducedDipolePolar[0]*u[0]*v[0];

        gridValue.re += term01*t[0] + term11*t[1];
        gridValue.im += term02*t[0] + term12*t[1];

    }
    return gridValue;
}

void ReferencePmeMBPolElectrostatics::spreadInducedDipolesOnGrid( const std::vector<Vec3>& inputInducedDipole,
                                                                   const std::vector<Vec3>& inputInducedDipolePolar )
{
    Vec3 scale;
    getPmeScale( scale );

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++ )
    {
        IntVec gridPoint;
        getGridPointGivenGridIndex( gridIndex, gridPoint );

        t_complex gridValue;
        gridValue.re  = gridValue.im = 0.0;

        for (int ix = 0; ix < MBPOL_PME_ORDER; ++ix)
        {
            int x = gridPoint[0]-ix+(gridPoint[0] >= ix ? 0 : _pmeGridDimensions[0]);
            for (int iy = 0; iy < MBPOL_PME_ORDER; ++iy)
            {
                int y   = gridPoint[1]-iy+(gridPoint[1] >= iy ? 0 : _pmeGridDimensions[1]);
                int z1  = gridPoint[2]-MBPOL_PME_ORDER+1;
                z1     += (z1 >= 0 ? 0 : _pmeGridDimensions[2]);
                int z2  = (z1 < gridPoint[2] ? gridPoint[2] : _pmeGridDimensions[2]-1);

                int2 particleGridIndices;
                particleGridIndices[0] = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z1;
                particleGridIndices[1] = x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z2;
                gridValue             += computeInducedDipoleGridValue( particleGridIndices, scale, ix, iy, gridPoint, inputInducedDipole, inputInducedDipolePolar );

                if (z1 > gridPoint[2])
                {
                    particleGridIndices[0]  =  x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2];
                    particleGridIndices[1]  =  x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+gridPoint[2];
                    gridValue              +=  computeInducedDipoleGridValue( particleGridIndices, scale, ix, iy, gridPoint, inputInducedDipole, inputInducedDipolePolar );
                }
            }
        }
        _pmeGrid[gridIndex] = gridValue;
    }

    return;
}

void ReferencePmeMBPolElectrostatics::computeInducedPotentialFromGrid( void )
{
    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv100_1 = 0.0;
        double tuv010_1 = 0.0;
        double tuv001_1 = 0.0;
        double tuv200_1 = 0.0;
        double tuv020_1 = 0.0;
        double tuv002_1 = 0.0;
        double tuv110_1 = 0.0;
        double tuv101_1 = 0.0;
        double tuv011_1 = 0.0;
        double tuv100_2 = 0.0;
        double tuv010_2 = 0.0;
        double tuv001_2 = 0.0;
        double tuv200_2 = 0.0;
        double tuv020_2 = 0.0;
        double tuv002_2 = 0.0;
        double tuv110_2 = 0.0;
        double tuv101_2 = 0.0;
        double tuv011_2 = 0.0;
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        for (int iz = 0; iz < MBPOL_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*MBPOL_PME_ORDER+iz];
            double tu00_1 = 0.0;
            double tu01_1 = 0.0;
            double tu10_1 = 0.0;
            double tu20_1 = 0.0;
            double tu11_1 = 0.0;
            double tu02_1 = 0.0;
            double tu00_2 = 0.0;
            double tu01_2 = 0.0;
            double tu10_2 = 0.0;
            double tu20_2 = 0.0;
            double tu11_2 = 0.0;
            double tu02_2 = 0.0;
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            for (int iy = 0; iy < MBPOL_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*MBPOL_PME_ORDER+iy];
                double t0_1 = 0.0;
                double t1_1 = 0.0;
                double t2_1 = 0.0;
                double t0_2 = 0.0;
                double t1_2 = 0.0;
                double t2_2 = 0.0;
                double t3 = 0.0;
                for (int ix = 0; ix < MBPOL_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    t_complex tq = _pmeGrid[gridIndex];
                    double4 tadd = _thetai[0][m*MBPOL_PME_ORDER+ix];
                    t0_1 += tq.re*tadd[0];
                    t1_1 += tq.re*tadd[1];
                    t2_1 += tq.re*tadd[2];
                    t0_2 += tq.im*tadd[0];
                    t1_2 += tq.im*tadd[1];
                    t2_2 += tq.im*tadd[2];
                    t3 += (tq.re+tq.im)*tadd[3];
                }
                tu00_1 += t0_1*u[0];
                tu10_1 += t1_1*u[0];
                tu01_1 += t0_1*u[1];
                tu20_1 += t2_1*u[0];
                tu11_1 += t1_1*u[1];
                tu02_1 += t0_1*u[2];
                tu00_2 += t0_2*u[0];
                tu10_2 += t1_2*u[0];
                tu01_2 += t0_2*u[1];
                tu20_2 += t2_2*u[0];
                tu11_2 += t1_2*u[1];
                tu02_2 += t0_2*u[2];
                double t0 = t0_1 + t0_2;
                double t1 = t1_1 + t1_2;
                double t2 = t2_1 + t2_2;
                tu00 += t0*u[0];
                tu10 += t1*u[0];
                tu01 += t0*u[1];
                tu20 += t2*u[0];
                tu11 += t1*u[1];
                tu02 += t0*u[2];
                tu30 += t3*u[0];
                tu21 += t2*u[1];
                tu12 += t1*u[2];
                tu03 += t0*u[3];
            }
            tuv100_1 += tu10_1*v[0];
            tuv010_1 += tu01_1*v[0];
            tuv001_1 += tu00_1*v[1];
            tuv200_1 += tu20_1*v[0];
            tuv020_1 += tu02_1*v[0];
            tuv002_1 += tu00_1*v[2];
            tuv110_1 += tu11_1*v[0];
            tuv101_1 += tu10_1*v[1];
            tuv011_1 += tu01_1*v[1];
            tuv100_2 += tu10_2*v[0];
            tuv010_2 += tu01_2*v[0];
            tuv001_2 += tu00_2*v[1];
            tuv200_2 += tu20_2*v[0];
            tuv020_2 += tu02_2*v[0];
            tuv002_2 += tu00_2*v[2];
            tuv110_2 += tu11_2*v[0];
            tuv101_2 += tu10_2*v[1];
            tuv011_2 += tu01_2*v[1];
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
        }
        _phid[10*m]   = 0.0;
        _phid[10*m+1] = tuv100_1;
        _phid[10*m+2] = tuv010_1;
        _phid[10*m+3] = tuv001_1;
        _phid[10*m+4] = tuv200_1;
        _phid[10*m+5] = tuv020_1;
        _phid[10*m+6] = tuv002_1;
        _phid[10*m+7] = tuv110_1;
        _phid[10*m+8] = tuv101_1;
        _phid[10*m+9] = tuv011_1;

        _phip[10*m]   = 0.0;
        _phip[10*m+1] = tuv100_2;
        _phip[10*m+2] = tuv010_2;
        _phip[10*m+3] = tuv001_2;
        _phip[10*m+4] = tuv200_2;
        _phip[10*m+5] = tuv020_2;
        _phip[10*m+6] = tuv002_2;
        _phip[10*m+7] = tuv110_2;
        _phip[10*m+8] = tuv101_2;
        _phip[10*m+9] = tuv011_2;

        _phidp[20*m] = tuv000;
        _phidp[20*m+1] = tuv100;
        _phidp[20*m+2] = tuv010;
        _phidp[20*m+3] = tuv001;
        _phidp[20*m+4] = tuv200;
        _phidp[20*m+5] = tuv020;
        _phidp[20*m+6] = tuv002;
        _phidp[20*m+7] = tuv110;
        _phidp[20*m+8] = tuv101;
        _phidp[20*m+9] = tuv011;
        _phidp[20*m+10] = tuv300;
        _phidp[20*m+11] = tuv030;
        _phidp[20*m+12] = tuv003;
        _phidp[20*m+13] = tuv210;
        _phidp[20*m+14] = tuv201;
        _phidp[20*m+15] = tuv120;
        _phidp[20*m+16] = tuv021;
        _phidp[20*m+17] = tuv102;
        _phidp[20*m+18] = tuv012;
        _phidp[20*m+19] = tuv111;
    }
    return;
}

double ReferencePmeMBPolElectrostatics::computeReciprocalSpaceFixedElectrostaticsForceAndEnergy( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                                                 std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential ) const
{
    double multipole[10];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
    Vec3 scale;
    getPmeScale( scale );
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++ ) {

        for (int k = 0; k < 10; k++)
            multipole[k] = 0.0;
        multipole[0] = particleData[i].charge;

        const double* phi = &_phi[20*i];

        electrostaticPotential[i] += phi[0] ;

        Vec3 f = Vec3( 0.0, 0.0, 0.0);
        for (int k = 0; k < 1; k++) {
            energy += multipole[k]*phi[k];
            f[0]   += multipole[k]*phi[deriv1[k]];
            f[1]   += multipole[k]*phi[deriv2[k]];
            f[2]   += multipole[k]*phi[deriv3[k]];
        }

        // charge-redistribution forces are added in calculateElectrostatic from the total site potential

        f[0]           *= scale[0];
        f[1]           *= scale[1];
        f[2]           *= scale[2];
        f              *= (_electric);
        forces[i]      -= f;

    }
    return (0.5*_electric*energy);
}

double ReferencePmeMBPolElectrostatics::computeReciprocalSpaceInducedDipoleForceAndEnergy( ReferenceMBPolElectrostatics::PolarizationType polarizationType,
                                                                                                const std::vector<ElectrostaticsParticleData>& particleData,
                                                                                                std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential) const
{

    double multipole[10];
    double inducedDipole[3];
    double inducedDipolePolar[3];
    double scales[3];
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18};
    Vec3 scale;
    getPmeScale( scale );
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++ ) {

        unsigned int iIndex = particleData[i].particleIndex;

        for (int k = 0; k < 10; k++)
            multipole[k] = 0.0;
        multipole[0] = particleData[i].charge;

        inducedDipole[0] = _inducedDipole[i][0];
        inducedDipole[1] = _inducedDipole[i][1];
        inducedDipole[2] = _inducedDipole[i][2];

        inducedDipolePolar[0] = _inducedDipolePolar[i][0];
        inducedDipolePolar[1] = _inducedDipolePolar[i][1];
        inducedDipolePolar[2] = _inducedDipolePolar[i][2];

        energy += scale[0]*inducedDipole[0]*_phi[20*i+1];
        energy += scale[1]*inducedDipole[1]*_phi[20*i+2];
        energy += scale[2]*inducedDipole[2]*_phi[20*i+3];

        electrostaticPotential[i] += .5* _phidp[20*i];

        Vec3 f        = Vec3(0.0, 0.0, 0.0 );

        for (int k = 0; k < 3; k++) {

            int j1 = deriv1[k+1];
            int j2 = deriv2[k+1];
            int j3 = deriv3[k+1];

            f[0] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j1]*(scale[k]/scale[0]);
            f[1] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j2]*(scale[k]/scale[1]);
            f[2] += (inducedDipole[k]+inducedDipolePolar[k])*_phi[20*i+j3]*(scale[k]/scale[2]);

            if( polarizationType == ReferenceMBPolElectrostatics::Mutual )
            {
                f[0] += (inducedDipole[k]*_phip[10*i+j1] + inducedDipolePolar[k]*_phid[10*i+j1])*(scale[k]/scale[0]);
                f[1] += (inducedDipole[k]*_phip[10*i+j2] + inducedDipolePolar[k]*_phid[10*i+j2])*(scale[k]/scale[1]);
                f[2] += (inducedDipole[k]*_phip[10*i+j3] + inducedDipolePolar[k]*_phid[10*i+j3])*(scale[k]/scale[2]);
            }

        }

        f[0] *= scale[0];
        f[1] *= scale[1];
        f[2] *= scale[2];

        for (int k = 0; k < 10; k++) {
            f[0] += multipole[k]*_phidp[20*i+deriv1[k]];
            f[1] += multipole[k]*_phidp[20*i+deriv2[k]];
            f[2] += multipole[k]*_phidp[20*i+deriv3[k]];
        }

        f[0]           *= scale[0];
        f[1]           *= scale[1];
        f[2]           *= scale[2];
        f              *= (0.5*_electric);
        forces[iIndex] -= f;

    }

    return (0.5*_electric*energy);
}

void ReferencePmeMBPolElectrostatics::recordFixedElectrostaticsField( void )
{
    Vec3 scale;
    getPmeScale( scale );
    scale *= -1.0;
    for (int i = 0; i < _numParticles; i++ ){
        _fixedElectrostaticsField[i][0] = scale[0]*_phi[20*i+1];
        _fixedElectrostaticsField[i][1] = scale[1]*_phi[20*i+2];
        _fixedElectrostaticsField[i][2] = scale[2]*_phi[20*i+3];
    }
    return;
}

void ReferencePmeMBPolElectrostatics::initializeInducedDipoles( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields )
{

    this->ReferenceMBPolElectrostatics::initializeInducedDipoles( updateInducedDipoleFields );
    calculateReciprocalSpaceInducedDipoleField( updateInducedDipoleFields );
    return;
}

void ReferencePmeMBPolElectrostatics::recordInducedDipoleField( vector<Vec3>& field, vector<Vec3>& fieldPolar )
{
    Vec3 scale;
    getPmeScale( scale );
    scale *= -1.0;
    for (int i = 0; i < _numParticles; i++ ) {

        field[i][0]      += scale[0]*_phid[10*i+1];
        field[i][1]      += scale[1]*_phid[10*i+2];
        field[i][2]      += scale[2]*_phid[10*i+3];

        fieldPolar[i][0] += scale[0]*_phip[10*i+1];
        fieldPolar[i][1] += scale[1]*_phip[10*i+2];
        fieldPolar[i][2] += scale[2]*_phip[10*i+3];
    }
    return;
}

void ReferencePmeMBPolElectrostatics::calculateReciprocalSpaceInducedDipoleField( std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields )
{
    initializePmeGrid();
    spreadInducedDipolesOnGrid( *(updateInducedDipoleFields[0].inducedDipoles), *(updateInducedDipoleFields[1].inducedDipoles) );
    fftpack_exec_3d( _fftplan, FFTPACK_FORWARD, _pmeGrid, _pmeGrid);
    performMBPolReciprocalConvolution();
    fftpack_exec_3d( _fftplan, FFTPACK_BACKWARD, _pmeGrid, _pmeGrid);
    computeInducedPotentialFromGrid();
    recordInducedDipoleField( updateInducedDipoleFields[0].inducedDipoleField, updateInducedDipoleFields[1].inducedDipoleField );
}

void ReferencePmeMBPolElectrostatics::calculateInducedDipoleFields( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                     std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    // direct space

    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        for( unsigned int jj = ii + 1; jj < particleData.size(); jj++ ){
            calculateDirectInducedDipolePairIxns( particleData[ii], particleData[jj], updateInducedDipoleFields );
        }
    }

    // reciprocal space

    calculateReciprocalSpaceInducedDipoleField( updateInducedDipoleFields );

    // self term

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for( unsigned int ii = 0; ii < updateInducedDipoleFields.size(); ii++ ){
        vector<Vec3>& inducedDipoles = *(updateInducedDipoleFields[ii].inducedDipoles);
        vector<Vec3>& field          = updateInducedDipoleFields[ii].inducedDipoleField;
        for( unsigned int jj = 0; jj < particleData.size(); jj++ ){
            field[jj] += inducedDipoles[jj]*term;
        }
    }

    return;
}

void ReferencePmeMBPolElectrostatics::calculateDirectInducedDipolePairIxn( unsigned int iIndex, unsigned int jIndex,
                                                                            double preFactor1, double preFactor2,
                                                                            const Vec3& delta,
                                                                            const std::vector<Vec3>& inducedDipole,
                                                                            std::vector<Vec3>& field ) const
{

    double dur  = inducedDipole[jIndex].dot( delta );
    field[iIndex]  += delta*(dur*preFactor2) + inducedDipole[jIndex]*preFactor1;

               dur  = inducedDipole[iIndex].dot( delta );
    field[jIndex]  += delta*(dur*preFactor2) + inducedDipole[iIndex]*preFactor1;

    return;
}

void ReferencePmeMBPolElectrostatics::calculateDirectInducedDipolePairIxns( const ElectrostaticsParticleData& particleI,
                                                                             const ElectrostaticsParticleData& particleJ,
                                                                             std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields )
{

    // compute the real space portion of the Ewald summation

    double uscale = 1.0;
    Vec3 deltaR    = particleJ.position - particleI.position;

    getPeriodicDelta( deltaR );
    double r2     = deltaR.dot( deltaR );

    if( r2 > _cutoffDistanceSquared )return;

    double r           = SQRT(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = EXP(-(ralpha*ralpha));
    alsq2n                *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    alsq2n                *= alsq2;
    double bn2         = (3.0*bn1+alsq2n*exp2a)/r2;

    // erfc-screened dipole field minus its (1 - Thole) part

    double scale3 = getAndScaleInverseRs(particleI, particleJ, r, true, 3, TDD);
    double scale5 = getAndScaleInverseRs(particleI, particleJ, r, true, 5, TDD);

    double dsc3        = uscale*scale3;
    double dsc5        = uscale*scale5;

    double r3          = (r*r2);
    double r5          = (r3*r2);
    double rr3         = (1.0-dsc3)/r3;
    double rr5         = 3.0*(1.0-dsc5)/r5;

    double preFactor1  = rr3 - bn1;
    double preFactor2  = bn2 - rr5;

    for( unsigned int ii = 0; ii < updateInducedDipoleFields.size(); ii++ ){
        calculateDirectInducedDipolePairIxn( particleI.particleIndex, particleJ.particleIndex, preFactor1, preFactor2, deltaR,
                                            *(updateInducedDipoleFields[ii].inducedDipoles),
                                              updateInducedDipoleFields[ii].inducedDipoleField );
    }

    return;
}

double ReferencePmeMBPolElectrostatics::calculatePmeSelfEnergy( const std::vector<ElectrostaticsParticleData>& particleData,
        std::vector<Vec3>& forces, std::vector<double>& electrostaticPotential ) const
{

    double cii = 0.0;
    double dii = 0.0;
    double qii = 0.0;
    for( unsigned int ii = 0; ii < _numParticles; ii++ ){

        const ElectrostaticsParticleData& particleI = particleData[ii];

        cii      +=  particleI.charge*particleI.charge;

    }

    double term      = 2.0*_alphaEwald*_alphaEwald;
    double energy    = (cii + term*(dii/3.0 + 2.0*term*qii/5.0));
               energy   *= -(_electric*_alphaEwald/(_dielectric*SQRT_PI));

    for( unsigned int ii = 0; ii < _numParticles; ii++ ){

        const ElectrostaticsParticleData& particleI = particleData[ii];

        electrostaticPotential[ii] += particleI.charge * -(_alphaEwald/(SQRT_PI)) * 2;

    }

    return energy;
}

double ReferencePmeMBPolElectrostatics::calculatePmeDirectElectrostaticPairIxn( const std::vector<ElectrostaticsParticleData>& particleData,
                                             unsigned int iIndex,
                                             unsigned int jIndex,
                                                                                         std::vector<Vec3>& forces,
                                                                                         std::vector<double>& electrostaticPotential ) const
{

    ElectrostaticsParticleData particleI = particleData[iIndex];
    ElectrostaticsParticleData particleJ = particleData[jIndex];

    double energy;
    Vec3 deltaR   = particleJ.position - particleI.position;
    getPeriodicDelta( deltaR );
    double r2    = deltaR.dot( deltaR );

    if( r2 > _cutoffDistanceSquared )return 0.0;

    double xr    = deltaR[0];
    double yr    = deltaR[1];
    double zr    = deltaR[2];

    double r      = SQRT(r2);
    double ck     = particleJ.charge;

    double ci    = particleI.charge;

    // real-space error function terms bn0..bn3

    double ralpha = _alphaEwald*r;
    double bn0    = erfc(ralpha)/r;

    double alsq2  = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n = 0.0;
    if( _alphaEwald > 0.0 ){
        alsq2n = 1.0/(SQRT_PI*_alphaEwald);
    }
    double exp2a  = EXP(-(ralpha*ralpha));

    alsq2n           *= alsq2;
    double bn1    = (bn0+alsq2n*exp2a)/r2;

    alsq2n           *= alsq2;
    double bn2    = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n           *= alsq2;
    double bn3    = (5.0*bn2+alsq2n*exp2a)/r2;

    // undamped (n-2)!!/r^n; Thole scale factors are applied below

    double rr1    = 1.0/r;
    double rr3    = rr1/r2;
    double rr5    = 3.0*rr3/r2;
    double rr7    = 5.0*rr5/r2;

    // scalar products of the induced dipoles (sci) and of the d/p sets (scip)

    double sci3  = _inducedDipole[iIndex].dot(deltaR);
    double sci4  = _inducedDipole[jIndex].dot(deltaR);
    double scip2 = _inducedDipole[iIndex].dot(_inducedDipolePolar[jIndex])
             + _inducedDipolePolar[iIndex].dot(_inducedDipole[jIndex]);

    double scip3 = _inducedDipolePolar[iIndex].dot(deltaR);
    double scip4 = _inducedDipolePolar[jIndex].dot(deltaR);

    // gl: permanent (charge-charge); gli/glip: charge-induced dipole

    double gl0           = ci*ck;

    double gli1          = ck*sci3 - ci*sci4;

    double glip1         = ck*scip3 - ci*scip4;

    bool isSameWater = (particleI.multipoleAtomZs == particleJ.particleIndex) or
                       (particleI.multipoleAtomYs == particleJ.particleIndex) or
                       (particleI.multipoleAtomXs == particleJ.particleIndex);

    // same-water pairs are not skipped in PME: their scale factors are set to 0 below, so the full
    // bare Coulomb term is subtracted (cancelling their reciprocal-space contribution)

    double e             = bn0*gl0;
    double ei            = 0.5 * (bn1*gli1);

    // subtract the unscreened Coulomb terms times (1 - Thole scale)

    double scale1CC =getAndScaleInverseRs(particleI,particleJ,r,true,1,TCC);
    double scale3CD =getAndScaleInverseRs(particleI,particleJ,r,true,3,TCD);

    if( isSameWater ) {
    scale1CC = scale3CD = 0.;
    }
    double erl  =       rr1*gl0 *(1 - scale1CC) ; // charge-charge
    double erli = 0.5*( rr3*gli1*(1 - scale3CD)); // charge - induced dipole

    e                   = e - erl;
    ei                  = ei - erli;

    energy              = (e + ei);

    electrostaticPotential[iIndex] += ck * (bn0 - rr1 * (1 - scale1CC));
    electrostaticPotential[jIndex] += ci * (bn0 - rr1 * (1 - scale1CC));

    electrostaticPotential[iIndex] -= sci4 * (bn1 - rr3 * (1 - scale3CD));
    electrostaticPotential[jIndex] += sci3 * (bn1 - rr3 * (1 - scale3CD));

    double scale3CC =getAndScaleInverseRs(particleI,particleJ,r,true,3,TCC);
    double scale5CD =getAndScaleInverseRs(particleI,particleJ,r,true,5,TCD);
    double scale5DD =getAndScaleInverseRs(particleI,particleJ,r,true,5,TDD);
    double scale7DD =getAndScaleInverseRs(particleI,particleJ,r,true,7,TDD);

    if( isSameWater ) {
        scale3CC = scale5CD = 0.;
    }

    // permanent force terms (gfr: unscreened part to subtract)

    double gf1 = bn1*gl0;

    double gfr1 = (1 - scale3CC) * rr3*gl0;

    // induced force terms

    double gfi1  = 0.5*( bn2* ( gli1
                              + glip1
                              + scip2 ) // inddip - inddip
               - bn3*(sci3*scip4+scip3*sci4));

    double gfi2 = -ck*bn1;
    double gfi3 =  ci*bn1;

    double gfri1 = 0.5*(rr5 * ( gli1  * (1 - scale5CD)   // charge - inddip
                        + glip1 * (1 - scale5CD)   // charge - inddip
                  + scip2 * (1 - scale5DD) ) // inddip - inddip
                      - rr7 * (sci3*scip4+scip3*sci4)
                              * (1 - scale7DD)   // inddip - inddip
               );

    // permanent force with screening

    Vec3 ftm2 = deltaR*gf1;

    // permanent force without screening

    Vec3 ftm2r = deltaR*gfr1;

    // induced force with screening

    Vec3 ftm2i = deltaR*gfi1;
    // charge_i * inddip_j
    ftm2i += ( (       _inducedDipole[iIndex]
              +       _inducedDipolePolar[iIndex]) * gfi2
              + (       _inducedDipole[jIndex]
              +       _inducedDipolePolar[jIndex]) * gfi3
    // inddipP_i* inddip_j
                  + ( _inducedDipolePolar[iIndex] * sci4
              + _inducedDipole[iIndex] * scip4   ) * bn2
              +  ( _inducedDipolePolar[jIndex] * sci3
              + _inducedDipole[jIndex]  * scip3   ) * bn2) * 0.5;

    // induced force without screening

    Vec3 ftm2ri = deltaR * gfri1;
    ftm2ri += ( _inducedDipolePolar[iIndex] * sci4
                      + _inducedDipole[iIndex] * scip4
                          +  _inducedDipolePolar[jIndex] * sci3
                      + _inducedDipole[jIndex] * scip3)*0.5*rr5*(1 - scale5DD);

    // charge-induced dipole; scale3CD = 0 for same-water pairs removes it fully

    ftm2ri += ( - (_inducedDipole[iIndex]
                                      +_inducedDipolePolar[iIndex])*ck
                        + (_inducedDipole[jIndex]
                              +_inducedDipolePolar[jIndex])*ci)*0.5*rr3*(1 - scale3CD);

    ftm2  -= ftm2r;
    ftm2i -= ftm2ri;

    double conversionFactor  = (_electric/_dielectric);

    energy                 *= conversionFactor;

    forces[iIndex]      -= (ftm2 + ftm2i)*conversionFactor;

    forces[jIndex]      += (ftm2 + ftm2i)*conversionFactor;

    return energy;

}

double ReferencePmeMBPolElectrostatics::calculateElectrostatic( const std::vector<ElectrostaticsParticleData>& particleData,
                                                                     std::vector<Vec3>& forces )
{

    double energy = 0.0;

    std::vector<double> electrostaticPotentialDirect(particleData.size());
    std::vector<double> electrostaticPotentialInduced(particleData.size());
    std::vector<double> electrostaticPotentialReciprocal(particleData.size());
    std::vector<double> electrostaticPotentialSelf(particleData.size());
    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        electrostaticPotentialDirect[ii] = 0.;
        electrostaticPotentialReciprocal[ii] = 0.;
        electrostaticPotentialSelf[ii] = 0.;
    }
    // direct space

    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        for( unsigned int jj = ii+1; jj < particleData.size(); jj++ ){

            energy += calculatePmeDirectElectrostaticPairIxn( particleData, ii, jj, forces, electrostaticPotentialDirect );

        }
    }

    printPotential (electrostaticPotentialDirect, energy ,"Direct Space", particleData);

    double previousEnergy = energy;

    energy += computeReciprocalSpaceInducedDipoleForceAndEnergy( getPolarizationType(), particleData, forces, electrostaticPotentialInduced );
    printPotential (electrostaticPotentialInduced, energy - previousEnergy , "Reciprocal Induced", particleData);

    previousEnergy = energy;
    energy += computeReciprocalSpaceFixedElectrostaticsForceAndEnergy( particleData, forces, electrostaticPotentialReciprocal );
    printPotential (electrostaticPotentialReciprocal, energy - previousEnergy , "Reciprocal Space Fixed", particleData);

    previousEnergy = energy;
    energy += calculatePmeSelfEnergy( particleData, forces, electrostaticPotentialSelf );

    printPotential (electrostaticPotentialSelf, energy - previousEnergy , "Pme Self energy", particleData);

    for (int i=0; i<particleData.size(); i++) {
        electrostaticPotentialDirect[i] += electrostaticPotentialReciprocal[i];
        electrostaticPotentialDirect[i] += electrostaticPotentialInduced[i];
        electrostaticPotentialDirect[i] += electrostaticPotentialSelf[i];
    }

    printPotential (electrostaticPotentialDirect, energy, "Total", particleData);
    _sitePotential = electrostaticPotentialDirect;

    // charge-redistribution forces: F_i -= sum_s dq(otherSiteIndex[s])/dr_i * phi(otherSiteIndex[s])
    for( unsigned int ii = 0; ii < particleData.size(); ii++ ){
        for( unsigned int s = 0; s < 3; s++ ){
            for( unsigned int xyz = 0; xyz < 3; xyz++ ){

            forces[ii][xyz] += particleData[ii].chargeDerivatives[s][xyz] * electrostaticPotentialDirect[particleData[ii].otherSiteIndex[s]] * -(_electric/(_dielectric));

        }}    }

    return energy;
}

void ReferenceMBPolElectrostatics::printPotential (std::vector<double> electrostaticPotential, double energy, std::string name, const std::vector<ElectrostaticsParticleData>& particleData ) {
}

void ReferenceMBPolElectrostatics::computeWaterCharge(
        ElectrostaticsParticleData& particleO, ElectrostaticsParticleData& particleH1,
        ElectrostaticsParticleData& particleH2,ElectrostaticsParticleData& particleM)
{
    const double Bohr_A = 0.52917721092; // CODATA 2010
    // M-site positioning (TTM2.1-F)
    const double gammaM = 0.426706882;

    const double gamma1 = 1.0 - gammaM;
    const double gamma2 = gammaM/2;
    const double ath0 = 1.82400520401572996557;
    const double costhe = -0.24780227221366464506;
    const double reoh = 0.958649;
    const double b1D = 1.0;
    const double a = 0.2999e0;
    const double b = -0.6932e0;
    const double c0 = 1.0099e0;
    const double c1 = -0.1801e0;
    const double c2 = 0.0892e0;

    const double e =  1.602176565e-19; // C CODATA 2010

    // E_cc, CHARGECON: unused (the formula expects c0 = speed of light, not the coefficient above)
    const double E_cc = 1.0e-7*(c0*e*c0*e)/1.0e-10;
    const double Na = 6.02214129e+23; // CODATA 2010
    const double kcal_J = 4184.0;

    const double CHARGECON = sqrt(E_cc*Na/kcal_J);

    const size_t idxD0[84] = {
           1, 1, 1, 2, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4,
           1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 1, 1, 1, 1, 1,
           1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 1, 1, 1, 1,
           1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
           5, 6, 6, 7
    };

    const size_t idxD1[84] = {
           1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1,
           1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4, 5,
           6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 3, 4,
           5, 6, 7, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2,
           3, 1, 2, 1
    };

    const size_t idxD2[84] = {
           1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1,
           5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1, 6, 5, 4, 3, 2,
           1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1, 2, 1, 1, 7, 6, 5, 4,
           3, 2, 1, 6, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 4, 3, 2, 1, 3, 2,
           1, 2, 1, 1
    };

    const double coefD[84] = {
          -2.1689686086730e-03, 1.4910379754728e-02, 5.3546078430060e-02,
          -7.4055995388666e-02,-3.7764333017616e-03, 1.4089887256484e-01,
          -6.2584207687264e-02,-1.1260393113022e-01,-5.7824159269319e-02,
           1.4360743650655e-02,-1.5469680141070e-02,-1.3036350092795e-02,
           2.7515837781556e-02, 1.4098478875076e-01,-2.7663168397781e-02,
          -5.2378176254797e-03,-1.0237198381792e-02, 8.9571999265473e-02,
           7.2920263098603e-03,-2.6873260551686e-01, 2.0220870325864e-02,
          -7.0764766270927e-02, 1.2140640273760e-01, 2.0978491966341e-02,
          -1.9443840512668e-01, 4.0826835370618e-02,-4.5365190474650e-02,
           6.2779900072132e-02,-1.3194351021000e-01,-1.4673032718563e-01,
           1.1894031277247e-01,-6.4952851564679e-03, 8.8503610374493e-02,
           1.4899437409291e-01, 1.3962841511565e-01,-2.6459446720450e-02,
          -5.0128914532773e-02, 1.8329676428116e-01,-1.5559089125095e-01,
          -4.0176879767592e-02, 3.6192059996636e-01, 1.0202887240343e-01,
           1.9318668580051e-01,-4.3435977107932e-01,-4.2080828803311e-02,
           1.9144626027273e-01,-1.7851138969948e-01, 1.0524533875070e-01,
          -1.7954071602185e-02, 5.2022455612120e-02,-2.8891891146828e-01,
          -4.7452036576319e-02,-1.0939400546289e-01, 3.5916564473568e-01,
          -2.0162789820172e-01,-3.5838629543696e-01, 5.6706523551202e-03,
           1.3849337488211e-01,-4.1733982195604e-01, 4.1641570764241e-01,
          -1.2243429796296e-01, 4.7141730971228e-02,-1.8224510249551e-01,
          -1.8880981556620e-01,-3.1992359561800e-01,-1.8567550546587e-01,
           6.1850530431280e-01,-6.1142756235141e-02,-1.6996135584933e-01,
           5.4252879499871e-01, 6.6128603899427e-01, 1.2107016404639e-02,
          -1.9633639729189e-01, 2.7652059420824e-03,-2.2684111109778e-01,
          -4.7924491598635e-01, 2.4287790137314e-01,-1.4296023329441e-01,
           8.9664665907006e-02,-1.4003228575602e-01,-1.3321543452254e-01,
          -1.8340983193745e-01, 2.3426707273520e-01, 1.5141050914514e-01
    };

    double ROH1[3], ROH2[3], RHH[3], dROH1(0), dROH2(0), dRHH(0);

    for (size_t i = 0; i < 3; ++i) {
        ROH1[i] = particleH1.position[i]*10. - particleO.position[i]*10.; // H1 - O
        ROH2[i] = particleH2.position[i]*10. - particleO.position[i]*10.; // H2 - O
        RHH[i] = particleH1.position[i]*10. - particleH2.position[i]*10.; // H1 - H2

        dROH1 += ROH1[i]*ROH1[i];
        dROH2 += ROH2[i]*ROH2[i];
        dRHH += RHH[i]*RHH[i];
    }

    dROH1 = std::sqrt(dROH1);
    dROH2 = std::sqrt(dROH2);
    dRHH = std::sqrt(dRHH);

    const double costh =
        (ROH1[0]*ROH2[0] + ROH1[1]*ROH2[1] + ROH1[2]*ROH2[2])/(dROH1*dROH2);

    const double efac = exp(-b1D*(std::pow((dROH1 - reoh), 2)
                                     + std::pow((dROH2 - reoh), 2)));

    const double x1 = (dROH1 - reoh)/reoh;
    const double x2 = (dROH2 - reoh)/reoh;
    const double x3 = costh - costhe;

    double fmat[3][16];

    for (size_t i = 0; i < 3; ++i) {
        fmat[i][0] = 0.0;
        fmat[i][1] = 1.0;
    }

    for (size_t j = 2; j < 16; ++j) {
        fmat[0][j] = fmat[0][j - 1]*x1;
        fmat[1][j] = fmat[1][j - 1]*x2;
        fmat[2][j] = fmat[2][j - 1]*x3;
    }

    // H charges p1, p2 and their derivatives from the dipole moment surface fit

    double p1(0), p2(0);
    double pl1 = costh;
    double pl2 = 0.5*(3*pl1*pl1 - 1.0);

    double dp1dr1(0);
    double dp1dr2(0);
    double dp1dcabc(0);
    double dp2dr1(0);
    double dp2dr2(0);
    double dp2dcabc(0);

    for (size_t j = 1; j < 84; ++j) {
        const size_t inI = idxD0[j];
        const size_t inJ = idxD1[j];
        const size_t inK = idxD2[j];

        p1 += coefD[j]*fmat[0][inI]*fmat[1][inJ]*fmat[2][inK];
        p2 += coefD[j]*fmat[0][inJ]*fmat[1][inI]*fmat[2][inK];

        dp1dr1 +=
            coefD[j]*(inI - 1)*fmat[0][inI - 1]*fmat[1][inJ]*fmat[2][inK];
        dp1dr2 +=
            coefD[j]*(inJ - 1)*fmat[0][inI]*fmat[1][inJ - 1]*fmat[2][inK];
        dp1dcabc +=
            coefD[j]*(inK - 1)*fmat[0][inI]*fmat[1][inJ]*fmat[2][inK - 1];
        dp2dr1 +=
            coefD[j]*(inJ - 1)*fmat[0][inJ - 1]*fmat[1][inI]*fmat[2][inK];
        dp2dr2 +=
            coefD[j]*(inI - 1)*fmat[0][inJ]*fmat[1][inI - 1]*fmat[2][inK];
        dp2dcabc +=
            coefD[j]*(inK - 1)*fmat[0][inJ]*fmat[1][inI]*fmat[2][inK - 1];
    }

    const double xx = Bohr_A;
    const double xx2 = xx*xx;

    dp1dr1 /= reoh/xx;
    dp1dr2 /= reoh/xx;
    dp2dr1 /= reoh/xx;
    dp2dr2 /= reoh/xx;

    const double pc0 =
        a*(std::pow(dROH1, b) + std::pow(dROH2, b))*(c0 + pl1*c1 + pl2*c2);

    const double dpc0dr1 =
        a*b*std::pow(dROH1, b - 1)*(c0 + pl1*c1 + pl2*c2)*xx2;
    const double dpc0dr2 =
        a*b*std::pow(dROH2, b - 1)*(c0 + pl1*c1 + pl2*c2)*xx2;
    double dpc0dcabc =
        a*(std::pow(dROH1, b) + std::pow(dROH2, b))*(c1 + 0.5*(6.0*pl1)*c2)*xx;

    const double defacdr1 = -2.0*b1D*(dROH1 - reoh)*efac*xx;
    const double defacdr2 = -2.0*b1D*(dROH2 - reoh)*efac*xx;

    dp1dr1 = dp1dr1*efac + p1*defacdr1 + dpc0dr1;
    dp1dr2 = dp1dr2*efac + p1*defacdr2 + dpc0dr2;
    dp1dcabc = dp1dcabc*efac + dpc0dcabc;
    dp2dr1 = dp2dr1*efac + p2*defacdr1 + dpc0dr1;
    dp2dr2 = dp2dr2*efac + p2*defacdr2 + dpc0dr2;
    dp2dcabc = dp2dcabc*efac + dpc0dcabc;

    p1 = coefD[0] + p1*efac + pc0*xx; // q^H1 in TTM2-F
    p2 = coefD[0] + p2*efac + pc0*xx; // q^H2

    double chargeO= -(p1 + p2);
    double chargeH1 = p1;
    double chargeH2 = p2;
    double gamma2div1 = gamma2/gamma1;

    particleO.charge = 0.;
    particleH1.charge = chargeH1 + gamma2div1*(chargeH1 + chargeH2);
    particleH2.charge = chargeH2 + gamma2div1*(chargeH1 + chargeH2);
    particleM.charge = chargeO/gamma1;

    dp1dr1 /= xx;
    dp1dr2 /= xx;
    dp2dr1 /= xx;
    dp2dr2 /= xx;

    const double f1q1r13 = (dp1dr1 - (dp1dcabc*costh/dROH1))/dROH1;
    const double f1q1r23 = dp1dcabc/(dROH1*dROH2);
    const double f2q1r23 = (dp1dr2 - (dp1dcabc*costh/dROH2))/dROH2;
    const double f2q1r13 = dp1dcabc/(dROH2*dROH1);
    const double f1q2r13 = (dp2dr1 - (dp2dcabc*costh/dROH1))/dROH1;
    const double f1q2r23 = dp2dcabc/(dROH1*dROH2);
    const double f2q2r23 = (dp2dr2 - (dp2dcabc*costh/dROH2))/dROH2;
    const double f2q2r13 = dp2dcabc/(dROH2*dROH1);

    // chargeDerivativesX[vsY] = d q_X / d r_Y (charges before M-site redistribution, e/Angstrom)

    enum ChargeDerivativesIndices { vsH1, vsH2, vsO };

    std::vector<Vec3> chargeDerivativesH1;
    chargeDerivativesH1.resize(3);

    for (size_t i = 0; i < 3; ++i) {
        chargeDerivativesH1[vsH1][i] = f1q1r13*ROH1[i] + f1q1r23*ROH2[i];
        chargeDerivativesH1[vsH2][i] = f2q1r13*ROH1[i] + f2q1r23*ROH2[i];
        chargeDerivativesH1[vsO][i] = -(chargeDerivativesH1[vsH1][i]+chargeDerivativesH1[vsH2][i]);
    }

    std::vector<Vec3> chargeDerivativesH2;
    chargeDerivativesH2.resize(3);

    for (size_t i = 0; i < 3; ++i) {
            chargeDerivativesH2[vsH1][i] = f1q2r13*ROH1[i] + f1q2r23*ROH2[i];
            chargeDerivativesH2[vsH2][i] = f2q2r13*ROH1[i] + f2q2r23*ROH2[i];
            chargeDerivativesH2[vsO][i] = -(chargeDerivativesH2[vsH1][i]+chargeDerivativesH2[vsH2][i]);
    }

    std::vector<Vec3> chargeDerivativesO;
    chargeDerivativesO.resize(3);

    for (size_t i = 0; i < 3; ++i) {
            chargeDerivativesO[vsH1][i] = -(chargeDerivativesH1[vsH1][i]+ chargeDerivativesH2[vsH1][i]);
            chargeDerivativesO[vsH2][i] =  -(chargeDerivativesH1[vsH2][i]+ chargeDerivativesH2[vsH2][i]);
            chargeDerivativesO[vsO][i] =  -(chargeDerivativesH1[vsO][i]+ chargeDerivativesH2[vsO][i]);
    }

    double sumH1, sumH2, sumO;

    for (size_t i = 0; i < 3; ++i) {
        particleM.chargeDerivatives[vsH1f][i] = 0.;
        particleM.chargeDerivatives[vsH2f][i] = 0.;
        particleM.chargeDerivatives[vsMf][i] = 0.;

        sumH1 = gamma2div1*(chargeDerivativesH1[vsH1][i]+chargeDerivativesH2[vsH1][i]);
        sumH2 = gamma2div1*(chargeDerivativesH1[vsH2][i]+chargeDerivativesH2[vsH2][i]);
        sumO = gamma2div1*(chargeDerivativesH1[vsO][i]+chargeDerivativesH2[vsO][i]);

        particleH1.chargeDerivatives[vsH1f][i] = chargeDerivativesH1[vsH1][i] + sumH1;
        particleH2.chargeDerivatives[vsH1f][i] = chargeDerivativesH1[vsH2][i] + sumH2;
        particleO.chargeDerivatives[vsH1f][i]  = chargeDerivativesH1[vsO][i] + sumO;

        particleH1.chargeDerivatives[vsH2f][i] = chargeDerivativesH2[vsH1][i] + sumH1;
        particleH2.chargeDerivatives[vsH2f][i] = chargeDerivativesH2[vsH2][i] + sumH2;
        particleO.chargeDerivatives[vsH2f][i]  = chargeDerivativesH2[vsO][i] +  sumO;

        particleH1.chargeDerivatives[vsMf][i] = chargeDerivativesO[vsH1][i] - 2*sumH1;
        particleH2.chargeDerivatives[vsMf][i] = chargeDerivativesO[vsH2][i] - 2*sumH2;
        particleO.chargeDerivatives[vsMf][i]  = chargeDerivativesO[vsO][i]  - 2*sumO;

    }

    // convert from q/A to q/nm
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int s = 0; s < 3; ++s) {
            particleH1.chargeDerivatives[s][i] *= 10;
            particleH2.chargeDerivatives[s][i] *= 10;
            particleO.chargeDerivatives[s][i] *= 10;
        }

    }

    particleH1.otherSiteIndex[vsH1f] = particleH1.particleIndex;
    particleH1.otherSiteIndex[vsH2f] = particleH2.particleIndex;
    particleH1.otherSiteIndex[vsMf]  = particleM.particleIndex;

    particleH2.otherSiteIndex[vsH1f] = particleH1.particleIndex;
    particleH2.otherSiteIndex[vsH2f] = particleH2.particleIndex;
    particleH2.otherSiteIndex[vsMf]  = particleM.particleIndex;

    particleM.otherSiteIndex[vsH1f] = particleH1.particleIndex;
    particleM.otherSiteIndex[vsH2f] = particleH2.particleIndex;
    particleM.otherSiteIndex[vsMf]  = particleM.particleIndex;

    particleO.otherSiteIndex[vsH1f] = particleH1.particleIndex;
    particleO.otherSiteIndex[vsH2f] = particleH2.particleIndex;
    particleO.otherSiteIndex[vsMf]  = particleM.particleIndex;

}

} // namespace SAPTMLPlugin
