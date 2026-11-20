%module saptmlplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "MBPolElectrostaticsForce.h"
#include "PartridgeSchwenkeForce.h"
#include "WaterDimerPIPForce.h"
#include "WaterTrimerPIPForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
using namespace OpenMM;
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Convert C++ exceptions to Python exceptions.
 */
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace SAPTMLPlugin {

class MBPolElectrostaticsForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        PME = 1
    };
    enum PolarizationType {
        Mutual = 0,
        Direct = 1
    };
    enum InducedDipoleMethod {
        DIIS = 0,
        SOR = 1
    };
    MBPolElectrostaticsForce();
    int getNumWaters() const;
    int addWater(int oxygen, int hydrogen1, int hydrogen2, int msite);
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2, int msite);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    PolarizationType getPolarizationType() const;
    void setPolarizationType(PolarizationType type);
    int getMutualInducedMaxIterations() const;
    void setMutualInducedMaxIterations(int iterations);
    double getMutualInducedTargetEpsilon() const;
    void setMutualInducedTargetEpsilon(double epsilon);
    InducedDipoleMethod getInducedDipoleMethod() const;
    void setInducedDipoleMethod(InducedDipoleMethod method);
    int getDIISHistorySize() const;
    void setDIISHistorySize(int size);
    double getSORParameter() const;
    void setSORParameter(double omega);
    int getPredictorOrder() const;
    void setPredictorOrder(int order);
    bool getIncludeChargeRedistribution() const;
    void setIncludeChargeRedistribution(bool include);
    void setFixedCharges(double chargeO, double chargeH, double chargeM);
    void setPolarizabilities(double alphaO, double alphaH);
    void setDampingFactors(double dampO, double dampH, double dampM);
    void setTholeParameters(double cc, double cd, double dd, double ddOH, double ddHH);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    %apply int& OUTPUT {int& oxygen};
    %apply int& OUTPUT {int& hydrogen1};
    %apply int& OUTPUT {int& hydrogen2};
    %apply int& OUTPUT {int& msite};
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2, int& msite) const;
    %clear int& oxygen;
    %clear int& hydrogen1;
    %clear int& hydrogen2;
    %clear int& msite;

    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    %apply double& OUTPUT {double& chargeO};
    %apply double& OUTPUT {double& chargeH};
    %apply double& OUTPUT {double& chargeM};
    void getFixedCharges(double& chargeO, double& chargeH, double& chargeM) const;
    %clear double& chargeO;
    %clear double& chargeH;
    %clear double& chargeM;

    %apply double& OUTPUT {double& alphaO};
    %apply double& OUTPUT {double& alphaH};
    void getPolarizabilities(double& alphaO, double& alphaH) const;
    %clear double& alphaO;
    %clear double& alphaH;

    %apply double& OUTPUT {double& dampO};
    %apply double& OUTPUT {double& dampH};
    %apply double& OUTPUT {double& dampM};
    void getDampingFactors(double& dampO, double& dampH, double& dampM) const;
    %clear double& dampO;
    %clear double& dampH;
    %clear double& dampM;

    %apply double& OUTPUT {double& cc};
    %apply double& OUTPUT {double& cd};
    %apply double& OUTPUT {double& dd};
    %apply double& OUTPUT {double& ddOH};
    %apply double& OUTPUT {double& ddHH};
    void getTholeParameters(double& cc, double& cd, double& dd, double& ddOH, double& ddHH) const;
    %clear double& cc;
    %clear double& cd;
    %clear double& dd;
    %clear double& ddOH;
    %clear double& ddHH;

    %apply std::vector<Vec3>& OUTPUT {std::vector<Vec3>& dipoles};
    void getInducedDipoles(OpenMM::Context& context, std::vector<Vec3>& dipoles);
    %clear std::vector<Vec3>& dipoles;

    %apply std::vector<double>& OUTPUT {std::vector<double>& charges};
    void getCharges(OpenMM::Context& context, std::vector<double>& charges);
    %clear std::vector<double>& charges;

    /*
     * Add methods for casting a Force to an MBPolElectrostaticsForce.
    */
    %extend {
        static SAPTMLPlugin::MBPolElectrostaticsForce& cast(OpenMM::Force& force) {
            return dynamic_cast<SAPTMLPlugin::MBPolElectrostaticsForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<SAPTMLPlugin::MBPolElectrostaticsForce*>(&force) != NULL);
        }
    }
};

class WaterDimerPIPForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        CutoffNonPeriodic = 0,
        CutoffPeriodic = 1
    };
    enum VariableForm {
        Exponential = 0,
        Coulomb = 1
    };
    WaterDimerPIPForce();
    int getNumWaters() const;
    int getNumVariables() const;
    int getNumMonomials() const;
    int getNumTerms() const;
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    int addVariable(int site1, int site2, double k, double d0, VariableForm form);
    void setVariableParameters(int index, int site1, int site2, double k, double d0, VariableForm form);
    int addMonomial(int variable1, int variable2, int variable3, int variable4);
    void setMonomialParameters(int index, int variable1, int variable2, int variable3, int variable4);
    int addTerm(double coefficient, int firstMonomial, int numMonomials);
    void setTermParameters(int index, double coefficient, int firstMonomial, int numMonomials);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    void setExtraPointParameters(double inPlane, double outOfPlane);
    void setOxygenOxygenSwitch(double zeroDistance, double fullDistance);
    void setIntermolecularOHSwitch(double zeroDistance, double fullDistance);
    void setIntermolecularHHSwitch(double zeroDistance, double fullDistance);
    void setIntramolecularOHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    void setIntramolecularHHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    int getMaxNeighbors() const;
    void setMaxNeighbors(int maxNeighbors);
    int getOverflowCheckInterval() const;
    void setOverflowCheckInterval(int interval);
    bool usesPeriodicBoundaryConditions() const;

    %apply int& OUTPUT {int& oxygen};
    %apply int& OUTPUT {int& hydrogen1};
    %apply int& OUTPUT {int& hydrogen2};
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    %clear int& oxygen;
    %clear int& hydrogen1;
    %clear int& hydrogen2;

    %apply int& OUTPUT {int& site1};
    %apply int& OUTPUT {int& site2};
    %apply double& OUTPUT {double& k};
    %apply double& OUTPUT {double& d0};
    %apply int& OUTPUT {VariableForm& form};
    void getVariableParameters(int index, int& site1, int& site2, double& k, double& d0, VariableForm& form) const;
    %clear int& site1;
    %clear int& site2;
    %clear double& k;
    %clear double& d0;
    %clear VariableForm& form;

    %apply int& OUTPUT {int& variable1};
    %apply int& OUTPUT {int& variable2};
    %apply int& OUTPUT {int& variable3};
    %apply int& OUTPUT {int& variable4};
    void getMonomialParameters(int index, int& variable1, int& variable2, int& variable3, int& variable4) const;
    %clear int& variable1;
    %clear int& variable2;
    %clear int& variable3;
    %clear int& variable4;

    %apply double& OUTPUT {double& coefficient};
    %apply int& OUTPUT {int& firstMonomial};
    %apply int& OUTPUT {int& numMonomials};
    void getTermParameters(int index, double& coefficient, int& firstMonomial, int& numMonomials) const;
    %clear double& coefficient;
    %clear int& firstMonomial;
    %clear int& numMonomials;

    %apply double& OUTPUT {double& inPlane};
    %apply double& OUTPUT {double& outOfPlane};
    void getExtraPointParameters(double& inPlane, double& outOfPlane) const;
    %clear double& inPlane;
    %clear double& outOfPlane;

    %apply double& OUTPUT {double& zeroDistance};
    %apply double& OUTPUT {double& fullDistance};
    void getOxygenOxygenSwitch(double& zeroDistance, double& fullDistance) const;
    void getIntermolecularOHSwitch(double& zeroDistance, double& fullDistance) const;
    void getIntermolecularHHSwitch(double& zeroDistance, double& fullDistance) const;
    %clear double& zeroDistance;
    %clear double& fullDistance;

    %apply double& OUTPUT {double& lowerZero};
    %apply double& OUTPUT {double& lowerFull};
    %apply double& OUTPUT {double& upperFull};
    %apply double& OUTPUT {double& upperZero};
    void getIntramolecularOHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    void getIntramolecularHHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    %clear double& lowerZero;
    %clear double& lowerFull;
    %clear double& upperFull;
    %clear double& upperZero;

    %extend {
        static SAPTMLPlugin::WaterDimerPIPForce& cast(OpenMM::Force& force) {
            return dynamic_cast<SAPTMLPlugin::WaterDimerPIPForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<SAPTMLPlugin::WaterDimerPIPForce*>(&force) != NULL);
        }
    }
};

class WaterTrimerPIPForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        CutoffNonPeriodic = 0,
        CutoffPeriodic = 1
    };
    WaterTrimerPIPForce();
    int getNumWaters() const;
    int getNumVariables() const;
    int getNumMonomials() const;
    int getNumTerms() const;
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    int addVariable(int site1, int site2, double k, double d0, double center=0.0);
    void setVariableParameters(int index, int site1, int site2, double k, double d0, double center);
    int addMonomial(int variable1, int variable2, int variable3, int variable4);
    void setMonomialParameters(int index, int variable1, int variable2, int variable3, int variable4);
    int addTerm(double coefficient, int firstMonomial, int numMonomials);
    void setTermParameters(int index, double coefficient, int firstMonomial, int numMonomials);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getSwitchingDistance() const;
    void setSwitchingDistance(double distance);
    void setOxygenOxygenSwitch(double zeroDistance, double fullDistance);
    void setIntermolecularOHSwitch(double zeroDistance, double fullDistance);
    void setIntermolecularHHSwitch(double zeroDistance, double fullDistance);
    void setIntramolecularOHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    void setIntramolecularHHWindow(double lowerZero, double lowerFull, double upperFull, double upperZero);
    int getMaxNeighbors() const;
    void setMaxNeighbors(int maxNeighbors);
    int getOverflowCheckInterval() const;
    void setOverflowCheckInterval(int interval);
    bool usesPeriodicBoundaryConditions() const;

    %apply int& OUTPUT {int& oxygen};
    %apply int& OUTPUT {int& hydrogen1};
    %apply int& OUTPUT {int& hydrogen2};
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    %clear int& oxygen;
    %clear int& hydrogen1;
    %clear int& hydrogen2;

    %apply int& OUTPUT {int& site1};
    %apply int& OUTPUT {int& site2};
    %apply double& OUTPUT {double& k};
    %apply double& OUTPUT {double& d0};
    %apply double& OUTPUT {double& center};
    void getVariableParameters(int index, int& site1, int& site2, double& k, double& d0, double& center) const;
    %clear int& site1;
    %clear int& site2;
    %clear double& k;
    %clear double& d0;
    %clear double& center;

    %apply int& OUTPUT {int& variable1};
    %apply int& OUTPUT {int& variable2};
    %apply int& OUTPUT {int& variable3};
    %apply int& OUTPUT {int& variable4};
    void getMonomialParameters(int index, int& variable1, int& variable2, int& variable3, int& variable4) const;
    %clear int& variable1;
    %clear int& variable2;
    %clear int& variable3;
    %clear int& variable4;

    %apply double& OUTPUT {double& coefficient};
    %apply int& OUTPUT {int& firstMonomial};
    %apply int& OUTPUT {int& numMonomials};
    void getTermParameters(int index, double& coefficient, int& firstMonomial, int& numMonomials) const;
    %clear double& coefficient;
    %clear int& firstMonomial;
    %clear int& numMonomials;

    %apply double& OUTPUT {double& zeroDistance};
    %apply double& OUTPUT {double& fullDistance};
    void getOxygenOxygenSwitch(double& zeroDistance, double& fullDistance) const;
    void getIntermolecularOHSwitch(double& zeroDistance, double& fullDistance) const;
    void getIntermolecularHHSwitch(double& zeroDistance, double& fullDistance) const;
    %clear double& zeroDistance;
    %clear double& fullDistance;

    %apply double& OUTPUT {double& lowerZero};
    %apply double& OUTPUT {double& lowerFull};
    %apply double& OUTPUT {double& upperFull};
    %apply double& OUTPUT {double& upperZero};
    void getIntramolecularOHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    void getIntramolecularHHWindow(double& lowerZero, double& lowerFull, double& upperFull, double& upperZero) const;
    %clear double& lowerZero;
    %clear double& lowerFull;
    %clear double& upperFull;
    %clear double& upperZero;

    %extend {
        static SAPTMLPlugin::WaterTrimerPIPForce& cast(OpenMM::Force& force) {
            return dynamic_cast<SAPTMLPlugin::WaterTrimerPIPForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<SAPTMLPlugin::WaterTrimerPIPForce*>(&force) != NULL);
        }
    }
};


class PartridgeSchwenkeForce : public OpenMM::Force {
public:
    PartridgeSchwenkeForce();
    int getNumWaters() const;
    int addWater(int oxygen, int hydrogen1, int hydrogen2);
    void setWaterParameters(int index, int oxygen, int hydrogen1, int hydrogen2);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    %apply int& OUTPUT {int& oxygen};
    %apply int& OUTPUT {int& hydrogen1};
    %apply int& OUTPUT {int& hydrogen2};
    void getWaterParameters(int index, int& oxygen, int& hydrogen1, int& hydrogen2) const;
    %clear int& oxygen;
    %clear int& hydrogen1;
    %clear int& hydrogen2;

    %extend {
        static SAPTMLPlugin::PartridgeSchwenkeForce& cast(OpenMM::Force& force) {
            return dynamic_cast<SAPTMLPlugin::PartridgeSchwenkeForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<SAPTMLPlugin::PartridgeSchwenkeForce*>(&force) != NULL);
        }
    }
};

}
