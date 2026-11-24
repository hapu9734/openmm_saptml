"""ForceField XML generators of the SAPT-ML water model.

Importing this module registers the tags PartridgeSchwenkeForce, MBPolElectrostaticsForce, WaterDimerPIPForce,
WaterTrimerPIPForce and SAPTFFForce with openmm.app.ForceField.
"""
import openmm as mm
import openmm.unit as unit
from openmm.app import forcefield as _ff
from openmm.app.forcefield import NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, LJPME

from . import find_file, pipmodel
from .saptmlplugin import MBPolElectrostaticsForce, PartridgeSchwenkeForce, WaterDimerPIPForce, WaterTrimerPIPForce

_PERIODIC = (CutoffPeriodic, Ewald, PME, LJPME)


def _bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _floats(s, n):
    v = [float(x) for x in str(s).replace(",", " ").split()]
    if len(v) != n:
        raise ValueError("saptml: expected %d numbers, got %r" % (n, s))
    return v


def _distance(value):
    return value.value_in_unit(unit.nanometer) if unit.is_quantity(value) else float(value)


class _WaterTypes(object):
    """The atom types of a water residue, read from a <Water oxygen= hydrogen= [msite=]/> child."""

    def __init__(self, element, tag, needM):
        waters = element.findall("Water")
        if len(waters) != 1:
            raise ValueError("saptml: <%s> needs exactly one <Water oxygen= hydrogen= [msite=]/> child" % tag)
        w = waters[0].attrib
        self.oxygen = w["oxygen"]
        self.hydrogen = w["hydrogen"]
        self.msite = w.get("msite")
        self.needM = needM
        if needM and self.msite is None:
            raise ValueError("saptml: <%s> needs the msite type of the water" % tag)

    def waters(self, data, tag):
        """[(O, H1, H2, M or None)] atom indices of every residue made of these types."""
        groups = {}
        for atom in data.atoms:
            if data.atomType[atom] in (self.oxygen, self.hydrogen, self.msite):
                groups.setdefault(atom.residue, []).append(atom)
        out = []
        for res in sorted(groups, key=lambda r: r.index):
            atoms = groups[res]
            O = [a.index for a in atoms if data.atomType[a] == self.oxygen]
            H = [a.index for a in atoms if data.atomType[a] == self.hydrogen]
            M = [a.index for a in atoms if self.msite is not None and data.atomType[a] == self.msite]
            if len(O) != 1 or len(H) != 2 or len(M) > 1 or (self.needM and len(M) != 1):
                raise ValueError("saptml <%s>: residue %s %d is not a water of types %s / %s / %s (found %d O, %d H, %d M); "
                                 "add the M sites with Modeller.addExtraParticles" % (tag, res.name, res.index, self.oxygen,
                                                                                      self.hydrogen, self.msite, len(O), len(H), len(M)))
            out.append((O[0], H[0], H[1], M[0] if M else None))
        return out


class PartridgeSchwenkeGenerator(object):
    """<PartridgeSchwenkeForce forceGroup="2"><Water oxygen= hydrogen=/>: the MB-pol one-body potential."""

    def __init__(self, ff, element):
        a = element.attrib
        self.group = int(a.get("forceGroup", 2))
        self.types = _WaterTypes(element, "PartridgeSchwenkeForce", needM=False)

    @staticmethod
    def parseElement(element, ff):
        ff.registerGenerator(PartridgeSchwenkeGenerator(ff, element))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        waters = self.types.waters(data, "PartridgeSchwenkeForce")
        force = PartridgeSchwenkeForce()
        for (o, h1, h2, m) in waters:
            force.addWater(o, h1, h2)
        force.setName("PartridgeSchwenke1B")
        force.setUsesPeriodicBoundaryConditions(False)
        force.setForceGroup(self.group)
        sys.addForce(force)


_ff.parsers["PartridgeSchwenkeForce"] = PartridgeSchwenkeGenerator.parseElement


class MBPolElectrostaticsGenerator(object):
    """<MBPolElectrostaticsForce ... forceGroup="1"><Water oxygen= hydrogen= msite=/>: MB-pol electrostatics with mutual
    induction.  Unset attributes keep the force defaults; createSystem keyword arguments override the KEYS attributes.
    Units: polarizabilities and damping factors nm^3, charges e, mutualInducedTargetEpsilon Debye."""
    KEYS = ("mutualInducedMaxIterations", "mutualInducedTargetEpsilon", "inducedDipoleMethod", "diisHistorySize",
            "sorParameter", "predictorOrder", "ewaldErrorTolerance")

    def __init__(self, ff, element):
        self.a = dict(element.attrib)
        self.types = _WaterTypes(element, "MBPolElectrostaticsForce", needM=True)

    @staticmethod
    def parseElement(element, ff):
        ff.registerGenerator(MBPolElectrostaticsGenerator(ff, element))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        a = dict(self.a)
        for k in self.KEYS:
            if args.get(k, None) is not None:
                a[k] = args[k]
        force = MBPolElectrostaticsForce()
        for (o, h1, h2, m) in self.types.waters(data, "MBPolElectrostaticsForce"):
            force.addWater(o, h1, h2, m)
        if nonbondedMethod in _PERIODIC:
            force.setNonbondedMethod(MBPolElectrostaticsForce.PME)
            force.setCutoffDistance(_distance(nonbondedCutoff))
        elif nonbondedMethod == NoCutoff:
            force.setNonbondedMethod(MBPolElectrostaticsForce.NoCutoff)
        else:
            raise ValueError("saptml: MBPolElectrostaticsForce supports NoCutoff (clusters) and PME / CutoffPeriodic (-> PME)")
        if "ewaldErrorTolerance" in a:
            force.setEwaldErrorTolerance(float(a["ewaldErrorTolerance"]))
        force.setPolarizationType(MBPolElectrostaticsForce.Direct if a.get("polarization", "mutual").lower() == "direct" else MBPolElectrostaticsForce.Mutual)
        aO, aH = force.getPolarizabilities()
        force.setPolarizabilities(float(a.get("polarizabilityO", aO)), float(a.get("polarizabilityH", aH)))
        dO, dH, dM = force.getDampingFactors()
        force.setDampingFactors(float(a.get("dampingO", dO)), float(a.get("dampingH", dH)), float(a.get("dampingM", dM)))
        cc, cd, dd, ddOH, ddHH = force.getTholeParameters()
        force.setTholeParameters(float(a.get("tholeCC", cc)), float(a.get("tholeCD", cd)), float(a.get("tholeDD", dd)),
                                 float(a.get("tholeDDOH", ddOH)), float(a.get("tholeDDHH", ddHH)))
        force.setIncludeChargeRedistribution(_bool(a.get("chargeRedistribution", "true")))
        qO, qH, qM = force.getFixedCharges()
        force.setFixedCharges(float(a.get("chargeO", qO)), float(a.get("chargeH", qH)), float(a.get("chargeM", qM)))
        force.setMutualInducedMaxIterations(int(a.get("mutualInducedMaxIterations", force.getMutualInducedMaxIterations())))
        force.setMutualInducedTargetEpsilon(float(a.get("mutualInducedTargetEpsilon", force.getMutualInducedTargetEpsilon())))
        method = str(a.get("inducedDipoleMethod", "DIIS")).upper()
        if method not in ("DIIS", "SOR"):
            raise ValueError("saptml: inducedDipoleMethod must be DIIS or SOR")
        force.setInducedDipoleMethod(MBPolElectrostaticsForce.DIIS if method == "DIIS" else MBPolElectrostaticsForce.SOR)
        force.setDIISHistorySize(int(a.get("diisHistorySize", force.getDIISHistorySize())))
        force.setSORParameter(float(a.get("sorParameter", force.getSORParameter())))
        force.setPredictorOrder(int(a.get("predictorOrder", force.getPredictorOrder())))
        force.setForceGroup(int(a.get("forceGroup", 1)))
        force.setName("MBPolElectrostatics")
        sys.addForce(force)


_ff.parsers["MBPolElectrostaticsForce"] = MBPolElectrostaticsGenerator.parseElement


class _PIPGenerator(object):
    """Shared implementation of <WaterDimerPIPForce> and <WaterTrimerPIPForce>: model="file" plus optional switching
    ranges (nm); the cutoff and the switching distance come from the model file."""
    TAG = None
    CLS = None
    LOADER = None
    GROUP = None
    MIN_WATERS = None

    def __init__(self, ff, element):
        self.a = dict(element.attrib)
        self.types = _WaterTypes(element, self.TAG, needM=False)

    @classmethod
    def parseElement(cls, element, ff):
        ff.registerGenerator(cls(ff, element))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        a = self.a
        waters = self.types.waters(data, self.TAG)
        if len(waters) < self.MIN_WATERS:
            return
        force = self.CLS()
        self.LOADER(force, find_file(a["model"]))
        force.setMaxNeighbors(int(a.get("maxNeighbors", 64)))
        force.setOverflowCheckInterval(int(a.get("overflowCheckInterval", 1000)))
        if "oxygenOxygenSwitch" in a:
            force.setOxygenOxygenSwitch(*_floats(a["oxygenOxygenSwitch"], 2))
        if "intermolecularOHSwitch" in a:
            force.setIntermolecularOHSwitch(*_floats(a["intermolecularOHSwitch"], 2))
        if "intermolecularHHSwitch" in a:
            force.setIntermolecularHHSwitch(*_floats(a["intermolecularHHSwitch"], 2))
        if "intramolecularOHWindow" in a:
            force.setIntramolecularOHWindow(*_floats(a["intramolecularOHWindow"], 4))
        if "intramolecularHHWindow" in a:
            force.setIntramolecularHHWindow(*_floats(a["intramolecularHHWindow"], 4))
        force.setNonbondedMethod(self.CLS.CutoffPeriodic if nonbondedMethod in _PERIODIC else self.CLS.CutoffNonPeriodic)
        for (o, h1, h2, m) in waters:
            force.addWater(o, h1, h2)
        force.setForceGroup(int(a.get("forceGroup", self.GROUP)))
        force.setName(self.TAG[:-5])
        sys.addForce(force)


class WaterDimerPIPGenerator(_PIPGenerator):
    TAG, CLS, GROUP, MIN_WATERS = "WaterDimerPIPForce", WaterDimerPIPForce, 6, 2
    LOADER = staticmethod(pipmodel.load_dimer_model)


class WaterTrimerPIPGenerator(_PIPGenerator):
    TAG, CLS, GROUP, MIN_WATERS = "WaterTrimerPIPForce", WaterTrimerPIPForce, 5, 3
    LOADER = staticmethod(pipmodel.load_trimer_model)


_ff.parsers["WaterDimerPIPForce"] = WaterDimerPIPGenerator.parseElement
_ff.parsers["WaterTrimerPIPForce"] = WaterTrimerPIPGenerator.parseElement


SAPT_EXPR = ("A*exp(-B*r) - disp;"
             "A=sqrt(Aexch1*Aexch2)-sqrt(Aelec1*Aelec2)-sqrt(Aind1*Aind2)-sqrt(Adhf1*Adhf2);"
             "B=(Bexp1+Bexp2)*Bexp1*Bexp2/(Bexp1^2+Bexp2^2);"
             "disp = f6*sqrt(C61*C62)/r^6 + f8*sqrt(C81*C82)/r^8 + f10*sqrt(C101*C102)/r^10 + f12*sqrt(C121*C122)/r^12;"
             "f12 = f10 - exBr*( (1/39916800)*(Br^11)*(1 + Br/12) );"
             "f10 = f8 - exBr*( (1/362880)*(Br^9)*(1 + Br/10 ) );"
             "f8 = f6 - exBr*( (1/5040)*(Br^7)*(1 + Br/8 ) );"
             "f6 = 1 - exBr*(1 + Br * (1 + (1/2)*Br*(1 + (1/3)*Br*(1 + (1/4)*Br*(1 + (1/5)*Br*(1 + (1/6)*Br ) ) ) ) ) );"
             "exBr = exp(-Br); Br = Bd*r; Bd=(bd1+bd2)*bd1*bd2/(bd1^2+bd2^2)")
TAIL_EXPR = "-step(r-rc)*(sqrt(C61*C62)/r^6 + sqrt(C81*C82)/r^8 + sqrt(C101*C102)/r^10 + sqrt(C121*C122)/r^12)"
SAPT_PARAMS = ("Aexch", "Aelec", "Aind", "Adhf", "Bexp", "bd", "C6", "C8", "C10", "C12")


class SAPTFFGenerator(object):
    """<SAPTFFForce bondCutoff="3" forceGroup="3" tail="true" [cutoff=]><Atom type= Aexch= ... C12=/>: SAPT-FF short range
    and damped dispersion; periodic systems also get the dispersion beyond the cutoff with a long-range correction.
    Units: A kJ/mol, Bexp and bd 1/nm, C_n kJ/mol nm^n, cutoff nm (default: nonbondedCutoff)."""

    def __init__(self, ff, element):
        a = element.attrib
        self.bondCutoff = int(a.get("bondCutoff", 3))
        self.group = int(a.get("forceGroup", 3))
        self.tail = _bool(a.get("tail", "true"))
        self.cutoff = float(a["cutoff"]) if "cutoff" in a else None
        self.params = {}
        for atom in element.findall("Atom"):
            key = atom.attrib.get("type")
            if key is None:
                raise ValueError("saptml: <SAPTFFForce><Atom> needs a type attribute")
            self.params[key] = [float(atom.attrib.get(n, 0.0)) for n in SAPT_PARAMS]

    @staticmethod
    def parseElement(element, ff):
        ff.registerGenerator(SAPTFFGenerator(ff, element))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        periodic = nonbondedMethod in _PERIODIC
        cutoff = _distance(nonbondedCutoff) if self.cutoff is None else self.cutoff
        force = mm.CustomNonbondedForce(SAPT_EXPR)
        for n in SAPT_PARAMS:
            force.addPerParticleParameter(n)
        values = []
        for atom in data.atoms:
            t = data.atomType[atom]
            if t not in self.params:
                raise ValueError("saptml: no SAPT-FF parameters for atom type %s (atom %d)" % (t, atom.index))
            values.append(self.params[t])
            force.addParticle(self.params[t])
        bonds = _ff._findBondsForExclusions(data, sys)
        force.createExclusionsFromBonds(bonds, self.bondCutoff)
        if periodic:
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(cutoff)
        elif nonbondedMethod == CutoffNonPeriodic:
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
            force.setCutoffDistance(cutoff)
        else:
            force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        force.setUseLongRangeCorrection(False)
        force.setForceGroup(self.group)
        force.setName("SAPTFF")
        sys.addForce(force)
        if periodic and self.tail:
            tail = mm.CustomNonbondedForce(TAIL_EXPR)
            tail.addGlobalParameter("rc", cutoff)
            for n in ("C6", "C8", "C10", "C12"):
                tail.addPerParticleParameter(n)
            for v in values:
                tail.addParticle(v[6:10])
            tail.createExclusionsFromBonds(bonds, self.bondCutoff)
            tail.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            tail.setCutoffDistance(cutoff)
            tail.setUseLongRangeCorrection(True)
            tail.setForceGroup(self.group)
            tail.setName("SAPTFFDispersionTail")
            sys.addForce(tail)


_ff.parsers["SAPTFFForce"] = SAPTFFGenerator.parseElement


def forces(system):
    """{name: force} of the SAPT-ML forces of a System; the plugin forces are cast to their classes (MBPolElectrostatics,
    PartridgeSchwenke1B, WaterDimerPIP, WaterTrimerPIP), SAPTFF and SAPTFFDispersionTail are found by name."""
    out = {}
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if MBPolElectrostaticsForce.isinstance(f):
            out["MBPolElectrostatics"] = MBPolElectrostaticsForce.cast(f)
        elif PartridgeSchwenkeForce.isinstance(f):
            out["PartridgeSchwenke1B"] = PartridgeSchwenkeForce.cast(f)
        elif WaterDimerPIPForce.isinstance(f):
            out["WaterDimerPIP"] = WaterDimerPIPForce.cast(f)
        elif WaterTrimerPIPForce.isinstance(f):
            out["WaterTrimerPIP"] = WaterTrimerPIPForce.cast(f)
        elif f.getName() in ("SAPTFF", "SAPTFFDispersionTail"):
            out[f.getName()] = f
    return out
