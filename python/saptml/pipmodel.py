"""Readers of the polynomial model files of the SAPT-ML water model.

A two-body model (``.wdpip``) and a three-body model (``.wtpip``) are text files::

    WDPIP_MODEL 1                         WTPIP_MODEL 1
    COUNTS nv maxDegree nt nm             COUNTS nv maxDegree nt nm
    SWITCH switchingDistance cutoff       SWITCH switchingDistance cutoff
    EXTRAPOINTS inPlane outOfPlane        VARIABLES
    VARIABLES                               i site1 site2 k d0            (nv lines)
      i site1 site2 k d0 form             CENTER c_0 ... c_(nv-1)        (optional)
    TERMS                                 TERMS
      i coefficient first count degree      i coefficient first count degree
    MONOMIALS                             MONOMIALS
      i v1 v2 v3 v4                         i v1 v2 v3 v4
    END                                   END

Distances are in nm, k in 1/Angstrom, d0 in Angstrom, coefficients in kJ/mol; form is 0 (exponential) or 1 (Coulomb);
unused monomial slots are -1.
"""
from .saptmlplugin import WaterDimerPIPForce, WaterTrimerPIPForce


def _tokens(path):
    with open(path) as f:
        return f.read().split()


def _read_polynomial(tokens, pos, force, numVariables, numTerms, numMonomials):
    """Add the TERMS and MONOMIALS sections to the force; returns the position of END."""
    if tokens[pos] != "TERMS":
        raise ValueError("expected TERMS, found %s" % tokens[pos])
    pos += 1
    terms = []
    for i in range(numTerms):
        if int(tokens[pos]) != i:
            raise ValueError("term indices must be contiguous")
        terms.append((float(tokens[pos+1]), int(tokens[pos+2]), int(tokens[pos+3])))
        pos += 5
    if tokens[pos] != "MONOMIALS":
        raise ValueError("expected MONOMIALS, found %s" % tokens[pos])
    pos += 1
    for i in range(numMonomials):
        if int(tokens[pos]) != i:
            raise ValueError("monomial indices must be contiguous")
        force.addMonomial(int(tokens[pos+1]), int(tokens[pos+2]), int(tokens[pos+3]), int(tokens[pos+4]))
        pos += 5
    for coefficient, first, count in terms:
        force.addTerm(coefficient, first, count)
    if tokens[pos] != "END":
        raise ValueError("expected END, found %s" % tokens[pos])
    return pos


def load_dimer_model(force, path):
    """Define the polynomial, the switching distances and the extra points of a WaterDimerPIPForce from a .wdpip file."""
    t = _tokens(path)
    if t[0] != "WDPIP_MODEL" or int(t[1]) != 1:
        raise ValueError("%s is not a WDPIP_MODEL 1 file" % path)
    if t[2] != "COUNTS":
        raise ValueError("expected COUNTS")
    numVariables, numTerms, numMonomials = int(t[3]), int(t[5]), int(t[6])
    if t[7] != "SWITCH" or t[10] != "EXTRAPOINTS" or t[13] != "VARIABLES":
        raise ValueError("unexpected section order in %s" % path)
    force.setSwitchingDistance(float(t[8]))
    force.setCutoffDistance(float(t[9]))
    force.setExtraPointParameters(float(t[11]), float(t[12]))
    pos = 14
    for i in range(numVariables):
        if int(t[pos]) != i:
            raise ValueError("variable indices must be contiguous")
        form = WaterDimerPIPForce.Coulomb if int(t[pos+5]) == 1 else WaterDimerPIPForce.Exponential
        force.addVariable(int(t[pos+1]), int(t[pos+2]), float(t[pos+3]), float(t[pos+4]), form)
        pos += 6
    _read_polynomial(t, pos, force, numVariables, numTerms, numMonomials)
    return force


def load_trimer_model(force, path):
    """Define the polynomial and the switching distances of a WaterTrimerPIPForce from a .wtpip file."""
    t = _tokens(path)
    if t[0] != "WTPIP_MODEL" or int(t[1]) != 1:
        raise ValueError("%s is not a WTPIP_MODEL 1 file" % path)
    if t[2] != "COUNTS":
        raise ValueError("expected COUNTS")
    numVariables, numTerms, numMonomials = int(t[3]), int(t[5]), int(t[6])
    if t[7] != "SWITCH" or t[10] != "VARIABLES":
        raise ValueError("unexpected section order in %s" % path)
    force.setSwitchingDistance(float(t[8]))
    force.setCutoffDistance(float(t[9]))
    pos = 11
    variables = []
    for i in range(numVariables):
        if int(t[pos]) != i:
            raise ValueError("variable indices must be contiguous")
        variables.append((int(t[pos+1]), int(t[pos+2]), float(t[pos+3]), float(t[pos+4])))
        pos += 5
    centers = [0.0]*numVariables
    if t[pos] == "CENTER":
        centers = [float(v) for v in t[pos+1:pos+1+numVariables]]
        pos += 1+numVariables
    for (site1, site2, k, d0), center in zip(variables, centers):
        force.addVariable(site1, site2, k, d0, center)
    _read_polynomial(t, pos, force, numVariables, numTerms, numMonomials)
    return force
