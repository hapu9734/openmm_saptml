#!/usr/bin/env python3
"""Write a SAPT-ML force field XML from a parameter file (JSON).
usage: write_xml.py <params.json> [output.xml] [2B model file] [3B model file]"""
import json, os, sys

P = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                                           "python", "saptml", "data", "saptml.xml")
m2 = sys.argv[3] if len(sys.argv) > 3 else "saptml_2b.wdpip"
m3 = sys.argv[4] if len(sys.argv) > 4 else "saptml_3b.wtpip"
assert P.get("disp_form") == "atomtype", "the XML form needs atom-type C_n (geometric-mean rule)"


def atom(t):
    cn = P["Cn_site"].get(t, [0.0, 0.0, 0.0, 0.0])
    return ('    <Atom type="saptml-%s" Aexch="%.10g" Aelec="%.10g" Aind="%.10g" Adhf="%.10g" Bexp="%.10g" bd="%.10g" C6="%.10g" C8="%.10g" C10="%.10g" C12="%.10g"/>'
            % (t, P["Aexch"][t], P["Aelec"][t], P["Aind"][t], P["Adhf"][t], P["Bexp"][t], P["b_disp_nm"].get(t, 1.0), *cn))


xml = """<?xml version="1.0" encoding="utf-8"?>
<!-- SAPT-ML flexible, polarizable water.  Requires `import saptml`.  Use nonbondedMethod=PME, nonbondedCutoff=0.9 nm,
     a 0.2 fs time step, and Modeller.addExtraParticles for the M sites. -->
<ForceField>
  <Info>
    <Reference>S. Bae and C. Y. Son, Δ-ML Corrected Many-Body Water Model from Symmetry-Adapted Perturbation Theory (in preparation)</Reference>
  </Info>
  <AtomTypes>
    <Type name="saptml-O" class="OW" element="O" mass="15.99943"/>
    <Type name="saptml-H" class="HW" element="H" mass="1.007947"/>
    <Type name="saptml-M" class="MW" mass="0"/>
  </AtomTypes>
  <Residues>
    <Residue name="HOH" rigidWater="false">
      <Atom name="O" type="saptml-O"/>
      <Atom name="H1" type="saptml-H"/>
      <Atom name="H2" type="saptml-H"/>
      <Atom name="M" type="saptml-M"/>
      <VirtualSite type="average3" siteName="M" atomName1="O" atomName2="H1" atomName3="H2" weight1="0.573293118" weight2="0.213353441" weight3="0.213353441"/>
      <Bond atomName1="O" atomName2="H1"/>
      <Bond atomName1="O" atomName2="H2"/>
    </Residue>
  </Residues>
  <!-- one-body: Partridge-Schwenke surface with the MB-pol scaling -->
  <PartridgeSchwenkeForce forceGroup="2">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </PartridgeSchwenkeForce>
  <!-- electrostatics: geometry-dependent charges, Thole-damped induced dipoles (polarizabilities in nm^3) -->
  <MBPolElectrostaticsForce polarizabilityO="0.001310" polarizabilityH="0.000294" dampingO="0.001310" dampingH="0.000294" dampingM="0.001310"
      tholeCC="0.4" tholeCD="0.4" tholeDD="0.055" tholeDDOH="0.626" tholeDDHH="0.055" chargeRedistribution="true" polarization="mutual"
      mutualInducedMaxIterations="60" mutualInducedTargetEpsilon="1e-5" inducedDipoleMethod="DIIS" diisHistorySize="20" sorParameter="0.55"
      predictorOrder="3" ewaldErrorTolerance="1e-5" forceGroup="1">
    <Water oxygen="saptml-O" hydrogen="saptml-H" msite="saptml-M"/>
  </MBPolElectrostaticsForce>
  <!-- SAPT-FF short range and Tang-Toennies damped dispersion (kJ/mol, 1/nm, kJ/mol nm^n) -->
  <SAPTFFForce bondCutoff="3" forceGroup="3" tail="true">
%(atoms)s
  </SAPTFFForce>
  <!-- two-body PIP; the switching range and cutoff are read from the model file -->
  <WaterDimerPIPForce model="%(m2)s" maxNeighbors="64" overflowCheckInterval="1000" oxygenOxygenSwitch="0.18 0.19" intermolecularOHSwitch="0.085 0.100" intermolecularHHSwitch="0.050 0.065" forceGroup="6">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </WaterDimerPIPForce>
  <!-- three-body PIP -->
  <WaterTrimerPIPForce model="%(m3)s" maxNeighbors="64" overflowCheckInterval="1000" oxygenOxygenSwitch="0.18 0.19" intermolecularOHSwitch="0.090 0.110" intermolecularHHSwitch="0.045 0.058" forceGroup="5">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </WaterTrimerPIPForce>
</ForceField>
""" % dict(m2=m2, m3=m3, atoms="\n".join(atom(t) for t in ("O", "H", "M")))
with open(out, "w", encoding="utf-8") as f:
    f.write(xml)
print("wrote", os.path.abspath(out))
