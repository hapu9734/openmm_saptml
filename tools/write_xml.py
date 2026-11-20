#!/usr/bin/env python3
"""Write python/saptml/data/saptml_<tag>.xml from a base parameter file (JSON) of a SAPT-ML water tag with atom-type dispersion.
usage: write_xml.py <base_params.json> <tag> [2B model file name] [3B model file name]"""
import json, sys, os
P = json.load(open(sys.argv[1])); tag = sys.argv[2]
m2 = sys.argv[3] if len(sys.argv) > 3 else "saptml_%s_2b.wdpip" % tag
m3 = sys.argv[4] if len(sys.argv) > 4 else "saptml_3b_c.wtpip"
assert P.get("disp_form") == "atomtype", "the XML form needs atom-type C_n (geometric-mean rule)"
def atom(t):
    cn = P["Cn_site"].get(t, [0.0, 0.0, 0.0, 0.0])
    return ('    <Atom type="saptml-%s" Aexch="%.10g" Aelec="%.10g" Aind="%.10g" Adhf="%.10g" Bexp="%.10g" bd="%.10g" C6="%.10g" C8="%.10g" C10="%.10g" C12="%.10g"/>'
            % (t, P["Aexch"][t], P["Aelec"][t], P["Aind"][t], P["Adhf"][t], P["Bexp"][t], P["b_disp_nm"].get(t, 1.0), *cn))
xml = """<?xml version="1.0" encoding="utf-8"?>
<!-- SAPT-ML polarizable water, tag %(tag)s (2B PIP %(m2)s, 3B PIP %(m3)s).
     Flexible water: MB-pol one-body surface, MB-pol / TTM4-F electrostatics with geometry-dependent charges and Thole-damped
     mutual induction, SAPT-FF short range (fitted to SAPT2+ components on top of the MB-pol electrostatics), Tang-Toennies
     damped C6-C12 dispersion (CKS atom-type coefficients, geometric-mean rule, no global scale), 2B and 3B PIP corrections
     fitted to CCSD(T).  Use with: nonbondedMethod=PME, nonbondedCutoff=0.9 nm, rigidWater=False (default from the template),
     a 0.2 fs time step and the M sites added with Modeller.addExtraParticles.  Requires `import saptml` first. -->
<ForceField>
  <Info>
    <Reference>SAPT-ML water %(tag)s (2026)</Reference>
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
  <!-- 1B: Partridge-Schwenke surface with the MB-pol scaling, evaluated analytically (defined for every geometry) -->
  <PartridgeSchwenkeForce forceGroup="2">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </PartridgeSchwenkeForce>
  <!-- MB-pol electrostatics: PS charges on H and M, point dipoles on O and H (nm^3), TTM4-F Thole damping; PME when periodic.
       Solver: DIIS (history 20) from the quadratic-extrapolation predictor, RMS change < 1e-5 D, at most 60 iterations.
       Alternatives: inducedDipoleMethod="SOR" sorParameter="0.55" (over-relaxation), predictorOrder="0" (alpha E_fixed start). -->
  <MBPolElectrostaticsForce polarizabilityO="0.001310" polarizabilityH="0.000294" dampingO="0.001310" dampingH="0.000294" dampingM="0.001310"
      tholeCC="0.4" tholeCD="0.4" tholeDD="0.055" tholeDDOH="0.626" tholeDDHH="0.055" chargeRedistribution="true" polarization="mutual"
      mutualInducedMaxIterations="60" mutualInducedTargetEpsilon="1e-5" inducedDipoleMethod="DIIS" diisHistorySize="20" sorParameter="0.55"
      predictorOrder="3" ewaldErrorTolerance="1e-5" forceGroup="1">
    <Water oxygen="saptml-O" hydrogen="saptml-H" msite="saptml-M"/>
  </MBPolElectrostaticsForce>
  <!-- SAPT-FF short range on top of the electrostatics + TT-damped dispersion (kJ/mol, 1/nm, kJ/mol nm^n); the M site carries
       exchange / electrostatic / induction overlap terms but no dispersion -->
  <SAPTFFForce bondCutoff="3" forceGroup="3" tail="true">
%(atoms)s
  </SAPTFFForce>
  <!-- 2B PIP (switch 0.45-0.65 nm inside the model file); kernels: stream64 (default), stream32, double -->
  <WaterDimerPIPForce model="%(m2)s" maxNeighbors="64" overflowCheckInterval="1000" oxygenOxygenSwitch="0.18 0.19" intermolecularOHSwitch="0.085 0.100" intermolecularHHSwitch="0.050 0.065" forceGroup="6">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </WaterDimerPIPForce>
  <!-- 3B PIP (centred model for the float32 kernels); kernels: stream32 (default), f32, stream64, double -->
  <WaterTrimerPIPForce model="%(m3)s" maxNeighbors="64" overflowCheckInterval="1000" oxygenOxygenSwitch="0.18 0.19" intermolecularOHSwitch="0.090 0.110" intermolecularHHSwitch="0.045 0.058" forceGroup="5">
    <Water oxygen="saptml-O" hydrogen="saptml-H"/>
  </WaterTrimerPIPForce>
</ForceField>
""" % dict(tag=tag, m2=m2, m3=m3, atoms="\n".join(atom(t) for t in ("O", "H", "M")))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "saptml", "data", "saptml_%s.xml" % tag)
open(out, "w").write(xml); print("wrote", os.path.abspath(out))
