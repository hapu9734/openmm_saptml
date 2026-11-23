#!/usr/bin/env python3
"""Checks of the installed saptml package against reference energies from an independent implementation.
  1. 12-water cluster (no cutoff), Reference platform: per-term energies and forces (reference_cluster.npz);
  2. 256-water box (examples/water256/input.pdb, PME 0.9 nm), CUDA double: per-term energies (reference_box.npz);
  3. 1000 steps of NPT dynamics on CUDA mixed precision with the default settings.
CUDA checks are skipped when no CUDA device is available.  Usage: python tests/test_saptml.py"""
import os, sys, time
import numpy as np
import saptml
import openmm as mm, openmm.app as app, openmm.unit as unit

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"1B": 2, "elec": 1, "sapt": 3, "3B": 5, "2B": 6}
TIGHT = dict(mutualInducedTargetEpsilon=1e-7, mutualInducedMaxIterations=1000)
ok = True


def report(name, cond, msg):
    global ok
    ok &= bool(cond)
    print("%s %-36s %s" % ("PASS" if cond else "FAIL", name, msg), flush=True)


def energies(ctx):
    out = {k: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for k, g in GROUPS.items()}
    out["total"] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return out


def compare(tag, ctx, ref, tol):
    e = energies(ctx)
    for k in list(GROUPS) + ["total"]:
        d = e[k] - float(ref["E_" + k])
        report("%s %s" % (tag, k), abs(d) < tol[k], "%.5f vs %.5f kJ/mol (diff %.1e, tol %.0e)" % (e[k], float(ref["E_" + k]), d, tol[k]))


print("saptml %s from %s; kernels from %s: %s" % (saptml.__version__, os.path.dirname(saptml.__file__), saptml.plugin_dir,
                                                 saptml.loaded_platforms()))
ff = app.ForceField(saptml.xml())

# 1. cluster, Reference platform
refc = np.load(os.path.join(HERE, "reference_cluster.npz"))
xw = refc["positions"].reshape(12, 4, 3)
top = app.Topology(); chain = top.addChain(); pos = []
for i in range(12):
    res = top.addResidue("HOH", chain)
    O = top.addAtom("O", app.element.oxygen, res); H1 = top.addAtom("H1", app.element.hydrogen, res)
    H2 = top.addAtom("H2", app.element.hydrogen, res)
    top.addBond(O, H1); top.addBond(O, H2); pos += [xw[i, 0], xw[i, 1], xw[i, 2]]
mc = app.Modeller(top, np.array(pos) * unit.nanometer); mc.addExtraParticles(ff)
system = ff.createSystem(mc.topology, nonbondedMethod=app.NoCutoff, **TIGHT)
F = saptml.forcefield.forces(system)
report("force classes", set(F) >= {"MBPolElectrostatics", "WaterDimerPIP", "WaterTrimerPIP", "PartridgeSchwenke1B", "SAPTFF"}, str(sorted(F)))
ctx = mm.Context(system, mm.VerletIntegrator(1e-4), mm.Platform.getPlatformByName("Reference"))
ctx.setPositions(mc.positions); ctx.computeVirtualSites()
compare("cluster/Reference", ctx, refc, {"1B": 1e-3, "elec": 2e-3, "sapt": 1e-4, "2B": 1e-4, "3B": 2e-3, "total": 4e-3})
f = np.array(ctx.getState(getForces=True).getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
report("cluster/Reference forces", np.abs(f - refc["forces"]).max() < 0.05, "max |dF| %.1e kJ/mol/nm (max |F| %.0f)" % (
    np.abs(f - refc["forces"]).max(), np.abs(f).max()))
del ctx

# 2. and 3. box on CUDA
try:
    cuda = mm.Platform.getPlatformByName("CUDA")
    _s = mm.System(); _s.addParticle(1.0); mm.Context(_s, mm.VerletIntegrator(1e-4), cuda)
except Exception as err:
    cuda = None
    print("SKIP CUDA checks (%s)" % str(err).splitlines()[0])
if cuda is not None:
    refb = np.load(os.path.join(HERE, "reference_box.npz"))
    pdb = app.PDBFile(os.path.join(HERE, os.pardir, "examples", "water256", "input.pdb"))
    mb = app.Modeller(pdb.topology, pdb.positions); mb.addExtraParticles(ff)
    system = ff.createSystem(mb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer, **TIGHT)
    ctx = mm.Context(system, mm.VerletIntegrator(1e-4), cuda, {"Precision": "double"})
    ctx.setPositions(mb.positions); ctx.computeVirtualSites()
    compare("box/CUDA double", ctx, refb, {"1B": 5e-3, "elec": 5e-3, "sapt": 1e-3, "2B": 1e-3, "3B": 5e-3, "total": 1e-2})
    del ctx
    system = ff.createSystem(mb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 298.15 * unit.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond, 0.2 * unit.femtosecond); integ.setRandomNumberSeed(7)
    ctx = mm.Context(system, integ, cuda, {"Precision": "mixed"})
    ctx.setPositions(mb.positions); ctx.computeVirtualSites(); ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, 7)
    t0 = time.time(); integ.step(1000); st = ctx.getState(getEnergy=True); wall = time.time() - t0
    T = 2 * st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / (3 * 768 * 8.314462618e-3)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    report("NPT 1000 steps (mixed)", np.isfinite(pe) and 250 < T < 350, "PE %.1f kJ/mol, T %.1f K, %.2f ns/day (%s)" % (
        pe, T, 1000 * 0.2e-6 / wall * 86400, cuda.getPropertyValue(ctx, "DeviceName")))
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
