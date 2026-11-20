#!/usr/bin/env python3
"""NPT or NVT simulation of SAPT-ML water with the standard OpenMM workflow: input.pdb (+ forcefield.xml) -> run.py.

    python run.py [--steps N] [--pdb input.pdb] [--xml saptml.xml()] [--outdir .] [--platform CUDA] [--precision mixed]
                  [--temperature 298.15] [--pressure 1] [--barostat isotropic] [--friction 1] [--dt 0.2] [--cutoff 0.9]
                  [--integrator langevin|qtb] [--qtb-segment 0.4] [--qtb-cutoff 1500] [--qtb-adaptation-rate R]
                  [--report 500] [--dcd 5000] [--dipole 500] [--checkpoint 50000]
                  [--induced-dipole-method DIIS|SOR] [--sor 0.55] [--diis-history 20] [--scf-maxiter 60] [--scf-eps 1e-5]
                  [--predictor 3]

The water model is flexible (0.2 fs time step, no constraints: the residue template carries rigidWater="false").  The M sites
are added by Modeller.addExtraParticles when the PDB has none.  --pressure 0 gives NVT; --barostat anisotropic scales the
three box lengths independently (crystals).  The thermostat is LangevinMiddle with --friction (1/ps; a weak coupling such as
0.1/ps for dynamics and spectra).  --integrator qtb replaces it by the adaptive quantum thermal bath (adQTB,
QTBIntegrator, OpenMM >= 8.3) for nuclear quantum effects: O and H are separate particle types whose noise spectra adapt
segment by segment; adqtb.log records the adapted friction averaged over frequency bands and adqtb_friction.npz the last
full spectra.  The adQTB setup at Context creation costs ~(segment/dt)^3 on the CPU: 3 min for 0.4 ps at 0.2 fs, ~50 min
for OpenMM's default 1 ps.  With adQTB the temperature column of ener.log is the kinetic temperature of the quantum velocities.
The induced-dipole solver of the electrostatics (DIIS or successive over-relaxation, maximum iterations, convergence target
in Debye, predictor order) is set by ForceField XML attributes that can be overridden here through keyword arguments of
createSystem.  dipole.log every 10 steps (2 fs) gives the IR spectrum; each record recomputes the forces at the current
positions, so it adds one force evaluation.

Files written to --outdir:
    system.pdb    the starting structure with the M sites and the periodic box
    ener.log      step, time, potential / kinetic / total energy, temperature, box volume, density, speed (every --report steps)
    traj.dcd      trajectory (every --dcd steps)
    dipole.log    box dipole M = sum q r + sum mu (e nm, every --dipole steps; 0 = off), for the dielectric constant
    adqtb.log     adapted adQTB friction per frequency band (every --checkpoint steps, --integrator qtb)
    restart.chk   checkpoint (every --checkpoint steps and at the end)
    final.pdb     the last structure
--steps is the total length of the run: when restart.chk exists, the run continues from it up to --steps and the log files
and the trajectory are appended to.
"""
import argparse, os, sys, time
import saptml                                                    # loads the plugin and registers the XML tags
import openmm as mm
import openmm.app as app
import openmm.unit as unit

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pdb", default="input.pdb")
p.add_argument("--xml", default=None, help="force field XML (default: the packaged model, saptml.xml())")
p.add_argument("--outdir", default=".")
p.add_argument("--steps", type=int, default=5000, help="total number of steps of the run")
p.add_argument("--platform", default="CUDA")
p.add_argument("--precision", default="mixed")
p.add_argument("--temperature", type=float, default=298.15, help="K")
p.add_argument("--pressure", type=float, default=1.0, help="bar (0 = NVT)")
p.add_argument("--barostat", default="isotropic", choices=["isotropic", "anisotropic"], help="Monte Carlo barostat type")
p.add_argument("--friction", type=float, default=1.0, help="friction coefficient of the Langevin thermostat (1/ps)")
p.add_argument("--integrator", default="langevin", choices=["langevin", "qtb"], help="Langevin or adQTB")
p.add_argument("--qtb-segment", type=float, default=None, help="adQTB segment length (ps)")
p.add_argument("--qtb-cutoff", type=float, default=None, help="adQTB cutoff frequency of the colored noise (1/ps)")
p.add_argument("--qtb-adaptation-rate", type=float, default=None, help="adQTB adaptation rate of the O and H types")
p.add_argument("--dt", type=float, default=0.2, help="fs")
p.add_argument("--cutoff", type=float, default=0.9, help="nm")
p.add_argument("--report", type=int, default=500, help="steps between ener.log lines")
p.add_argument("--dcd", type=int, default=5000, help="steps between trajectory frames")
p.add_argument("--dipole", type=int, default=0, help="steps between dipole.log lines (0 = off)")
p.add_argument("--checkpoint", type=int, default=50000, help="steps between checkpoints")
p.add_argument("--induced-dipole-method", default=None, choices=["DIIS", "SOR"], help="induced-dipole solver")
p.add_argument("--sor", type=float, default=None, help="over-relaxation parameter of SOR (1 = Jacobi)")
p.add_argument("--diis-history", type=int, default=None, help="number of DIIS vectors (2-32)")
p.add_argument("--scf-maxiter", type=int, default=None, help="maximum induction iterations")
p.add_argument("--scf-eps", type=float, default=None, help="convergence target, RMS change of the induced dipoles (Debye)")
p.add_argument("--predictor", type=int, default=None, help="initial-guess predictor order 0-3")
p.add_argument("--seed", type=int, default=1)
a = p.parse_args()
os.makedirs(a.outdir, exist_ok=True)
out = lambda name: os.path.join(a.outdir, name)


class DipoleReporter(object):
    """Writes the box dipole M = sum_i q_i r_i + sum_i mu_i (e nm) of the MBPolElectrostaticsForce."""

    def __init__(self, file, reportInterval, force, append=False):
        self._out = open(file, "a" if append else "w")
        self._interval = reportInterval
        self._force = force
        if not append:
            self._out.write("# time (ps)  Mx  My  Mz (e nm)\n")

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        M = saptml.total_dipole(simulation.context, self._force)
        self._out.write("%.4f %.6f %.6f %.6f\n" % (state.getTime().value_in_unit(unit.picosecond), M[0], M[1], M[2]))
        self._out.flush()

    def __del__(self):
        self._out.close()


class AdQTBReporter(object):
    """Writes the adapted friction of the adQTB (O and H types, 1/ps) averaged over frequency bands (cm^-1) to adqtb.log and
    the last full spectra to adqtb_friction.npz."""

    BANDS = ((0, 1000), (1000, 2000), (2000, 3000), (3000, 4500))

    def __init__(self, file, reportInterval, integrator, particles, append=False):
        self._out = open(file, "a" if append else "w")
        self._interval = reportInterval
        self._integrator = integrator
        self._particles = particles
        if not append:
            self._out.write("# time (ps)  adapted friction (1/ps) of %s in the bands %s cm^-1\n" % (" and ".join(particles),
                            " ".join("%d-%d" % b for b in self.BANDS)))

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, False, False, False, False, None)

    def report(self, simulation, state):
        import numpy as np
        dt = self._integrator.getStepSize().value_in_unit(unit.picosecond)
        spectra = {name: np.asarray(self._integrator.getAdaptedFriction(i)) for name, i in self._particles.items()}
        n = len(next(iter(spectra.values())))
        wavenumber = np.arange(n)*np.pi/(n*dt)/(2*np.pi*0.0299792458)
        line = ["%.1f" % state.getTime().value_in_unit(unit.picosecond)]
        for name in self._particles:
            line += ["%.3f" % spectra[name][(wavenumber >= lo) & (wavenumber < hi)].mean() for lo, hi in self.BANDS]
        self._out.write(" ".join(line) + "\n")
        self._out.flush()
        np.savez(os.path.join(os.path.dirname(self._out.name), "adqtb_friction.npz"), wavenumber=wavenumber, **spectra)

    def __del__(self):
        self._out.close()



pdb = app.PDBFile(a.pdb)
ff = app.ForceField(a.xml or saptml.xml())
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addExtraParticles(ff)                                   # M sites (no-op when the PDB already has them)
overrides = {}
if a.induced_dipole_method is not None: overrides["inducedDipoleMethod"] = a.induced_dipole_method
if a.sor is not None: overrides["sorParameter"] = a.sor
if a.diis_history is not None: overrides["diisHistorySize"] = a.diis_history
if a.scf_maxiter is not None: overrides["mutualInducedMaxIterations"] = a.scf_maxiter
if a.scf_eps is not None: overrides["mutualInducedTargetEpsilon"] = a.scf_eps
if a.predictor is not None: overrides["predictorOrder"] = a.predictor
system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=a.cutoff * unit.nanometer, **overrides)
if a.pressure > 0:
    if a.barostat == "anisotropic":
        system.addForce(mm.MonteCarloAnisotropicBarostat(mm.Vec3(a.pressure, a.pressure, a.pressure) * unit.bar,
                                                         a.temperature * unit.kelvin, True, True, True, 25))
    else:
        system.addForce(mm.MonteCarloBarostat(a.pressure * unit.bar, a.temperature * unit.kelvin, 25))
if a.integrator == "qtb":
    integrator = mm.QTBIntegrator(a.temperature * unit.kelvin, a.friction / unit.picosecond, a.dt * unit.femtosecond)
    if a.qtb_segment is not None: integrator.setSegmentLength(a.qtb_segment * unit.picosecond)
    if a.qtb_cutoff is not None: integrator.setCutoffFrequency(a.qtb_cutoff / unit.picosecond)
    types = {"O": 0, "H": 1}
    for atom in modeller.topology.atoms():
        if atom.element is not None and atom.element.symbol in types:
            integrator.setParticleType(atom.index, types[atom.element.symbol])
    if a.qtb_adaptation_rate is not None:
        for t in types.values():
            integrator.setTypeAdaptationRate(t, a.qtb_adaptation_rate)
else:
    integrator = mm.LangevinMiddleIntegrator(a.temperature * unit.kelvin, a.friction / unit.picosecond, a.dt * unit.femtosecond)
integrator.setRandomNumberSeed(a.seed)
platform = mm.Platform.getPlatformByName(a.platform)
props = {"Precision": a.precision} if a.platform in ("CUDA", "OpenCL") else {}
sim = app.Simulation(modeller.topology, system, integrator, platform, props)

restart = os.path.exists(out("restart.chk"))
if restart:
    sim.loadCheckpoint(out("restart.chk"))
else:
    sim.context.setPositions(modeller.positions)
    sim.context.computeVirtualSites()
    sim.context.setVelocitiesToTemperature(a.temperature * unit.kelvin, a.seed)
    state = sim.context.getState(getPositions=True)
    sim.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    app.PDBFile.writeFile(sim.topology, state.getPositions(), open(out("system.pdb"), "w"))

F = saptml.forcefield.forces(system)
elec = F["MBPolElectrostatics"]
diis = elec.getInducedDipoleMethod() == saptml.MBPolElectrostaticsForce.DIIS
print("SAPT-ML water: %d atoms, %d waters, %s %.2f K%s, %s" % (system.getNumParticles(), elec.getNumWaters(),
      "NPT" if a.pressure > 0 else "NVT", a.temperature, " %.3f bar (%s)" % (a.pressure, a.barostat) if a.pressure > 0 else "",
      "restart from step %d" % sim.currentStep if restart else "new run"))
print("integrator %s, friction %g/ps\n" % ("adQTB" if a.integrator == "qtb" else "LangevinMiddle", a.friction), end="")
print("induction: %s, max %d iterations, target %.1e D, predictor order %d; platform %s %s" % (
      "DIIS (history %d)" % elec.getDIISHistorySize() if diis else "SOR (omega %.2f)" % elec.getSORParameter(),
      elec.getMutualInducedMaxIterations(), elec.getMutualInducedTargetEpsilon(), elec.getPredictorOrder(),
      a.platform, props.get("Precision", "")))
if not restart:
    print("initial potential energy %.3f kJ/mol" % sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    for g, name in ((2, "1B"), (1, "electrostatics"), (3, "SAPT-FF + dispersion"), (6, "2B PIP"), (5, "3B PIP")):
        e = sim.context.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print("   %-22s %12.3f kJ/mol" % (name, e))

remaining = a.steps - sim.currentStep
sim.reporters.append(app.StateDataReporter(out("ener.log"), a.report, step=True, time=True, potentialEnergy=True,
                                           kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True,
                                           speed=True, separator="\t", append=restart))
sim.reporters.append(app.StateDataReporter(sys.stdout, max(a.report, a.steps // 100), step=True, time=True, potentialEnergy=True,
                                           temperature=True, density=True, speed=True, progress=True, remainingTime=True,
                                           totalSteps=a.steps, separator="  "))
sim.reporters.append(app.DCDReporter(out("traj.dcd"), a.dcd, append=restart and os.path.exists(out("traj.dcd"))))
if a.dipole > 0:
    sim.reporters.append(DipoleReporter(out("dipole.log"), a.dipole, elec, append=restart and os.path.exists(out("dipole.log"))))
if a.integrator == "qtb":
    first = {}
    for atom in modeller.topology.atoms():
        if atom.element is not None and atom.element.symbol in ("O", "H") and atom.element.symbol not in first:
            first[atom.element.symbol] = atom.index
    sim.reporters.append(AdQTBReporter(out("adqtb.log"), a.checkpoint, integrator, first,
                                       append=restart and os.path.exists(out("adqtb.log"))))
sim.reporters.append(app.CheckpointReporter(out("restart.chk"), a.checkpoint))
t0 = time.time()
if remaining > 0:
    sim.step(remaining)
dt = time.time() - t0
if remaining > 0:
    print("%d steps in %.1f s: %.2f ns/day" % (remaining, dt, remaining * a.dt * 1e-6 / dt * 86400))
sim.saveCheckpoint(out("restart.chk"))
state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
sim.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
app.PDBFile.writeFile(sim.topology, state.getPositions(), open(out("final.pdb"), "w"))
print("done: step %d" % sim.currentStep)
