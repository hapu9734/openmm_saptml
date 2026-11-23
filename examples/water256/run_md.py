from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import saptml

pdb = PDBFile('input.pdb')
forcefield = ForceField(saptml.xml())
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addExtraParticles(forcefield)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer)
integrator = LangevinMiddleIntegrator(298.15*kelvin, 1/picosecond, 0.0002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator, Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'})
simulation.context.setPositions(modeller.positions)
simulation.context.setVelocitiesToTemperature(298.15*kelvin)
simulation.reporters.append(DCDReporter('output.dcd', 1000))
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True, speed=True))
simulation.step(10000)
