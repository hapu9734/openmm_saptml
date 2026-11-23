"""SAPT-ML polarizable water for OpenMM.

Importing this package loads the SAPTMLPlugin platform kernels and registers the force field XML tags::

    import saptml
    from openmm import app, unit
    pdb = app.PDBFile("input.pdb")
    forcefield = app.ForceField(saptml.xml())
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addExtraParticles(forcefield)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9*unit.nanometer)

The kernels are loaded from ``SAPTML_PLUGIN_DIR`` if set, else from the OpenMM plugin directory (by ``import openmm``),
else from ``<prefix>/lib/plugins``.
"""
import glob as _glob
import os as _os
import openmm as _mm

__version__ = "1.0.0"
_here = _os.path.dirname(_os.path.abspath(__file__))
prefix = _os.path.abspath(_os.path.join(_here, _os.pardir, _os.pardir))
data_dir = _os.path.join(_here, "data")

from .saptmlplugin import MBPolElectrostaticsForce, PartridgeSchwenkeForce, WaterDimerPIPForce, WaterTrimerPIPForce


def _has_kernels(d):
    return bool(d) and bool(_glob.glob(_os.path.join(d, "*SAPTMLPlugin*")))


_auto_dirs = [_mm.Platform.getDefaultPluginsDirectory()]          # directories OpenMM loads at import
try:
    from openmm import version as _omm_version
    _auto_dirs.insert(0, _os.path.join(_omm_version.openmm_library_path, "plugins"))
except Exception:
    pass
_auto = next((d for d in _auto_dirs if _has_kernels(d)), None)
_loaded = []
if _os.environ.get("SAPTML_PLUGIN_DIR") or _auto is None:
    plugin_dir = _os.environ.get("SAPTML_PLUGIN_DIR", _os.path.join(prefix, "lib", "plugins"))
    if _has_kernels(plugin_dir):
        _loaded = list(_mm.Platform.loadPluginsFromDirectory(plugin_dir))
    else:
        import warnings
        warnings.warn("saptml: no SAPTMLPlugin kernels in %s or %s; the platform kernels are not loaded" % (plugin_dir, _auto_dirs))
else:
    plugin_dir = _auto                                            # loaded by `import openmm`
    _loaded = [p for p in getattr(_mm, "pluginLoadedLibNames", []) if "SAPTMLPlugin" in p]
    if not _loaded:
        import warnings
        warnings.warn("saptml: %s has the SAPTMLPlugin kernels but OpenMM did not load them (OPENMM_PLUGIN_DIR=%s)"
                      % (plugin_dir, _os.environ.get("OPENMM_PLUGIN_DIR")))
_failures = [f for f in _mm.Platform.getPluginLoadFailures() if "SAPTML" in f]
if _failures:
    import warnings
    warnings.warn("saptml: some platform kernels could not be loaded: " + "; ".join(_failures))


def loaded_platforms():
    """Names of the platform kernel libraries loaded from the plugin directory."""
    return [_os.path.basename(p) for p in _loaded]


def data(name):
    """Absolute path of a file shipped in saptml/data (force field XML, model files)."""
    return _os.path.join(data_dir, name)


def xml(name="saptml.xml"):
    """Path of a force field XML shipped with the package (default: the SAPT-ML water model)."""
    return data(name)


def find_file(name):
    """Resolve a data file named in a force field XML: an absolute path, a path relative to the working directory,
    the directories of SAPTML_PATH, or the package data directory."""
    candidates = [name]
    for d in _os.environ.get("SAPTML_PATH", "").split(_os.pathsep) + [data_dir]:
        if d:
            candidates.append(_os.path.join(d, name))
    for c in candidates:
        if _os.path.isfile(c):
            return _os.path.abspath(c)
    raise FileNotFoundError("saptml: cannot find %s (searched %s)" % (name, ", ".join(candidates)))


def total_dipole(context, force=None):
    """Box dipole sum_i q_i r_i + sum_i mu_i (e nm) from the charges and induced dipoles of the MBPolElectrostaticsForce
    (found in the System if force is None)."""
    import numpy as _np
    import openmm.unit as _u
    if force is None:
        system = context.getSystem()
        for i in range(system.getNumForces()):
            if MBPolElectrostaticsForce.isinstance(system.getForce(i)):
                force = MBPolElectrostaticsForce.cast(system.getForce(i))
                break
        else:
            raise ValueError("saptml.total_dipole: the System has no MBPolElectrostaticsForce")
    x = _np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(_u.nanometer))
    q = _np.asarray(force.getCharges(context))
    mu = _np.asarray([[v.x, v.y, v.z] for v in force.getInducedDipoles(context)])
    return (q[:, None]*x).sum(0) + mu.sum(0)


from . import pipmodel
from . import forcefield as _forcefield                               # registers the XML generators
__all__ = ["MBPolElectrostaticsForce", "PartridgeSchwenkeForce", "WaterDimerPIPForce", "WaterTrimerPIPForce", "xml",
           "data", "find_file", "loaded_platforms", "total_dipole", "pipmodel", "prefix", "plugin_dir"]
