"""saptml Python package; configured by CMake and installed with `make PythonInstall`."""
import os
from setuptools import setup, Extension

openmm_dir = '@OPENMM_DIR@'
saptml_header_dir = '@SAPTML_HEADER_DIR@'
saptml_library_dir = '@SAPTML_LIBRARY_DIR@'
install_lib_dir = os.path.join('@CMAKE_INSTALL_PREFIX@', 'lib')

extension = Extension(name='saptml._saptmlplugin',
                      sources=['SAPTMLPluginWrapper.cpp'],
                      libraries=['OpenMM', 'SAPTMLPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), saptml_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), saptml_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), install_lib_dir],
                      extra_compile_args=['-std=c++11'],
                     )

setup(name='saptml',
      version='1.0.0',
      description='SAPT-ML polarizable water model for OpenMM',
      author='Sangwon Bae, Chang Yun Son',
      author_email='sangone@snu.ac.kr, changyunson@snu.ac.kr',
      url='https://github.com/hapu9734/openmm_saptml',
      license='MIT',
      packages=['saptml'],
      package_data={'saptml': ['data/*']},
      ext_modules=[extension],
      python_requires='>=3.8',
     )
