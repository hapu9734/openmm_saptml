set(WRAP_FILE SAPTMLPluginWrapper.cpp)
set(MODULE_NAME saptmlplugin)

# Copy the saptml package into the build tree and run SWIG, which writes the Python half of the module (saptmlplugin.py)
# into the package and the C++ half (SAPTMLPluginWrapper.cpp) next to setup.py.

file(GLOB SAPTML_PY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/saptml/*.py" "${CMAKE_CURRENT_SOURCE_DIR}/saptml/data/*")
add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/saptml" "${CMAKE_CURRENT_BINARY_DIR}/saptml"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -outdir "${CMAKE_CURRENT_BINARY_DIR}/saptml"
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/saptmlplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/saptmlplugin.i" ${SAPTML_PY_FILES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# make PythonInstall: build the extension and install the saptml package into the Python that runs OpenMM
# (run it after make install: the extension links to <prefix>/lib/libSAPTMLPlugin).

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(SAPTML_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(SAPTML_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
add_custom_command(TARGET PythonInstall POST_BUILD
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install . --no-deps --no-build-isolation --force-reinstall
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
