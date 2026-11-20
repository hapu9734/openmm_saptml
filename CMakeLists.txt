#---------------------------------------------------
# OpenMM SAPT-ML Plugin
#
# Creates the SAPTMLPlugin library and its Reference and CUDA platform implementations, the C++ tests and the Python
# wrappers.  Requires an installed OpenMM (OPENMM_DIR).
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
PROJECT(SAPTMLPlugin LANGUAGES CXX)

SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
FIND_PATH(OPENMM_INCLUDE_DIR openmm/Context.h PATHS "${OPENMM_DIR}/include" NO_DEFAULT_PATH)
FIND_LIBRARY(OPENMM_LIBRARY NAMES OpenMM PATHS "${OPENMM_DIR}/lib" NO_DEFAULT_PATH)
IF(NOT OPENMM_INCLUDE_DIR OR NOT OPENMM_LIBRARY)
    MESSAGE(FATAL_ERROR "OpenMM was not found below OPENMM_DIR=${OPENMM_DIR}")
ENDIF()
INCLUDE_DIRECTORIES("${OPENMM_INCLUDE_DIR}")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# Specify the C++ version we are building for.
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set flags for linking on mac
IF(APPLE)
    SET(CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    SET(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH "Where to install the plugin" FORCE)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# Put all the tests and libraries in a single output directory.
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL "Single output directory for building all executables.")
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR} CACHE INTERNAL "Single output directory for building all libraries.")

# The source is organized into subdirectories, but we handle them all from this CMakeLists file rather than letting
# each subdirectory make its own CMakeLists.
SET(SAPTML_PLUGIN_SOURCE_SUBDIRS openmmapi)
SET(SAPTML_LIBRARY_NAME SAPTMLPlugin)
SET(SHARED_SAPTML_TARGET ${SAPTML_LIBRARY_NAME})
SET(SAPTML_REFERENCE_LIBRARY_NAME SAPTMLPluginReference)
SET(SAPTML_CUDA_LIBRARY_NAME SAPTMLPluginCUDA)

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files.
SET(SOURCE_FILES)
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${SAPTML_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.
ADD_LIBRARY(${SHARED_SAPTML_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_SAPTML_TARGET}
    PROPERTIES COMPILE_FLAGS "-DSAPTML_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_SAPTML_TARGET} ${OPENMM_LIBRARY})
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_SAPTML_TARGET})

# Install headers.
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing.
ENABLE_TESTING()
ADD_SUBDIRECTORY(platforms/reference)
ADD_SUBDIRECTORY(platforms/common)

# Build the CUDA platform when the toolkit and the OpenMM CUDA platform are found.
FIND_PACKAGE(CUDAToolkit QUIET)
FIND_LIBRARY(OPENMM_CUDA_LIBRARY NAMES OpenMMCUDA PATHS "${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins" NO_DEFAULT_PATH)
IF(CUDAToolkit_FOUND AND OPENMM_CUDA_LIBRARY)
    SET(SAPTML_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
ELSE()
    SET(SAPTML_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
ENDIF()
IF(SAPTML_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(SAPTML_BUILD_CUDA_LIB)

# Build the Python wrappers.
FIND_PROGRAM(PYTHON_EXECUTABLE NAMES python3 python)
FIND_PROGRAM(SWIG_EXECUTABLE NAMES swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(SAPTML_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE()
    SET(SAPTML_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF()
IF(SAPTML_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(SAPTML_BUILD_PYTHON_WRAPPERS)
