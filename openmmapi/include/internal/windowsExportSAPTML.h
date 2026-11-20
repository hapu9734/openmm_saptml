#ifndef OPENMM_WINDOWSEXPORTSAPTML_H_
#define OPENMM_WINDOWSEXPORTSAPTML_H_

/*
 * Shared libraries are messy in Visual Studio. We have to distinguish three
 * cases:
 *   (1) this header is being used to build the SAPTMLPlugin shared library
 *       (dllexport)
 *   (2) this header is being used by a *client* of the SAPTMLPlugin shared
 *       library (dllimport)
 *   (3) we are building the SAPTMLPlugin static library, or the client is
 *       being compiled with the expectation of linking with the
 *       SAPTMLPlugin static library (nothing special needed)
 * In the CMake script for building this library, we define one of the symbols
 *     SAPTML_BUILDING_SHARED_LIBRARY or
 *     SAPTML_BUILDING_STATIC_LIBRARY
 * Client code normally has no special symbol defined, in which case we'll
 * assume it wants to use the shared library. However, if the client defines
 * the symbol SAPTML_USE_STATIC_LIBRARIES we'll suppress the dllimport so
 * that the client code can be linked with static libraries.
 */

#ifdef _MSC_VER
    // We don't want to hear about how sprintf is "unsafe".
    #pragma warning(disable:4996)
    // Keep MS VC++ quiet about lack of dll export of private members.
    #pragma warning(disable:4251)
    #if defined(SAPTML_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_SAPTML __declspec(dllexport)
    #elif defined(SAPTML_BUILDING_STATIC_LIBRARY) || defined(SAPTML_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_SAPTML
    #else
        #define OPENMM_EXPORT_SAPTML __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_SAPTML __attribute__((visibility("default")))
#endif

#endif // OPENMM_WINDOWSEXPORTSAPTML_H_
