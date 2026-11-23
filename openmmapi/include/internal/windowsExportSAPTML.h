#ifndef OPENMM_WINDOWSEXPORTSAPTML_H_
#define OPENMM_WINDOWSEXPORTSAPTML_H_

/*
 * OPENMM_EXPORT_SAPTML on Visual Studio: dllexport when building the shared library (SAPTML_BUILDING_SHARED_LIBRARY),
 * nothing for a static build (SAPTML_BUILDING_STATIC_LIBRARY) or a client linking statically
 * (SAPTML_USE_STATIC_LIBRARIES), dllimport otherwise.
 */

#ifdef _MSC_VER
    // Silence the warnings about "unsafe" sprintf and about private members without dll export.
    #pragma warning(disable:4996)
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
