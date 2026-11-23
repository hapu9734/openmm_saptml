#ifndef OPENMM_REFERENCESAPTMLKERNELFACTORY_H_
#define OPENMM_REFERENCESAPTMLKERNELFACTORY_H_

/* OpenMM SAPT-ML plugin.
 * Copyright (c) 2025-2026 Sangwon Bae and Chang Yun Son, Seoul National University.
 * MIT License; see LICENSE.
 */

#include "openmm/KernelFactory.h"

namespace OpenMM {

class ReferenceSAPTMLKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCESAPTMLKERNELFACTORY_H_*/
