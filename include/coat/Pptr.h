#ifndef COAT_PPTR_H
#define COAT_PPTR_H

namespace coat {

// Pointer to a pointer
template<class CC, typename T>
struct Pptr;

} // namespace

#ifdef ENABLE_ASMJIT
#  include "asmjit/Pptr.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/Pptr.h"
#endif


#endif //COAT_PPTR_H
