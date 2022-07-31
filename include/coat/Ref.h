#ifndef COAT_REF_H_
#define COAT_REF_H_


namespace coat {

template<typename T>
struct Ref;

} // namespace

#ifdef ENABLE_ASMJIT
#  include "asmjit/Ref.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/Ref.h"
#endif


#endif
