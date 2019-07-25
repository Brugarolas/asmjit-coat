#ifndef VALUE_H_
#define VALUE_H_


template<class CC>
struct ValueBase;

template<class CC, typename T>
struct Value;

#ifdef ENABLE_ASMJIT
#  include "asmjit/ValueBase.h"
#  include "asmjit/Value.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/ValueBase.h"
#  include "llvmjit/Value.h"
#endif


#endif
