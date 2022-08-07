#ifndef COAT_CONDITION_H_
#define COAT_CONDITION_H_

#include "Global.h"

namespace coat {

enum class ConditionFlag {
    e, ne,
    //z, nz,
    l, le, b, be,
    g, ge, a, ae,
    // float
    e_f, ne_f,
    l_f, le_f,
    g_f, ge_f,
};

struct Condition;
} // namespace

#ifdef ENABLE_ASMJIT
#  include "asmjit/Condition.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/Condition.h"
#endif


#endif
