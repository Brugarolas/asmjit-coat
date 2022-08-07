#ifndef COAT_LABEL_H_
#define COAT_LABEL_H_

#include "Global.h"

namespace coat {

struct Label;

} // namespace

#ifdef ENABLE_ASMJIT
#  include "asmjit/Label.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/Label.h"
#endif


#endif
