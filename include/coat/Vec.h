#ifndef COAT_VECTOR_H_
#define COAT_VECTOR_H_

#include <asmjit/asmjit.h>

namespace coat {

template<typename T, unsigned width>
struct Vec;

template<unsigned width>
struct VectorMask;

} // namespace

#include "asmjit/Vec.h"

#endif
