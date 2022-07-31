#ifndef COAT_FUNCTION_H_
#define COAT_FUNCTION_H_

#include <type_traits>


namespace coat {

//FIXME: forwards
template<typename T> struct Value;
template<typename T> struct Ptr;
template<typename T> struct Struct;

template<typename T>
using reg_type = std::conditional_t<std::is_pointer_v<T>,
						Ptr<Value<std::remove_cv_t<std::remove_pointer_t<T>>>>,
						Value<std::remove_cv_t<T>>
				>;

// decay - converts array types to pointer types
template<typename T>
using wrapper_type = std::conditional_t<std::is_arithmetic_v<std::remove_pointer_t<std::decay_t<T>>>,
						reg_type<std::decay_t<T>>,
						Struct<std::remove_cv_t<std::remove_pointer_t<T>>>
					>;

template<typename T>
struct Function;

template<typename T>
struct InternalFunction;

} // namespace


#ifdef ENABLE_ASMJIT
#  include "asmjit/Function.h"
#endif

#ifdef ENABLE_LLVMJIT
#  include "llvmjit/Function.h"
#endif


#include "Condition.h"
#include "Ref.h"
#include "Value.h"
#include "Ptr.h"
#include "Struct.h"

#endif
