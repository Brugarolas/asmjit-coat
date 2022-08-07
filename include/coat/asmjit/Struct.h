#ifndef COAT_ASMJIT_STRUCT_H_
#define COAT_ASMJIT_STRUCT_H_

#include <asmjit/asmjit.h>

#include "../constexpr_helper.h"
#include "coat/Function.h"
#include "Ref.h"


namespace coat {


template<typename T>
struct Struct
    : public std::conditional_t<has_custom_base<T>::value,
                                StructBase<Struct<T>>,
                                StructBaseEmpty
            >
{
    using struct_type = T;

    static_assert(std::is_standard_layout_v<T>, "wrapped class needs to have standard layout");

    ::asmjit::x86::Gp reg;
    size_t offset=0;

    Struct(const char *name="") {
        reg = _CC.newIntPtr(name);
    }
#ifdef PROFILING_SOURCE
    Struct(T *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Struct(name) {
        *this = D<T*>{val, file, line};
    }
    Struct(const T *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Struct(name) {
        *this = D<T*>{const_cast<T*>(val), file, line};
    }
#else
    Struct(T *val, const char *name="") : Struct(name) {
        *this = val;
    }
    Struct(const T *val, const char *name="") : Struct(name) {
        *this = const_cast<T*>(val);
    }
#endif

    operator const ::asmjit::x86::Gp&() const { return reg; }
    operator       ::asmjit::x86::Gp&()       { return reg; }

    // load base pointer
    Struct &operator=(const D<T*> &other){
        _CC.mov(reg, ::asmjit::imm(OP));
        DL;
        offset = 0;
        return *this;
    }

    // pre-increment
    Struct &operator++(){ _CC.add(reg, sizeof(T)); return *this; }
    Struct operator[](int amount) {
        Struct res;
        res.reg = reg; // pass ptr register
        res.offset = amount*sizeof(T) + offset; // change offset
        return res;
    }

    Struct operator+ (int amount) const {
        Struct res;
        res.reg = reg; // pass ptr register
        res.offset = amount*sizeof(T) + offset; // change offset
        return res;
    }

    template<int I>
    Ref<reg_type<std::tuple_element_t<I, typename T::types>>> get_reference(){
        static_assert(sizeof(std::tuple_element_t<I, typename T::types>) <= 8, "data length must less than 8");
        switch(sizeof(std::tuple_element_t<I, typename T::types>)){
            case 1: return { ::asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))};
            case 2: return { ::asmjit::x86:: word_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))};
            case 4: return { ::asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))};
            case 8: return { ::asmjit::x86::qword_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))};
        }
        assert(false);
        return { ::asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I, typename T::types> + offset))};
    }

    template<int I>
    wrapper_type<std::tuple_element_t<I, typename T::types>> get_value(
        const char *name=""
#ifdef PROFILING_SOURCE
        ,const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
    ) const {
        using type = std::tuple_element_t<I, typename T::types>;
        wrapper_type<type> ret(name);
        if constexpr(std::is_floating_point_v<type>) {
            _CC.movss(ret.reg, ::asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I,typename T::types> + offset)));
        } else if constexpr(std::is_array_v<type>) {
            // array decay to pointer, just add offset to struct pointer
            //TODO: could just use struct pointer with fixed offset, no need for new register, similar to nested struct
            //_CC.lea(ret.reg, ::asmjit::x86::ptr(reg, offset_of_v<I,typename T::types> + offset));
            ret.reg = reg; // pass ptr register
            ret.offset = offset + offset_of_v<I,typename T::types>; // change offset

#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        }else if constexpr(std::is_arithmetic_v<std::remove_pointer_t<type>>){
#if 0
            //FIXME: VRegMem not defined for pointer types currently
            ret = get_reference<I>();
#else
            switch(sizeof(type)){
                case 1: _CC.mov(ret.reg, ::asmjit::x86:: byte_ptr(reg, (int32_t)(offset_of_v<I,typename T::types> + offset))); break;
                case 2: _CC.mov(ret.reg, ::asmjit::x86:: word_ptr(reg, (int32_t)(offset_of_v<I,typename T::types> + offset))); break;
                case 4: _CC.mov(ret.reg, ::asmjit::x86::dword_ptr(reg, (int32_t)(offset_of_v<I,typename T::types> + offset))); break;
                case 8: _CC.mov(ret.reg, ::asmjit::x86::qword_ptr(reg, (int32_t)(offset_of_v<I,typename T::types> + offset))); break;
            }
#endif
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        }else if constexpr(std::is_pointer_v<type>){
            // pointer to struct, load pointer
            _CC.mov(ret.reg, ::asmjit::x86::qword_ptr(reg, offset_of_v<I,typename T::types> + offset));
#ifdef PROFILING_SOURCE
            ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        }else{
            // nested struct
            ret.reg = reg; // pass ptr register
            ret.offset = offset + offset_of_v<I,typename T::types>; // change offset
        }
        return ret;
    }
};


} // namespace


#endif
