#ifndef COAT_ASMJIT_PPTR_H
#define COAT_ASMJIT_PPTR_H

#include <asmjit/asmjit.h>

namespace coat {

template<class T>
// TODO Ideally, there would be only the Ptr class which can also hold pointer to pointer
struct Pptr<::asmjit::x86::Compiler,T>{
    using F = ::asmjit::x86::Compiler;
    using value_type = std::remove_pointer_t<T>;
    using value_base_type = ValueBase<F>;
    // TODO *pptr2 = ptr1 should result in **pptr2 = *ptr1
    //  A solution can be adding an specific Ref for pointer type.
    //  This is necessary because pointers don't support <, <=, >, >=
    using mem_type = std::conditional_t<std::is_pointer_v<value_type>,
                Pptr<F, value_type>, Ref<F,Value<F, value_type>>
            >;

    // Assert that T is a pointer
    // TODO assert that is pointer to integer
    static_assert(std::is_pointer<T>::value, "Only pointer types supported");

    ::asmjit::x86::Compiler &cc; //FIXME: pointer stored in every value type
    ::asmjit::x86::Gp reg;

    // TODO why are names used?
    Pptr(F &cc, const char *name="") : cc(cc) {
        // TODO why IntPtr and not UIntPtr?
        reg = cc.newIntPtr(name);
    }

#ifdef PROFILING_SOURCE
    Pptr(F &cc, value_type *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Ptr(cc, name) {
        *this = D<value_type*>{val, file, line};
    }
    Pptr(F &cc, const value_type *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Ptr(cc, name) {
        *this = D<value_type*>{const_cast<value_type*>(val), file, line};
    }
#else
    Pptr(F &cc, value_type *val, const char *name="") : Pptr(cc, name) {
        *this = val;
    }
    Pptr(F &cc, const value_type *val, const char *name="") : Pptr(cc, name) {
        *this = const_cast<value_type*>(val);
    }
    Pptr(F &cc, ::asmjit::x86::Mem mem) : Pptr(cc) {
        cc.mov(reg, mem);
    }
#endif

    // real copy requires new register and copy of content
    Pptr(const Pptr &other) : Pptr(other.cc) {
        *this = other;
    }
    // move, just take the register
    Pptr(const Pptr &&other) : cc(other.cc), reg(other.reg) {}

    // assignment
    Pptr &operator=(const D<value_type*> &other){
        cc.mov(reg, ::asmjit::imm(OP));
        DL;
        return *this;
    }
    Pptr &operator=(const Pptr &other){
        cc.mov(reg, other.reg);
        return *this;
    }

    operator const ::asmjit::x86::Gp&() const { return reg; }
    operator       ::asmjit::x86::Gp&()       { return reg; }

    // dereference
    mem_type operator*(){
        switch(sizeof(value_type)){
            case 1: return {cc, ::asmjit::x86::byte_ptr (reg)};
            case 2: return {cc, ::asmjit::x86::word_ptr (reg)};
            case 4: return {cc, ::asmjit::x86::dword_ptr(reg)};
            case 8: return {cc, ::asmjit::x86::qword_ptr(reg)};
        }
    }
    // indexing with variable
    mem_type operator[](const value_base_type &idx){
        switch(sizeof(value_type)){
            case 1: return {cc, ::asmjit::x86::byte_ptr (reg, idx.reg, clog2(sizeof(value_type)))};
            case 2: return {cc, ::asmjit::x86::word_ptr (reg, idx.reg, clog2(sizeof(value_type)))};
            case 4: return {cc, ::asmjit::x86::dword_ptr(reg, idx.reg, clog2(sizeof(value_type)))};
            case 8: return {cc, ::asmjit::x86::qword_ptr(reg, idx.reg, clog2(sizeof(value_type)))};
        }
    }
    // indexing with constant -> use offset
    mem_type operator[](int idx){
        switch(sizeof(value_type)){
            case 1: return {cc, ::asmjit::x86::byte_ptr (reg, idx*sizeof(value_type))};
            case 2: return {cc, ::asmjit::x86::word_ptr (reg, idx*sizeof(value_type))};
            case 4: return {cc, ::asmjit::x86::dword_ptr(reg, idx*sizeof(value_type))};
            case 8: return {cc, ::asmjit::x86::qword_ptr(reg, idx*sizeof(value_type))};
        }
    }
    // get memory operand with displacement
    mem_type byteOffset(long offset){
        switch(sizeof(value_type)){
            case 1: return {cc, ::asmjit::x86::byte_ptr (reg, offset)};
            case 2: return {cc, ::asmjit::x86::word_ptr (reg, offset)};
            case 4: return {cc, ::asmjit::x86::dword_ptr(reg, offset)};
            case 8: return {cc, ::asmjit::x86::qword_ptr(reg, offset)};
        }
    }

    Pptr operator+(const D<value_base_type> &other) const {
        Pptr res(cc);
        cc.lea(res, ::asmjit::x86::ptr(reg, OP, clog2(sizeof(value_type))));
        DL;
        return res;
    }
    Pptr operator+(size_t value) const {
        Pptr res(cc);
        cc.lea(res, ::asmjit::x86::ptr(reg, value*sizeof(value_type)));
        return res;
    }

    Pptr &operator+=(const value_base_type &value){
        cc.lea(reg, ::asmjit::x86::ptr(reg, value.reg, clog2(sizeof(value_type))));
        return *this;
    }
    Pptr &operator+=(const D<int> &other){ cc.add(reg, OP*sizeof(value_type)); DL; return *this; }
    Pptr &operator-=(int amount){ cc.sub(reg, amount*sizeof(value_type)); return *this; }

    // like "+=" without pointer arithmetic
    Pptr &addByteOffset(const value_base_type &value){ //TODO: any integer value should be possible as operand
        cc.lea(reg, ::asmjit::x86::ptr(reg, value.reg));
        return *this;
    }

    // operators creating temporary virtual registers
    Value<F,size_t> operator- (const Pptr &other) const {
        Value<F,size_t> ret(cc, "ret");
        cc.mov(ret, reg);
        cc.sub(ret, other.reg);
        cc.sar(ret, clog2(sizeof(value_type))); // compilers also do arithmetic shift...
        return ret;
    }

    // pre-increment, post-increment not provided as it creates temporary
    Pptr &operator++(){ cc.add(reg, sizeof(value_type)); return *this; }
    // pre-decrement
    Pptr &operator--(){ cc.sub(reg, sizeof(value_type)); return *this; }

    // comparisons
    Condition<F> operator==(const Pptr &other) const { return {cc, reg, other.reg, ConditionFlag::e};  }
    Condition<F> operator!=(const Pptr &other) const { return {cc, reg, other.reg, ConditionFlag::ne}; }
};


template<typename dest_type, typename src_type>
Ptr<::asmjit::x86::Compiler, dest_type>
cast(const Ptr<::asmjit::x86::Compiler, src_type> &src){
    static_assert(std::is_pointer_v<dest_type>, "a pointer type can only be casted to another pointer type");

    //TODO: find a way to do it without copies but no surprises for user
    // create new pointer with new register
    Ptr<::asmjit::x86::Compiler, dest_type> res(src.cc);
    // copy pointer address between registers
    src.cc.mov(res.reg, src.reg);
    // return new pointer
    return res;
}

} // namespace

#endif //COAT_ASMJIT_PPTR_H
