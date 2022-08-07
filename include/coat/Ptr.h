#pragma once

#include "Global.h"
#include "ValueBase.h"
#include "constexpr_helper.h"

namespace coat {

template<class T>
struct Ptr {
    using value_type = typename T::value_type;
    using value_base_type = ValueBase;
    using mem_type = Ref<T>;

    asmjit::x86::Gp reg;

    Ptr(const char *name="") {
        reg = _CC.newIntPtr(name);
    }
    Ptr(asmjit::x86::Gp reg) : reg(reg) {
    }
#ifdef PROFILING_SOURCE
    Ptr(value_type *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Ptr(name) {
        *this = D<value_type*>{val, file, line};
    }
    Ptr(const value_type *val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Ptr(name) {
        *this = D<value_type*>{const_cast<value_type*>(val), file, line};
    }
#else
    Ptr(value_type *val, const char *name="") : Ptr(name) {
        *this = val;
    }
    Ptr(const value_type *val, const char *name="") : Ptr(name) {
        *this = const_cast<value_type*>(val);
    }
#endif

    // real copy requires new register and copy of content
    Ptr(const Ptr &other) : Ptr() {
        *this = other;
    }
    // move, just take the register
    Ptr(const Ptr &&other) : reg(other.reg) {}

    // assignment
    Ptr &operator=(const D<value_type*> &other){
        _CC.mov(reg, asmjit::imm(OP));
        DL;
        return *this;
    }
    Ptr &operator=(const Ptr &other){
        _CC.mov(reg, other.reg);
        return *this;
    }

    void setName(const char * /*name*/){
        //TODO
    }

    operator const asmjit::x86::Gp&() const { return reg; }
    operator       asmjit::x86::Gp&()       { return reg; }

    // dereference
    mem_type operator*(){
        switch(sizeof(value_type)){
            case 1: return {asmjit::x86::byte_ptr (reg)};
            case 2: return {asmjit::x86::word_ptr (reg)};
            case 4: return {asmjit::x86::dword_ptr(reg)};
            case 8: return {asmjit::x86::qword_ptr(reg)};
        }
        assert(false);
        return {asmjit::x86::byte_ptr (reg)};
    }
    // indexing with variable
    mem_type operator[](const value_base_type &idx){
        switch(sizeof(value_type)){
            case 1: return {asmjit::x86::byte_ptr (reg, idx.reg, clog2(sizeof(value_type)))};
            case 2: return {asmjit::x86::word_ptr (reg, idx.reg, clog2(sizeof(value_type)))};
            case 4: return {asmjit::x86::dword_ptr(reg, idx.reg, clog2(sizeof(value_type)))};
            case 8: return {asmjit::x86::qword_ptr(reg, idx.reg, clog2(sizeof(value_type)))};
        }
        assert(false);
        return { asmjit::x86::byte_ptr(reg, idx.reg) };
    }
    // indexing with constant -> use offset
    mem_type operator[](int idx){
        switch(sizeof(value_type)){
            case 1: return {asmjit::x86::byte_ptr (reg, idx*sizeof(value_type))};
            case 2: return {asmjit::x86::word_ptr (reg, idx*sizeof(value_type))};
            case 4: return {asmjit::x86::dword_ptr(reg, idx*sizeof(value_type))};
            case 8: return {asmjit::x86::qword_ptr(reg, idx*sizeof(value_type))};
        }
        assert(false);
        return { asmjit::x86::byte_ptr(reg, idx) };
    }
    // get memory operand with displacement
    mem_type byteOffset(long offset){
        switch(sizeof(value_type)){
            case 1: return {asmjit::x86::byte_ptr (reg, offset)};
            case 2: return {asmjit::x86::word_ptr (reg, offset)};
            case 4: return {asmjit::x86::dword_ptr(reg, offset)};
            case 8: return {asmjit::x86::qword_ptr(reg, offset)};
        }
        assert(false);
        return { asmjit::x86::byte_ptr(reg, offset) };
    }

    Ptr operator+(const D<value_base_type> &other) const {
        Ptr res;
        _CC.lea(res, asmjit::x86::ptr(reg, OP, clog2(sizeof(value_type))));
        DL;
        return res;
    }
    Ptr operator+(size_t value) const {
        Ptr res;
        _CC.lea(res, asmjit::x86::ptr(reg, value*sizeof(value_type)));
        return res;
    }

    Ptr &operator+=(const value_base_type &value){
        _CC.lea(reg, asmjit::x86::ptr(reg, value.reg, clog2(sizeof(value_type))));
        return *this;
    }
    Ptr &operator+=(const D<int> &other){ _CC.add(reg, OP*sizeof(value_type)); DL; return *this; }
    Ptr &operator-=(int amount){ _CC.sub(reg, amount*sizeof(value_type)); return *this; }

    // like "+=" without pointer arithmetic
    Ptr &addByteOffset(const value_base_type &value){ //TODO: any integer value should be possible as operand
        _CC.lea(reg, asmjit::x86::ptr(reg, value.reg));
        return *this;
    }

    // operators creating temporary virtual registers
    Value<size_t> operator- (const Ptr &other) const {
        Value<size_t> ret("ret");
        _CC.mov(ret, reg);
        _CC.sub(ret, other.reg);
        _CC.sar(ret, clog2(sizeof(value_type))); // compilers also do arithmetic shift...
        return ret;
    }

    // pre-increment, post-increment not provided as it creates temporary
    Ptr &operator++(){ _CC.add(reg, sizeof(value_type)); return *this; }
    // pre-decrement
    Ptr &operator--(){ _CC.sub(reg, sizeof(value_type)); return *this; }

    // comparisons
    Condition operator==(const Ptr &other) const { return {reg, other.reg, ConditionFlag::e};  }
    Condition operator!=(const Ptr &other) const { return {reg, other.reg, ConditionFlag::ne}; }

    // cast to any pointer type
    template<typename dest_type>
    Ptr<Value<dest_type>>
    cast() {
        Ptr<Value<dest_type>> res(reg);
        return res;
    }
    // index: [base.reg + (idx << shift) + offset]
    mem_type index(const value_base_type &idx, int offset){
        return {asmjit::x86::ptr (reg, idx.reg, clog2(sizeof(value_type)), offset)};
    }
    // index: [base.reg + (idx << shift) + offset], scale = 1 << shift
    mem_type index(const value_base_type &idx, int scale, int offset){
        return {asmjit::x86::ptr (reg, idx.reg, clog2(scale), offset)};
    }    
};


template<typename dest_type, typename src_type>
Ptr<Value<std::remove_pointer_t<dest_type>>>
cast(const Ptr<Value<src_type>> &src){
    static_assert(std::is_pointer_v<dest_type>, "a pointer type can only be casted to another pointer type");

    //TODO: find a way to do it without copies but no surprises for user
    // create new pointer with new register
    Ptr<Value<std::remove_pointer_t<dest_type>>> res;
    // copy pointer address between registers
    _CC.mov(res.reg, src.reg);
    // return new pointer
    return res;
}

} // namespace
