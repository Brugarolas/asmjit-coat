#pragma once

#include "Global.h"
#include <type_traits>

#include "constexpr_helper.h"
#include "DebugOperand.h"
#include "operator_helper.h"
#include "ValueBase.h"
#include "Label.h"

namespace coat {

template<typename T>
struct Value final : public ValueBase {
    using value_type = T;

    static_assert(sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4 || sizeof(T)==8,
        "only plain arithmetic types supported of sizes: 1, 2, 4 or 8 bytes");
    static_assert(std::is_signed_v<T> || std::is_unsigned_v<T>,
        "only plain signed or unsigned arithmetic types supported");

    Value(const char *name="") : ValueBase() {
        if constexpr(std::is_signed_v<T>){
            switch(sizeof(T)){
                case 1: reg = _CC.newInt8 (name); break;
                case 2: reg = _CC.newInt16(name); break;
                case 4: reg = _CC.newInt32(name); break;
                case 8: reg = _CC.newInt64(name); break;
            }
        }else{
            switch(sizeof(T)){
                case 1: reg = _CC.newUInt8 (name); break;
                case 2: reg = _CC.newUInt16(name); break;
                case 4: reg = _CC.newUInt32(name); break;
                case 8: reg = _CC.newUInt64(name); break;
            }
        }
    }
    Value(asmjit::x86::Gp reg) : ValueBase(reg) {}
#ifdef PROFILING_SOURCE
    Value(T val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Value(name) {
        *this = D<T>{val, file, line};
    }
#else
    Value(T val, const char *name="") : Value(name) {
        *this = val;
    }
#endif

    // real copy means new register and copy content
    Value(const D<Value> &other) : Value() {
        *this = other;
    }
    // move ctor
    Value(const Value &&other) : ValueBase(std::move(other)) {}
    // move assign
    Value &operator=(const Value &&other){
        reg = other.reg; // just take virtual register
        return *this;
    }

    // copy ctor for ref, basically loads value from memory and stores in register
    Value(const D<Ref<Value>> &other) : Value() {
        *this = other;
    }

    // explicit type conversion, assignment
    // always makes a copy
    // FIXME: implicit conversion between signed and unsigned, but not between types of same size ...
    template<class O>
    Value &narrow(const O &other){
        if constexpr(std::is_base_of_v<ValueBase,O>){
            static_assert(sizeof(T) < sizeof(typename O::value_type), "narrowing conversion called on wrong types");
            // copy from register
            switch(sizeof(T)){
                case 1: _CC.mov(reg, other.reg.r8 ()); break;
                case 2: _CC.mov(reg, other.reg.r16()); break;
                case 4: _CC.mov(reg, other.reg.r32()); break;
            }
        }else if(std::is_base_of_v<ValueBase,typename O::inner_type>){
            static_assert(sizeof(T) < sizeof(typename O::inner_type::value_type), "narrowing conversion called on wrong types");
            // memory operand
            switch(sizeof(T)){
                case 1: _CC.mov(reg, other); break;
                case 2: _CC.mov(reg, other); break;
                case 4: _CC.mov(reg, other); break;
            }
        }else{
            static_assert(should_not_be_reached<O>, "assignment only allowed for wrapped register types");
        }
        return *this;
    }
    template<class O>
    Value &widen(const O &other
#ifdef PROFILING_SOURCE
        , const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
    ){
        size_t other_size;
        if constexpr(std::is_base_of_v<ValueBase,O>){
            static_assert(sizeof(T) > sizeof(typename O::value_type), "widening conversion called on wrong types");
            other_size = sizeof(typename O::value_type);
        }else/* if(std::is_base_of_v<ValueBase,typename O::inner_type>)*/{ //FIXME
            static_assert(sizeof(T) > sizeof(typename O::inner_type::value_type), "widening conversion called on wrong types");
            other_size = sizeof(typename O::inner_type::value_type);
        }/*else{
            static_assert(should_not_be_reached<O>, "assignment only allowed for wrapped register types");
        }*/
        // syntax the same for register and memory operand, implicit conversion to reg or mem
        if constexpr(std::is_signed_v<T>){
            if(sizeof(T)==8 && other_size==4){
                _CC.movsxd(reg, other);
            }else{
                _CC.movsx(reg, other);
            }
        }else{
            // movzx only from 8/16 to whatever, otherwise use normal move as it implicitly zero's upper 32-bit
            if(other_size <= 2){
                _CC.movzx(reg, other);
            }else{
                // write to 32-bit register, upper half zero'd
                _CC.mov(reg.r32(), other);
            }
        }
#ifdef PROFILING_SOURCE
        ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
        return *this;
    }

    // explicit conversion, no assignment, new value/temporary
    //TODO: decide on one style and delete other
    template<typename O>
    Value<O> narrow(){
        static_assert(sizeof(O) < sizeof(T), "narrowing conversion called on wrong types");
        Value<O> tmp("narrowtmp");
        // copy from register
        switch(sizeof(O)){
            case 1: _CC.mov(tmp.reg, reg.r8 ()); break;
            case 2: _CC.mov(tmp.reg, reg.r16()); break;
            case 4: _CC.mov(tmp.reg, reg.r32()); break;
        }
        return tmp;
    }
    template<typename O>
    Value<O> widen(){
        static_assert(sizeof(O) > sizeof(T), "widening conversion called on wrong types");
        Value<O> tmp("widentmp");
        if constexpr(std::is_signed_v<T>){
            if(sizeof(O)==8 && sizeof(T)==4){
                _CC.movsxd(tmp.reg, reg);
            }else{
                _CC.movsx(tmp.reg, reg);
            }
        }else{
            // movzx only from 8/16 to whatever, otherwise use normal move as it implicitly zero's upper 32-bit
            if(sizeof(T) <= 2){
                _CC.movzx(tmp.reg, reg);
            }else{
                // write to 32-bit register, upper half zero'd
                _CC.mov(tmp.reg.r32(), reg);
            }
        }
        return tmp;
    }

    // assignment
    Value &operator=(const D<Value> &other){
        _CC.mov(reg, OP);
        DL;
        return *this;
    }
    Value &operator=(const D<value_type> &other){
        if(OP == 0){
            // just for fun, smaller opcode, writing 32-bit part zero's rest of 64-bit register too
            _CC.xor_(reg.r32(), reg.r32());
            DL;
        }else{
            _CC.mov(reg, asmjit::imm(OP));
            DL;
        }
        return *this;
    }
    Value &operator=(const D<Ref<Value>> &other){
        _CC.mov(reg, OP);
        DL;
        return *this;
    }

    // special handling of bit tests, for convenience and performance
    void bit_test(const Value &bit, Label &label, bool jump_on_set=true) const {
        _CC.bt(reg, bit);
        if(jump_on_set){
            _CC.jc(label);
        }else{
            _CC.jnc(label);
        }
    }

    void bit_test_and_set(const Value &bit, Label &label, bool jump_on_set=true){
        _CC.bts(reg, bit);
        if(jump_on_set){
            _CC.jc(label);
        }else{
            _CC.jnc(label);
        }
    }

    void bit_test_and_reset(const Value &bit, Label &label, bool jump_on_set=true){
        _CC.btr(reg, bit);
        if(jump_on_set){
            _CC.jc(label);
        }else{
            _CC.jnc(label);
        }
    }

    Value popcount() const {
        Value ret;
        _CC.popcnt(ret.reg, reg);
        return ret;
    }

    // operators with assignment
    Value &operator<<=(const D<Value> &other){
        if constexpr(std::is_signed_v<T>){
            _CC.sal(reg, OP);
        }else{
            _CC.shl(reg, OP);
        }
        DL;
        return *this;
    }
    Value &operator<<=(const D<int> &other){
        if constexpr(std::is_signed_v<T>){
            _CC.sal(reg, asmjit::imm(OP));
        }else{
            _CC.shl(reg, asmjit::imm(OP));
        }
        DL;
        return *this;
    }
    // memory operand not possible on right side

    Value &operator>>=(const D<Value> &other){
        if constexpr(std::is_signed_v<T>){
            _CC.sar(reg, OP);
        }else{
            _CC.shr(reg, OP);
        }
        DL;
        return *this;
    }
    Value &operator>>=(const D<int> &other){
        if constexpr(std::is_signed_v<T>){
            _CC.sar(reg, asmjit::imm(OP));
        }else{
            _CC.shr(reg, asmjit::imm(OP));
        }
        DL;
        return *this;
    }
    // memory operand not possible on right side

    Value &operator*=(const D<Value> &other){
        static_assert(sizeof(T) > 1, "multiplication of byte type currently not supported");
        if constexpr(std::is_signed_v<T>){
            _CC.imul(reg, OP);
        }else{
            asmjit::x86::Gp r_upper = _CC.newUInt64(); // don't care about type here
            switch(sizeof(T)){
                case 1: /* TODO */ break;
                case 2: _CC.mul(r_upper.r16(), reg, OP); break;
                case 4: _CC.mul(r_upper.r32(), reg, OP); break;
                case 8: _CC.mul(r_upper.r64(), reg, OP); break;
            }
            // ignores upper part for now
        }
        DL;
        return *this;
    }
    // immediate not possible in mul, but imul has support
    Value &operator*=(const D<int> &other){
        //static_assert(sizeof(T) > 1, "multiplication of byte type currently not supported");
        // special handling of stuff which can be done with lea
        switch(OP){
            case  0: _CC.xor_(reg, reg); break;
            case  1: /* do nothing */ break;
            case  2:
                // lea rax, [rax + rax]
                _CC.lea(reg, asmjit::x86::ptr(reg, reg));
                DL;
                break;
            case  3:
                // lea rax, [rax + rax*2]
                _CC.lea(reg, asmjit::x86::ptr(reg, reg, clog2(2)));
                DL;
                break;
            case  4:
                // lea rax, [0 + rax*4]
                _CC.lea(reg, asmjit::x86::ptr(0, reg, clog2(4)));
                DL;
                break;
            case  5:
                // lea rax, [rax + rax*4]
                _CC.lea(reg, asmjit::x86::ptr(reg, reg, clog2(4)));
                DL;
                break;
            case  6:
                // lea rax, [rax + rax*2]
                // add rax, rax
                _CC.lea(reg, asmjit::x86::ptr(reg, reg, clog2(2)));
                DL;
                _CC.add(reg, reg);
                DL;
                break;
            //case  7:
            //    // requires two registers
            //    // lea rbx, [0 + rax*8]
            //    // sub rbx, rax
            //    _CC.lea(reg, asmjit::x86::ptr(0, reg, clog2(8)));
            //    _CC.sub();
            //    break;
            case  8:
                // lea rax, [0 + rax*8]
                _CC.lea(reg, asmjit::x86::ptr(0, reg, clog2(8)));
                DL;
                break;
            case  9:
                // lea rax, [rax + rax*8]
                _CC.lea(reg, asmjit::x86::ptr(reg, reg, clog2(8)));
                DL;
                break;
            case 10:
                // lea rax, [rax + rax*4]
                // add rax, rax
                _CC.lea(reg, asmjit::x86::ptr(reg, reg, clog2(4)));
                DL;
                _CC.add(reg, reg);
                DL;
                break;

            default: {
                if(is_power_of_two(OP)){
                    operator<<=(PASSOP(clog2(OP)));
                }else{
                    if constexpr(std::is_signed_v<T>){
                        _CC.imul(reg, asmjit::imm(OP));
                        DL;
                    }else{
                        Value temp(T(OP), "constant");
                        operator*=(PASSOP(temp));
                    }
                }
            }
        }
        return *this;
    }

    Value &operator/=(const Value &other){
        static_assert(sizeof(T) > 1, "division of byte type currently not supported");
        if constexpr(std::is_signed_v<T>){
            asmjit::x86::Gp r_upper = _CC.newUInt64(); // don't care about type here
            switch(sizeof(T)){
                case 1: /* TODO */ break;
                case 2: _CC.cwd(r_upper.r16(), reg); _CC.idiv(r_upper.r16(), reg, other); break;
                case 4: _CC.cdq(r_upper.r32(), reg); _CC.idiv(r_upper.r32(), reg, other); break;
                case 8: _CC.cqo(r_upper.r64(), reg); _CC.idiv(r_upper.r64(), reg, other); break;
            }
        }else{
            asmjit::x86::Gp r_upper = _CC.newUInt64(); // don't care about type here
            _CC.xor_(r_upper.r32(), r_upper.r32());
            switch(sizeof(T)){
                case 1: /* TODO */ break;
                case 2: _CC.div(r_upper.r16(), reg, other); break;
                case 4: _CC.div(r_upper.r32(), reg, other); break;
                case 8: _CC.div(r_upper.r64(), reg, other); break;
            }
        }
        return *this;
    }
    // immediate not possible in div or idiv
    Value &operator/=(int constant){
        if(is_power_of_two(constant)){
            operator>>=(clog2(constant));
        }else{
            Value temp(T(constant), "constant");
            operator/=(temp);
        }
        return *this;
    }

    Value &operator%=(const Value &other){
        static_assert(sizeof(T) > 1, "division of byte type currently not supported");
        if constexpr(std::is_signed_v<T>){
            asmjit::x86::Gp r_upper;
            switch(sizeof(T)){
                case 1: /* TODO */ break;
                case 2: r_upper = _CC.newInt16(); _CC.cwd(r_upper, reg); break;
                case 4: r_upper = _CC.newInt32(); _CC.cdq(r_upper, reg); break;
                case 8: r_upper = _CC.newInt64(); _CC.cqo(r_upper, reg); break;
            }
            _CC.idiv(r_upper, reg, other);
            // remainder in upper part, use this register from now on
            reg = r_upper;
        }else{
            asmjit::x86::Gp r_upper;
            switch(sizeof(T)){
                case 1: /* TODO */ break;
                case 2: r_upper = _CC.newUInt16(); break;
                case 4: r_upper = _CC.newUInt32(); break;
                case 8: r_upper = _CC.newUInt64(); break;
            }
            _CC.xor_(r_upper.r32(), r_upper.r32());
            _CC.div(r_upper, reg, other);
            // remainder in upper part, use this register from now on
            reg = r_upper;
        }
        return *this;
    }
    // immediate not possible in div or idiv
    Value &operator%=(int constant){
        if(is_power_of_two(constant)){
            operator&=(constant - 1);
        }else{
            Value temp(T(constant), "constant");
            operator%=(temp);
        }
        return *this;
    }

    Value &operator+=(const D<int>          &other){ _CC.add(reg, asmjit::imm(OP)); DL; return *this; }
    Value &operator+=(const D<Value>        &other){ _CC.add(reg, OP);                DL; return *this; }
    Value &operator+=(const D<Ref<Value>> &other){ _CC.add(reg, OP);                DL; return *this; }

    Value &operator-=(const D<int>          &other){ _CC.sub(reg, asmjit::imm(OP)); DL; return *this; }
    Value &operator-=(const D<Value>        &other){ _CC.sub(reg, OP);                DL; return *this; }
    Value &operator-=(const D<Ref<Value>> &other){ _CC.sub(reg, OP);                DL; return *this; }

    Value &operator&=(const D<int>          &other){ _CC.and_(reg, asmjit::imm(OP)); DL; return *this; }
    Value &operator&=(const D<Value>        &other){ _CC.and_(reg, OP);                DL; return *this; }
    Value &operator&=(const D<Ref<Value>> &other){ _CC.and_(reg, OP);                DL; return *this; }

    Value &operator|=(const D<int>          &other){ _CC.or_(reg, asmjit::imm(OP)); DL; return *this; }
    Value &operator|=(const D<Value>        &other){ _CC.or_(reg, OP);                DL; return *this; }
    Value &operator|=(const D<Ref<Value>> &other){ _CC.or_(reg, OP);                DL; return *this; }

    Value &operator^=(const D<int>          &other){ _CC.xor_(reg, asmjit::imm(OP)); DL; return *this; }
    Value &operator^=(const D<Value>        &other){ _CC.xor_(reg, OP);                DL; return *this; }
    Value &operator^=(const D<Ref<Value>> &other){ _CC.xor_(reg, OP);                DL; return *this; }

    //TODO: cannot be attributed to a source line as we cannot pass a parameter
    //TODO: we do not know from where we are called
    //TODO: => use free standing function which has the object the unary operator applies to as parameter
    //FIXME: wrong, ~v returns new value, does not modify v, add new method bitwise_not() which does it inplace
    Value &operator~(){ _CC.not_(reg); return *this; }

    // operators creating temporary virtual registers
    ASMJIT_OPERATORS_WITH_TEMPORARIES(Value)

    // comparisons
    Condition operator==(const Value &other) const { return {reg, other.reg, ConditionFlag::e};  }
    Condition operator!=(const Value &other) const { return {reg, other.reg, ConditionFlag::ne}; }
    Condition operator< (const Value &other) const { return {reg, other.reg, less()};  }
    Condition operator<=(const Value &other) const { return {reg, other.reg, less_equal()}; }
    Condition operator> (const Value &other) const { return {reg, other.reg, greater()};  }
    Condition operator>=(const Value &other) const { return {reg, other.reg, greater_equal()}; }
    Condition operator==(int constant) const { return {reg, constant, ConditionFlag::e};  }
    Condition operator!=(int constant) const { return {reg, constant, ConditionFlag::ne}; }
    Condition operator< (int constant) const { return {reg, constant, less()};  }
    Condition operator<=(int constant) const { return {reg, constant, less_equal()}; }
    Condition operator> (int constant) const { return {reg, constant, greater()};  }
    Condition operator>=(int constant) const { return {reg, constant, greater_equal()}; }
    Condition operator==(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::e};  }
    Condition operator!=(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::ne}; }
    Condition operator< (const Ref<Value> &other) const { return {reg, other.mem, less()};  }
    Condition operator<=(const Ref<Value> &other) const { return {reg, other.mem, less_equal()}; }
    Condition operator> (const Ref<Value> &other) const { return {reg, other.mem, greater()};  }
    Condition operator>=(const Ref<Value> &other) const { return {reg, other.mem, greater_equal()}; }

    static inline constexpr ConditionFlag less(){
        if constexpr(std::is_signed_v<T>) return ConditionFlag::l;
        else                              return ConditionFlag::b;
    }
    static inline constexpr ConditionFlag less_equal(){
        if constexpr(std::is_signed_v<T>) return ConditionFlag::le;
        else                              return ConditionFlag::be;
    }
    static inline constexpr ConditionFlag greater(){
        if constexpr(std::is_signed_v<T>) return ConditionFlag::g;
        else                              return ConditionFlag::a;
    }
    static inline constexpr ConditionFlag greater_equal(){
        if constexpr(std::is_signed_v<T>) return ConditionFlag::ge;
        else                              return ConditionFlag::ae;
    }
};

template<>
struct Value<float> final {
    using T = float;
    using value_type = float;
    asmjit::x86::Xmm reg;

    static_assert(sizeof(T)==4,
        "only plain arithmetic types supported of sizes: 4 or 8 bytes");

    Value(const char *name="") {
        switch(sizeof(T)) {
            case 4: reg = _CC.newXmmSs(name); break;
            case 8: reg = _CC.newXmmSd(name); break;
        }
    }
#ifdef PROFILING_SOURCE
    Value(T val, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()) : Value(name) {
        *this = D<T>{val, file, line};
    }
#else
    Value(T val, const char *name="") : Value(name) {
        *this = val;
    }
#endif

    // real copy means new register and copy content
    Value(const D<Value> &other) : Value() {
        *this = other;
    }
    // move ctor
    Value(const Value &&other) : reg(std::move(other.reg)) {}
    // move assign
    Value &operator=(const Value &&other){
        reg = other.reg; // just take virtual register
        return *this;
    }

    // copy ctor for ref, basically loads value from memory and stores in register
    Value(const D<Ref<Value>> &other) : Value() {
        *this = other;
    }
    operator const asmjit::x86::Xmm&() const { return reg; }
    operator       asmjit::x86::Xmm&()       { return reg; }
    // assignment
    Value &operator=(const Value &other) {
        _CC.movss(reg, other.reg);
        DL;
        return *this;
    }
    Value &operator=(const float other) {
        if (other == 0){
            _CC.vpxor(reg, reg, reg);
            DL;
        } else {
            auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, other);
            _CC.vmovss(reg, c0);
            DL;
        }
        return *this;
    }
    Value &operator=(const Ref<Value>& other){
        _CC.movss(reg, other.mem);
        DL;
        return *this;
    }

    // memory operand not possible on right side

    Value &operator*=(const Value& other){
        _CC.vmulss(reg, reg, other.reg);
        DL;
        return *this;
    }

    Value &operator/=(const Value &other){
        _CC.vdivss(reg, reg, other.reg);
        DL;
        return *this;
    }
    Value &operator+=(const Value &other){ _CC.vaddss(reg, reg, other); DL; return *this; }
    Value &operator+=(const Ref<Value> &other){ _CC.vaddss(reg, reg, other); DL; return *this; }
    
    Value &operator-=(const Value &other){ _CC.vsubss(reg, reg, other); DL; return *this; }
    Value &operator-=(const Ref<Value> &other){ _CC.vsubss(reg, reg, other); DL; return *this; }
    Value &operator+=(const float other) {
        if (other == 0){
            DL;
        } else {
            auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, other);
            _CC.vaddss(reg, reg, c0);
            DL;
        }
        return *this;
    }
    Value &operator-=(const float other) {
        if (other == 0){
            DL;
        } else {
            auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, other);
            _CC.vsubss(reg, reg, c0);
            DL;
        }
        return *this;
    }
    Value &operator*=(const float other) {
        if (other == 0){
            _CC.vpxor(reg, reg, reg);
            DL;
        } else {
            auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, other);
            _CC.vmulss(reg, reg, c0);
            DL;
        }
        return *this;
    }
    Value &operator/=(const float other) {
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, other);
        _CC.vdivss(reg, reg, c0);
        DL;
        return *this;
    }
    // operators creating temporary virtual registers
    Value operator+(const Value &other){ Value tmp("tmp"); tmp = *this; tmp += other; DL; return tmp; }
    Value operator+(const Ref<Value> &other){ Value tmp("tmp"); tmp = *this; tmp += other; DL; return tmp; }
    Value operator-(const Value &other){ Value tmp("tmp"); tmp = *this; tmp -= other; DL; return tmp; }
    Value operator-(const Ref<Value> &other){ Value tmp("tmp"); tmp = *this; tmp -= other; DL; return tmp; }
    Value operator*(const Value &other){ Value tmp("tmp"); tmp = *this; tmp *= other; DL; return tmp; }
    Value operator*(const Ref<Value> &other){ Value tmp("tmp"); tmp = *this; tmp *= other; DL; return tmp; }
    Value operator/(const Value &other){ Value tmp("tmp"); tmp = *this; tmp /= other; DL; return tmp; }
    Value operator/(const Ref<Value> &other){ Value tmp("tmp"); tmp = *this; tmp /= other; DL; return tmp; }
    //ASMJIT_OPERATORS_WITH_TEMPORARIES(Value)

    // comparisons
    Condition operator==(const Value &other) const { return {reg, other.reg, ConditionFlag::e_f};  }
    Condition operator!=(const Value &other) const { return {reg, other.reg, ConditionFlag::ne_f}; }
    Condition operator< (const Value &other) const { return {reg, other.reg, ConditionFlag::l_f};  }
    Condition operator<=(const Value &other) const { return {reg, other.reg, ConditionFlag::le_f}; }
    Condition operator> (const Value &other) const { return {reg, other.reg, ConditionFlag::g_f};  }
    Condition operator>=(const Value &other) const { return {reg, other.reg, ConditionFlag::ge_f}; }
    Condition operator==(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::e_f};  }
    Condition operator!=(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::ne_f}; }
    Condition operator< (const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::l_f};  }
    Condition operator<=(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::le_f}; }
    Condition operator> (const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::g_f};  }
    Condition operator>=(const Ref<Value> &other) const { return {reg, other.mem, ConditionFlag::ge_f}; }

    Condition operator==(float f) const {
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::e_f};  }
    Condition operator!=(float f) const { 
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::ne_f}; }
    Condition operator< (float f) const { 
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::l_f};  }
    Condition operator<=(float f) const { 
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::le_f}; }
    Condition operator> (float f) const { 
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::g_f};  }
    Condition operator>=(float f) const { 
        auto c0 = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, f);
        return {reg, c0, ConditionFlag::ge_f}; }
};

// deduction guides
template<typename T> Value(asmjit::x86::Compiler&, T val, const char *) -> Value<T>;
template<typename FnPtr, typename T> Value(Function<FnPtr>&, T val, const char *) -> Value<T>;
template<typename T> Value(const Ref<Value<T>>&) -> Value<T>;

} // namespace
