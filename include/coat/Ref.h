#pragma once

#include "Global.h"
#include "Ref.h"
#include "operator_helper.h"
#include "Condition.h"

namespace coat {

//FIXME: looks like asmjit::x86::Mem is copied around quite often during construction
template<class T>
struct Ref {
    using inner_type = T;

    asmjit::x86::Mem mem;

    // NOTE: copy the reference!
    Ref(const Ref& other) : mem(other.mem) {}
    Ref(asmjit::x86::Mem mem) : mem(mem) {}
    operator const asmjit::x86::Mem&() const { return mem; }
    operator       asmjit::x86::Mem&()       { return mem; }

    // assignment storing to memory
    Ref &operator=(const D<T> &other){
        _CC.mov(mem, OP);
        DL;
        return *this;
    }
    Ref &operator=(const D<typename inner_type::value_type> &other){
        if constexpr(sizeof(typename inner_type::value_type) == 8){
            // mov to memory operand not allowed with imm64
            // copy immediate first to register, then store
            asmjit::x86::Gp temp;
            if constexpr(std::is_signed_v<typename inner_type::value_type>){
                temp = _CC.newInt64();
            }else{
                temp = _CC.newUInt64();
            }
            _CC.mov(temp, asmjit::imm(OP));
            DL;
            _CC.mov(mem, temp);
            DL;
        }else{
            _CC.mov(mem, asmjit::imm(OP));
            DL;
        }
        return *this;
    }
    // arithmetic + assignment skipped for now

    // operators creating temporary virtual registers
    ASMJIT_OPERATORS_WITH_TEMPORARIES(T)

    // comparisons
    // swap sides of operands and comparison, not needed for assembly, but avoids code duplication in wrapper
    Condition operator==(const T &other) const { return other==*this; }
    Condition operator!=(const T &other) const { return other!=*this; }
    Condition operator< (const T &other) const { return other>=*this; }
    Condition operator<=(const T &other) const { return other> *this; }
    Condition operator> (const T &other) const { return other<=*this; }
    Condition operator>=(const T &other) const { return other< *this; }
    //TODO: possible without temporary: cmp m32 imm32, complicates Condition
    Condition operator==(int constant) const { T tmp("tmp"); tmp = *this; return tmp==constant; }
    Condition operator!=(int constant) const { T tmp("tmp"); tmp = *this; return tmp!=constant; }
    Condition operator< (int constant) const { T tmp("tmp"); tmp = *this; return tmp< constant; }
    Condition operator<=(int constant) const { T tmp("tmp"); tmp = *this; return tmp<=constant; }
    Condition operator> (int constant) const { T tmp("tmp"); tmp = *this; return tmp> constant; }
    Condition operator>=(int constant) const { T tmp("tmp"); tmp = *this; return tmp>=constant; }
    // not possible in instruction, requires temporary
    Condition operator==(const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp==other; }
    Condition operator!=(const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp!=other; }
    Condition operator< (const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp< other; }
    Condition operator<=(const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp<=other; }
    Condition operator> (const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp> other; }
    Condition operator>=(const Ref &other) const { T tmp("tmp"); tmp = *this; return tmp>=other; }
};

} // namespace
