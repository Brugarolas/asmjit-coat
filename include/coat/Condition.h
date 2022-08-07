#pragma once

#include "Global.h"
#include <variant>

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

//TODO: combinations of conditions not possible, e.g. "i<j && r<s"
//      really needs expressiont tree in the end

// holds operands and comparison type
// cannot emit instructions directly as comparison emits multiple instructions at different locations
// while-loop: if(cond) do{ ... }while(cond);
struct Condition {
    // take by value as it might be a temporary which we have to store otherwise it's gone
    asmjit::x86::Gp reg;
    asmjit::x86::Xmm reg_xmm;
    //const asmjit::Operand &operand;
    using operand_t = std::variant<asmjit::x86::Gp, int, asmjit::x86::Mem, asmjit::x86::Xmm>;
    operand_t operand;
    ConditionFlag cond;
    bool is_float;

    Condition(asmjit::x86::Gp reg, operand_t operand, ConditionFlag cond)
        : reg(reg), operand(operand), cond(cond), is_float(cond >= ConditionFlag::e_f) {}

    Condition(asmjit::x86::Xmm reg_, operand_t operand, ConditionFlag cond)
        : reg_xmm(reg_), operand(operand), cond(cond), is_float(cond >= ConditionFlag::e_f) {}

    Condition operator!() const {
        ConditionFlag newcond;
        switch(cond){
            case ConditionFlag::e : newcond = ConditionFlag::ne; break;
            case ConditionFlag::ne: newcond = ConditionFlag::e ; break;
            case ConditionFlag::l : newcond = ConditionFlag::ge; break;
            case ConditionFlag::le: newcond = ConditionFlag::g ; break;
            case ConditionFlag::g : newcond = ConditionFlag::le; break;
            case ConditionFlag::ge: newcond = ConditionFlag::l ; break;
            case ConditionFlag::b : newcond = ConditionFlag::ae; break;
            case ConditionFlag::be: newcond = ConditionFlag::a ; break;
            case ConditionFlag::a : newcond = ConditionFlag::be; break;
            case ConditionFlag::ae: newcond = ConditionFlag::b ; break;

            case ConditionFlag::e_f : newcond = ConditionFlag::ne_f; break;
            case ConditionFlag::ne_f: newcond = ConditionFlag::e_f ; break;
            case ConditionFlag::l_f : newcond = ConditionFlag::ge_f; break;
            case ConditionFlag::le_f: newcond = ConditionFlag::g_f ; break;
            case ConditionFlag::g_f : newcond = ConditionFlag::le_f; break;
            case ConditionFlag::ge_f: newcond = ConditionFlag::l_f ; break;
            default:
                assert(false);
        }
        if (is_float)
            return {reg_xmm, operand, newcond};
        else
            return {reg, operand, newcond};
    }

    void compare(
#ifdef PROFILING_SOURCE
        const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
    ) const {
        if (!is_float) {
            switch(operand.index()){
                case 0: _CC.cmp(reg, std::get<asmjit::x86::Gp>(operand)); break;
                case 1: _CC.cmp(reg, asmjit::imm(std::get<int>(operand))); break;
                case 2: _CC.cmp(reg, std::get<asmjit::x86::Mem>(operand)); break;

                default:
                    assert(false);
            }
        } else {
            // 
            switch(operand.index()){
                case 3: _CC.comiss(reg_xmm, std::get<asmjit::x86::Xmm>(operand)); break;
                case 2: _CC.comiss(reg_xmm, std::get<asmjit::x86::Mem>(operand)); break;

                default:
                    assert(false);
            }
        }
#ifdef PROFILING_SOURCE
        ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
    }
    void setbyte(asmjit::x86::Gp &dest) const {
        if (!is_float) {
            switch(cond){
                case ConditionFlag::e : _CC.sete (dest); break;
                case ConditionFlag::ne: _CC.setne(dest); break;
                case ConditionFlag::l : _CC.setl (dest); break;
                case ConditionFlag::le: _CC.setle(dest); break;
                case ConditionFlag::g : _CC.setg (dest); break;
                case ConditionFlag::ge: _CC.setge(dest); break;
                case ConditionFlag::b : _CC.setb (dest); break;
                case ConditionFlag::be: _CC.setbe(dest); break;
                case ConditionFlag::a : _CC.seta (dest); break;
                case ConditionFlag::ae: _CC.setae(dest); break;

                default:
                    assert(false);
            }
        } else {
            assert(false);
        }
    }
    void jump(asmjit::Label label
#ifdef PROFILING_SOURCE
        , const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
    ) const {
        if (!is_float) {
            switch(cond){
                case ConditionFlag::e : _CC.je (label); break;
                case ConditionFlag::ne: _CC.jne(label); break;
                case ConditionFlag::l : _CC.jl (label); break;
                case ConditionFlag::le: _CC.jle(label); break;
                case ConditionFlag::g : _CC.jg (label); break;
                case ConditionFlag::ge: _CC.jge(label); break;
                case ConditionFlag::b : _CC.jb (label); break;
                case ConditionFlag::be: _CC.jbe(label); break;
                case ConditionFlag::a : _CC.ja (label); break;
                case ConditionFlag::ae: _CC.jae(label); break;

                default:
                    assert(false);
            }
        } else {
            switch(cond) {
                // https://stackoverflow.com/questions/30562968/xmm-cmp-two-32-bit-float
                case ConditionFlag::e_f : _CC.je(label); break;
                case ConditionFlag::ne_f: _CC.jne(label); break;
                case ConditionFlag::l_f : _CC.jb(label); break;
                case ConditionFlag::le_f: _CC.jbe(label); break;
                case ConditionFlag::g_f : _CC.ja(label); break;
                case ConditionFlag::ge_f: _CC.jae(label); break;

                default:
                    assert(false);
            }
        }
#ifdef PROFILING_SOURCE
        ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
    }
};

} // namespace
