#ifndef COAT_ASMJIT_CONDITION_H_
#define COAT_ASMJIT_CONDITION_H_

#include "coat/Condition.h"

#include <variant>

#include <asmjit/asmjit.h>
#include <assert.h>


namespace coat {

//TODO: combinations of conditions not possible, e.g. "i<j && r<s"
//      really needs expressiont tree in the end

// holds operands and comparison type
// cannot emit instructions directly as comparison emits multiple instructions at different locations
// while-loop: if(cond) do{ ... }while(cond);
struct Condition {
	::asmjit::x86::Compiler &cc; //FIXME: pointer stored in every value type
	// take by value as it might be a temporary which we have to store otherwise it's gone
	::asmjit::x86::Gp reg;
	::asmjit::x86::Xmm reg_xmm;
	//const ::asmjit::Operand &operand;
	using operand_t = std::variant<::asmjit::x86::Gp, int, ::asmjit::x86::Mem, asmjit::x86::Xmm>;
	operand_t operand;
	ConditionFlag cond;
	bool is_float;

	Condition(::asmjit::x86::Compiler &cc, ::asmjit::x86::Gp reg, operand_t operand, ConditionFlag cond)
		: cc(cc), reg(reg), operand(operand), cond(cond), is_float(cond >= ConditionFlag::e_f) {}

	Condition(::asmjit::x86::Compiler &cc, ::asmjit::x86::Xmm reg_, operand_t operand, ConditionFlag cond)
		: cc(cc), reg_xmm(reg_), operand(operand), cond(cond), is_float(cond >= ConditionFlag::e_f) {}

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
			return {cc, reg_xmm, operand, newcond};
		else
			return {cc, reg, operand, newcond};
	}

	void compare(
#ifdef PROFILING_SOURCE
		const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
	) const {
		if (!is_float) {
			switch(operand.index()){
				case 0: cc.cmp(reg, std::get<::asmjit::x86::Gp>(operand)); break;
				case 1: cc.cmp(reg, ::asmjit::imm(std::get<int>(operand))); break;
				case 2: cc.cmp(reg, std::get<::asmjit::x86::Mem>(operand)); break;

				default:
					assert(false);
			}
		} else {
			// 
			switch(operand.index()){
				case 3: cc.comiss(reg_xmm, std::get<::asmjit::x86::Xmm>(operand)); break;
				case 2: cc.comiss(reg_xmm, std::get<::asmjit::x86::Mem>(operand)); break;

				default:
					assert(false);
			}
		}
#ifdef PROFILING_SOURCE
		((PerfCompiler&)cc).attachDebugLine(file, line);
#endif
	}
	void setbyte(::asmjit::x86::Gp &dest) const {
		if (!is_float) {
			switch(cond){
				case ConditionFlag::e : cc.sete (dest); break;
				case ConditionFlag::ne: cc.setne(dest); break;
				case ConditionFlag::l : cc.setl (dest); break;
				case ConditionFlag::le: cc.setle(dest); break;
				case ConditionFlag::g : cc.setg (dest); break;
				case ConditionFlag::ge: cc.setge(dest); break;
				case ConditionFlag::b : cc.setb (dest); break;
				case ConditionFlag::be: cc.setbe(dest); break;
				case ConditionFlag::a : cc.seta (dest); break;
				case ConditionFlag::ae: cc.setae(dest); break;

				default:
					assert(false);
			}
		} else {
			assert(false);
		}
	}
	void jump(::asmjit::Label label
#ifdef PROFILING_SOURCE
		, const char *file=__builtin_FILE(), int line=__builtin_LINE()
#endif
	) const {
		if (!is_float) {
			switch(cond){
				case ConditionFlag::e : cc.je (label); break;
				case ConditionFlag::ne: cc.jne(label); break;
				case ConditionFlag::l : cc.jl (label); break;
				case ConditionFlag::le: cc.jle(label); break;
				case ConditionFlag::g : cc.jg (label); break;
				case ConditionFlag::ge: cc.jge(label); break;
				case ConditionFlag::b : cc.jb (label); break;
				case ConditionFlag::be: cc.jbe(label); break;
				case ConditionFlag::a : cc.ja (label); break;
				case ConditionFlag::ae: cc.jae(label); break;

				default:
					assert(false);
			}
		} else {
			switch(cond) {
				// https://stackoverflow.com/questions/30562968/xmm-cmp-two-32-bit-float
				case ConditionFlag::e_f : cc.je(label); break;
				case ConditionFlag::ne_f: cc.jne(label); break;
				case ConditionFlag::l_f : cc.jb(label); break;
				case ConditionFlag::le_f: cc.jbe(label); break;
				case ConditionFlag::g_f : cc.ja(label); break;
				case ConditionFlag::ge_f: cc.jae(label); break;

				default:
					assert(false);
			}
		}
#ifdef PROFILING_SOURCE
		((PerfCompiler&)cc).attachDebugLine(file, line);
#endif
	}
};

} // namespace

#endif
