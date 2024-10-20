#pragma once

#include "Global.h"

namespace coat {

struct Mask {
    asmjit::x86::KReg reg;

    Mask(const char* name="") {
        reg = _CC.newKq(name);
    }
    Mask(asmjit::x86::KReg reg) : reg(reg) {}
    Mask(const Mask&& other) : reg(other.reg) {}

    operator const asmjit::x86::KReg&() const { return reg; }
    operator       asmjit::x86::KReg&()       { return reg; }
};

} // namespace
