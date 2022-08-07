#pragma once

#include "Global.h"

namespace coat {

struct Label final {
    asmjit::Label label;

    Label() : label(_CC.newLabel()) {}

    void bind() {
        _CC.bind(label);
    }

    operator const asmjit::Label&() const { return label; }
    operator       asmjit::Label&()       { return label; }
};

} // namespace
