#pragma once

#include <vector>
#include <thread>
#include <asmjit/asmjit.h>
#include <assert.h>

namespace coat {

struct ThreadCompilerContext {
    asmjit::x86::Compiler cc;
};

// in order to avoid pass cc anywhere make it a global
inline ThreadCompilerContext& CcContext() {
   static thread_local ThreadCompilerContext g;

   return g;
}

// helper macro
#define _CC CcContext().cc

}