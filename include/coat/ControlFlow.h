#pragma once

#include "Global.h"

namespace coat {

inline void jump(asmjit::Label label
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    _CC.jmp(label);
#ifdef PROFILING_SOURCE
    ((PerfCompiler&)_CC).attachDebugLine(file, line);
#endif
}
inline void jump(const Condition& cond, asmjit::Label label
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
#ifdef PROFILING_SOURCE
    cond.compare(file, line);
    cond.jump(label, file, line);
#else
    cond.compare();
    cond.jump(label);
#endif
}


inline void ret() {
    _CC.ret();
}
template<typename VReg>
inline void ret(VReg& reg) {
    _CC.ret(reg);
}

template<typename Fn>
void if_then(Condition cond, Fn&& then
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_exit = _CC.newLabel();
    // check
#ifdef PROFILING_SOURCE
    jump(!cond, l_exit, file, line); // if not jump over
#else
    jump(!cond, l_exit); // if not jump over
#endif
    then();
    // label after then branch
    _CC.bind(l_exit);
}

template<typename Then, typename Else>
void if_then_else(Condition cond, Then&& then, Else&& else_
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_else = _CC.newLabel();
    asmjit::Label l_exit = _CC.newLabel();
    // check
#ifdef PROFILING_SOURCE
    jump(!cond, l_else, file, line); // if not jump to else
#else
    jump(!cond, l_else); // if not jump to else
#endif
    then();
#ifdef PROFILING_SOURCE
    jump(l_exit, file, line);
#else
    jump(l_exit);
#endif

    _CC.bind(l_else);
    else_();
    // label after then branch
    _CC.bind(l_exit);
}

template<typename Fn>
void loop_while(Condition cond, Fn&& body
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_loop = _CC.newLabel();
    asmjit::Label l_exit = _CC.newLabel();

    // check if even one iteration
#ifdef PROFILING_SOURCE
    jump(!cond, l_exit, file, line); // if not jump over
#else
    jump(!cond, l_exit); // if not jump over
#endif

    // loop
    _CC.bind(l_loop);
        body();
#ifdef PROFILING_SOURCE
    jump(cond, l_loop, file, line);
#else
    jump(cond, l_loop);
#endif

    // label after loop body
    _CC.bind(l_exit);
}

template<typename StepFn, typename Fn>
void for_loop(Condition cond, StepFn&& step, Fn&& body
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_loop = _CC.newLabel();
    asmjit::Label l_exit = _CC.newLabel();

    // check if even one iteration
#ifdef PROFILING_SOURCE
    jump(!cond, l_exit, file, line); // if not jump over
#else
    jump(!cond, l_exit); // if not jump over
#endif

    // loop
    _CC.bind(l_loop);
        body();
        step();
#ifdef PROFILING_SOURCE
    jump(cond, l_loop, file, line);
#else
    jump(cond, l_loop);
#endif

    // label after loop body
    _CC.bind(l_exit);
}

template<typename Fn>
void do_while(Fn&& body, Condition cond
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_loop = _CC.newLabel();

    // no checking if even one iteration

    // loop
    _CC.bind(l_loop);
        body();
#ifdef PROFILING_SOURCE
    jump(cond, l_loop, file, line);
#else
    jump(cond, l_loop);
#endif
}

template<class Ptr, typename Fn>
void for_each(Ptr& begin, Ptr& end, Fn&& body
#ifdef PROFILING_SOURCE
    , const char* file=__builtin_FILE(), int line=__builtin_LINE()
#endif
) {
    asmjit::Label l_loop = _CC.newLabel();
    asmjit::Label l_exit = _CC.newLabel();

    // check if even one iteration
#ifdef PROFILING_SOURCE
    jump(begin == end, l_exit, file, line);
#else
    jump(begin == end, l_exit);
#endif

    // loop over all elements
    _CC.bind(l_loop);
        typename Ptr::mem_type vr_ele = *begin;
        body(vr_ele);
#ifdef PROFILING_SOURCE
        begin += D<int>{1, file, line};
    jump(begin != end, l_loop, file, line);
#else
        ++begin;
    jump(begin != end, l_loop);
#endif

    // label after loop body
    _CC.bind(l_exit);
}

template<class T, typename Fn>
void for_each(const T& container, Fn&& body) {
    asmjit::Label l_loop = _CC.newLabel();
    asmjit::Label l_exit = _CC.newLabel();

    auto begin = container.begin();
    auto end = container.end();

    // check if even one iteration
    jump(begin == end, l_exit);

    // loop over all elements
    _CC.bind(l_loop);
        auto vr_ele = *begin;
        body(vr_ele);
        ++begin;
    jump(begin != end, l_loop);

    // label after loop body
    _CC.bind(l_exit);
}


// calling function pointer, from generated code to C++ function
template<typename R, typename ...Args>
std::conditional_t<std::is_void_v<R>, void, reg_type<R>>
FunctionCall(R(*fnptr)(Args...), const char*, const wrapper_type<Args>&... arguments) {
    if constexpr(std::is_void_v<R>) {
        asmjit::InvokeNode* c;
        _CC.invoke(&c, (uint64_t)(void*)fnptr, asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
    } else {
        reg_type<R> ret("");
        asmjit::InvokeNode* c;
        _CC.invoke(&c, (uint64_t)(void*)fnptr, asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
        // return value
        c->setRet(0, ret);
        return ret;
    }
}

// calling generated function
template<typename R, typename ...Args>
std::conditional_t<std::is_void_v<R>, void, reg_type<R>>
FunctionCall(const Function<R(*)(Args...)>& func, const wrapper_type<Args>&... arguments) {
    if constexpr(std::is_void_v<R>) {
        asmjit::InvokeNode* c;
        _CC.invoke(&c, func.funcNode->label(), asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
    } else {
        reg_type<R> ret("");
        asmjit::InvokeNode* c;
        _CC.invoke(&c, func.funcNode->label(), asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
        // return value
        c->setRet(0, ret);
        return ret;
    }
}

// calling internal function inside generated code
template<typename R, typename ...Args>
std::conditional_t<std::is_void_v<R>, void, reg_type<R>>
FunctionCall(const InternalFunction<R(*)(Args...)>& func, const wrapper_type<Args>&... arguments) {
    if constexpr(std::is_void_v<R>) {
        asmjit::InvokeNode* c;
        _CC.invoke(&c, func.funcNode->label(), asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
    } else {
        reg_type<R> ret("");
        asmjit::InvokeNode* c;
        _CC.invoke(&c, func.funcNode->label(), asmjit::FuncSignatureT<R, Args...>());
        int index = 0;
        ((c->setArg(index++, arguments)), ...);
        // return value
        c->setRet(0, ret);
        return ret;
    }
}

// pointer difference in bytes, no pointer arithmetic (used by Ptr operators)
template<typename T>
Value<size_t> distance(Ptr<Value<T>>& beg, Ptr<Value<T>>& end) {
    Value<size_t> vr_ret("distance");
    _CC.mov(vr_ret, end);
    _CC.sub(vr_ret, beg);
    return vr_ret;
}

}
