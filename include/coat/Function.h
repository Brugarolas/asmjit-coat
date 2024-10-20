#pragma once

#include <tuple> // apply
#include <cstdio>
#include <type_traits>
#include "Global.h"
#include <perf-util/jit_utils/jit_utils.hpp>
#ifdef PROFILING_SOURCE
#    include <asmjit-utilities/perf/perfcompiler.h>
#endif

namespace coat {

//FIXME: forwards
template<typename T> struct Value;
template<typename T> struct Ptr;
template<typename T> struct Struct;

template<typename T>
using reg_type = std::conditional_t<std::is_pointer_v<T>,
                        Ptr<Value<std::remove_cv_t<std::remove_pointer_t<T>>>>,
                        Value<std::remove_cv_t<T>>
                >;

// decay - converts array types to pointer types
template<typename T>
using wrapper_type = std::conditional_t<std::is_arithmetic_v<std::remove_pointer_t<std::decay_t<T>>>,
                        reg_type<std::decay_t<T>>,
                        Struct<std::remove_extent_t<std::remove_cv_t<std::remove_pointer_t<T>>>>
                    >;

template<typename T>
struct Function;

template<typename T>
struct InternalFunction;

template<typename R, typename ...Args>
struct Function<R(*)(Args...)> {
    using func_type = R (*)(Args...);
    using return_type = R;

    asmjit::CodeHolder code;
#ifdef PROFILING_SOURCE
    PerfCompiler cc;
#else
#endif

    asmjit::FileLogger logger;

    const char* funcName;
    asmjit::FuncNode* funcNode;

    class MyErrorHandler : public asmjit::ErrorHandler{
    public:
        void handleError(asmjit::Error /*err*/, const char* msg, asmjit::BaseEmitter* /*origin*/) override {
            fprintf(stderr, "ERROR: %s\n", msg);
        }
    } errorHandler;

    Function(const char* funcName="func") : funcName(funcName) {
        code.init(getJitRuntimeEnv().rt.environment());
        code.setErrorHandler(&errorHandler);
        code.attach(&_CC);

        funcNode = _CC.addFunc(asmjit::FuncSignatureT<R,Args...>());
    }
    NONCOPYABLE(Function);

    void enableCodeDump(FILE* fd=stdout) {
        logger.setFlags(asmjit::FormatFlags::kHexOffsets);
        logger.setFile(fd);
        code.setLogger(&logger);
    }

    template<typename FuncSig>
    InternalFunction<FuncSig> addFunction(const char* /* ignore function name */) {
        return InternalFunction<FuncSig>();
    }

    template<class IFunc>
    void startNextFunction(const IFunc& internalCall) {
        // close previous function
        _CC.endFunc();
        // start passed function
        _CC.addFunc(internalCall.funcNode);
    }

    template<typename ...Names>
    std::tuple<wrapper_type<Args>...> getArguments(Names... names) {
        static_assert(sizeof...(Args) == sizeof...(Names), "not enough or too many names specified");
        // create all parameter wrapper objects in a tuple
        std::tuple<wrapper_type<Args>...> ret { wrapper_type<Args>(names)... };
        // get argument value and put it in wrapper object
        std::apply(
            [&](auto&& ...args) {
                int idx = 0;
                ((funcNode->setArg(idx++, args)), ...);
            },
            ret
        );
        return ret;
    }

    //HACK: trying factory
    template<typename T>
    Value<T> getValue(const char* name="") {
        return Value<T>(name);
    }
    // embed value in the generated code, returns wrapper initialized to this value
    template<typename T>
#ifdef PROFILING_SOURCE
    wrapper_type<T> embedValue(T value, const char* name="", const char* file=__builtin_FILE(), int line=__builtin_LINE()) {
        return wrapper_type<T>(value, name, file, line);
    }
#else
    wrapper_type<T> embedValue(T value, const char* name="") {
        return wrapper_type<T>(value, name);
    }
#endif

    func_type finalize() {
        func_type fn;

        _CC.endFunc();
        _CC.finalize(
#ifdef PROFILING_SOURCE
            asmrt.jd
#endif
        );

        asmjit::Error err = getJitRuntimeEnv().rt.add(&fn, &code);
        if (err) {
            fprintf(stderr, "runtime add failed with CodeCompiler\n");
            std::exit(1);
        }
        dnnl::impl::cpu::jit_utils::register_jit_code((void*)fn, code.codeSize(), funcName, __FILE__);
        // dump generated code for profiling with perf
#if defined(PROFILING_ASSEMBLY) || defined(PROFILING_SOURCE)
        getJitRuntimeEnv().rt.jd.addCodeSegment(funcName, (void*)fn, code.codeSize());
#endif
        return fn;
    }
};


template<typename R, typename ...Args>
struct InternalFunction<R(*)(Args...)> {
    using func_type = R (*)(Args...);
    using return_type = R;

    asmjit::FuncNode* funcNode;

    InternalFunction() {
        funcNode = _CC.newFunc(asmjit::FuncSignatureT<R,Args...>());
    }
    InternalFunction(const InternalFunction& other) : funcNode(other.funcNode) {}


    template<typename ...Names>
    std::tuple<wrapper_type<Args>...> getArguments(Names... names) {
        static_assert(sizeof...(Args) == sizeof...(Names), "not enough or too many names specified");
        // create all parameter wrapper objects in a tuple
        std::tuple<wrapper_type<Args>...> ret { wrapper_type<Args>(names)... };
        // get argument value and put it in wrapper object
        std::apply(
            [&](auto&& ...args) {
                int idx = 0;
                ((funcNode->setArg(idx++, args)), ...);
            },
            ret
        );
        return ret;
    }
};

template<typename FnPtr>
Function<FnPtr> createFunction(const char* funcName="func") {
    return Function<FnPtr>(funcName);
}

} // namespace

#include "Condition.h"
#include "Ref.h"
#include "Value.h"
#include "Ptr.h"
#include "Struct.h"
