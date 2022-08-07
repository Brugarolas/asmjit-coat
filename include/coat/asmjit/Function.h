#ifndef COAT_ASMJIT_FUNCTION_H_
#define COAT_ASMJIT_FUNCTION_H_

#include "../runtimeasmjit.h"
#ifdef PROFILING_SOURCE
#    include <asmjit-utilities/perf/perfcompiler.h>
#endif

#include <tuple> // apply
#include <cstdio>


namespace coat {

template<typename R, typename ...Args>
struct Function<R(*)(Args...)> {
    using CC = runtimeasmjit;
    using func_type = R (*)(Args...);
    using return_type = R;

    runtimeasmjit &asmrt;
    ::asmjit::CodeHolder code;
#ifdef PROFILING_SOURCE
    PerfCompiler cc;
#else
#endif

    ::asmjit::FileLogger logger;

    const char *funcName;
    ::asmjit::FuncNode *funcNode;

    Function(runtimeasmjit &asmrt, const char *funcName="func") : asmrt(asmrt), funcName(funcName) {
        code.init(asmrt.rt.environment());
        code.setErrorHandler(&asmrt.errorHandler);
        code.attach(&_CC);

        funcNode = _CC.addFunc(::asmjit::FuncSignatureT<R,Args...>());
    }
    Function(const Function &other) = delete;

    void enableCodeDump(FILE *fd=stdout){
        logger.setFlags(asmjit::FormatFlags::kHexOffsets);
        logger.setFile(fd);
        code.setLogger(&logger);
    }

    template<typename FuncSig>
    InternalFunction<FuncSig> addFunction(const char* /* ignore function name */){
        return InternalFunction<FuncSig>();
    }

    template<class IFunc>
    void startNextFunction(const IFunc &internalCall){
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
            [&](auto &&...args){
                int idx=0;
                ((funcNode->setArg(idx++, args)), ...);
            },
            ret
        );
        return ret;
    }

    //HACK: trying factory
    template<typename T>
    Value<T> getValue(const char *name="") {
        return Value<T>(name);
    }
    // embed value in the generated code, returns wrapper initialized to this value
    template<typename T>
#ifdef PROFILING_SOURCE
    wrapper_type<T> embedValue(T value, const char *name="", const char *file=__builtin_FILE(), int line=__builtin_LINE()){
        return wrapper_type<T>(value, name, file, line);
    }
#else
    wrapper_type<T> embedValue(T value, const char *name=""){
        return wrapper_type<T>(value, name);
    }
#endif

    func_type finalize(){
        func_type fn;

        _CC.endFunc();
        _CC.finalize(
#ifdef PROFILING_SOURCE
            asmrt.jd
#endif
        );

        ::asmjit::Error err = asmrt.rt.add(&fn, &code);
        if(err){
            fprintf(stderr, "runtime add failed with CodeCompiler\n");
            std::exit(1);
        }
        // dump generated code for profiling with perf
#if defined(PROFILING_ASSEMBLY) || defined(PROFILING_SOURCE)
        asmrt.jd.addCodeSegment(funcName, (void*)fn, code.codeSize());
#endif
        return fn;
    }
};


template<typename R, typename ...Args>
struct InternalFunction<R(*)(Args...)> {
    using func_type = R (*)(Args...);
    using return_type = R;

    ::asmjit::FuncNode *funcNode;

    InternalFunction() {
        funcNode = _CC.newFunc(::asmjit::FuncSignatureT<R,Args...>());
    }
    InternalFunction(const InternalFunction &other) : funcNode(other.funcNode) {}


    template<typename ...Names>
    std::tuple<wrapper_type<Args>...> getArguments(Names... names) {
        static_assert(sizeof...(Args) == sizeof...(Names), "not enough or too many names specified");
        // create all parameter wrapper objects in a tuple
        std::tuple<wrapper_type<Args>...> ret { wrapper_type<Args>(names)... };
        // get argument value and put it in wrapper object
        std::apply(
            [&](auto &&...args){
                int idx=0;
                ((funcNode->setArg(idx++, args)), ...);
            },
            ret
        );
        return ret;
    }
};


} // namespace

#endif
