#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include <coat/Function.h>
#include <coat/ControlFlow.h>


uint32_t fib(uint32_t index){
    if(index < 2){
        return index;
    }else{
        return fib(index-1) + fib(index-2);
    }
}


template<class Fn>
void assemble_selfcall(Fn &fn){
    //auto [index] = fn.getArguments("index"); // clang does not like it, bindings cannot be used in lambda captures
    auto args = fn.getArguments("index");
    auto &index = std::get<0>(args);
    coat::if_then_else(index < 2, [&]{
        coat::ret(index);
    }, [&]{
        auto ret = coat::FunctionCall(fn, index-1);
        ret += coat::FunctionCall(fn, index-2);
        coat::ret(ret);
    });
}

template<class Fn, typename Fnptr>
void assemble_crosscall(Fn &fn, Fnptr fnptr, const char *funcname){
    auto [index] = fn.getArguments("index");
    //FIXME: needs symbol name of function for LLVM, that is not exposed...
    auto ret = coat::FunctionCall(fnptr, funcname, index);
    coat::ret(ret);
}


template<class Fn>
void assemble_allinone(Fn &fn){
    // signature of internal function inside generated code
    using internalfunc_t = uint32_t (*)(uint32_t index);

    // create internal function handle to call it and insert code into it later
    auto internalCall = fn.template addFunction<internalfunc_t>("rec2");
    // EDSL of outer function first
    {
        auto [index] = fn.getArguments("index");
        auto ret = coat::FunctionCall(internalCall, index);
        coat::ret(ret);
    }

    // end outer function and start internal function
    fn.startNextFunction(internalCall);
    // EDSL for internal function
    {
        //auto [index] = internalCall.getArguments("index");
        auto args = internalCall.getArguments("index");
        auto &index = std::get<0>(args);
        coat::if_then_else(index < 2, [&]{
            coat::ret(index);
        }, [&]{
            auto ret = coat::FunctionCall(internalCall, index-1);
            ret += coat::FunctionCall(internalCall, index-2);
            coat::ret(ret);
        });
    }
}


#ifdef ENABLE_LLVMJIT
template<typename F>
static void verifyAndOptimize(F &function, const char *fname1, const char *fname2){
    function.printIR(fname1);
    if(!function.verify()){
        puts("verification failed. aborting.");
        exit(EXIT_FAILURE); //FIXME: better error handling
    }
    function.optimize(2);
    function.printIR(fname2);
}
#endif

int main(int argc, char *argv[]){
    if(argc < 2){
        puts("argument required: index in fibonacci sequence");
        return -1;
    }
    uint32_t index = atoi(argv[1]);
    uint32_t expected = fib(index);

    // init JIT backends
#ifdef ENABLE_ASMJIT
    coat::runtimeasmjit asmrt;
#endif
#ifdef ENABLE_LLVMJIT
    coat::runtimellvmjit::initTarget();
    coat::runtimellvmjit llvmrt;
#endif

    // signature of the generated function
    using func_t = uint32_t (*)(uint32_t index);

#ifdef ENABLE_ASMJIT
    {
        // context object representing the generated function
        coat::Function<func_t> fn(asmrt);
        assemble_selfcall(fn);
        // finalize code generation and get function pointer to the generated function
        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("selfcall asmjit:\nresult: %u; expected: %u\n", result, expected);
    }
#endif
#ifdef ENABLE_LLVMJIT
    {
        // context object representing the generated function
        coat::Function<coat::runtimellvmjit,func_t> fn(llvmrt);
        assemble_selfcall(fn);
        verifyAndOptimize("selfcall.ll", "selfcall_opt.ll");
        // finalize code generation and get function pointer to the generated function
        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("selfcall llvm:\nresult: %u; expected: %u\n", result, expected);
    }
#endif

#ifdef ENABLE_ASMJIT
    {
        coat::Function<func_t> fnrec(asmrt);
        assemble_selfcall(fnrec);
        func_t foorec = fnrec.finalize();

        coat::Function<func_t> fn(asmrt);
        assemble_crosscall(fn, foorec, "");
        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("crosscall asmjit:\nresult: %u; expected: %u\n", result, expected);
    }
#endif
#ifdef ENABLE_LLVMJIT
    {
        coat::Function<coat::runtimellvmjit,func_t> fnrec(llvmrt, "rec");
        assemble_selfcall(fnrec);
        func_t foorec = fnrec.finalize();

        coat::Function<coat::runtimellvmjit,func_t> fn(llvmrt, "caller");
        assemble_crosscall(foorec, "rec");

        verifyAndOptimize("crosscall.ll", "crosscall_opt.ll");

        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("crosscall llvm:\nresult: %u; expected: %u\n", result, expected);
    }
#endif
#ifdef ENABLE_ASMJIT
    {
        // context object representing the generated function
        coat::Function<func_t> fn(asmrt);
        assemble_allinone(fn);
        // finalize code generation and get function pointer to the generated function
        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("allinone asmjit:\nresult: %u; expected: %u\n", result, expected);
    }
#endif
#ifdef ENABLE_LLVMJIT
    {
        // context object representing the generated function
        coat::Function<coat::runtimellvmjit,func_t> fn(llvmrt, "caller2");
        assemble_allinone(fn);

        verifyAndOptimize("allinone.ll", "allinone_opt.ll");

        // finalize code generation and get function pointer to the generated function
        func_t foo = fn.finalize();
        // execute the generated function
        uint32_t result = foo(index);
        printf("allinone llvmjit:\nresult: %u; expected: %u\n", result, expected);
    }
#endif

    return 0;
}
