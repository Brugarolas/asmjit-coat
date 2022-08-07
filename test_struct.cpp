#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>

#define ENABLE_DUMP 1

struct BinayParam {
    COAT_NAME("BinayParam");
    #define MEMBERS(x)    \
        x(float*, xx)    \
        x(float*, p)    \
        x(int, type)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS
};
// array should use alias to workaround macro
using BinayParamArr = BinayParam[2];
struct ConvParam {
    COAT_NAME("ConvParam");
    #define MEMBERS(x)    \
        x(float, tt)      \
        x(float*, src)    \
        x(float*, weight) \
        x(float*, dst)    \
        x(size_t, size)   \
        x(BinayParamArr, bin)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS
    // float* src;
    // float* weight;
    // float* dst;
    // size_t size;
    // BinayParam bin[2];

    // enum member_ids : int {
    //     member_src,
    //     member_weight,
    //     member_dst,
    //     member_size,
    //     member_bin
    // };
    // static constexpr std::array member_names {
    //     "src",
    //     "weight",
    //     "dst",
    //     "size",
    //     "bin"
    // };
    // using types = std::tuple<
    //     float*, float*, float*, size_t, BinayParam[2],
    // void>;
};

void test_struct() {
    printf("start test struct...\n");
    std::vector<float> array_x(16);
    std::vector<float> array_y(16);
    std::vector<float> array_z(16);
    std::iota(array_x.begin(), array_x.end(), 0.0f);
    std::iota(array_y.begin(), array_y.end(), 1.0f);
    std::vector<float> expected(16);
    using func_t = void (*)(ConvParam*);

    // context object representing the generated function
    auto fn = coat::createFunction<func_t>();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        // get function arguments as "meta-variables"
        auto [param] = fn.getArguments("param");
        //auto x = param.get_value<ConvParam::member_src>("src");
        auto y = param.get_value<ConvParam::member_weight>("weight");
        auto z = param.get_value<ConvParam::member_dst>("dst");
        auto size = param.get_value<ConvParam::member_size>("size");
        auto bin = param.get_value<ConvParam::member_bin>("bin");
        auto x = bin[1].get_value<BinayParam::member_p>("p");
        auto f = param.get_value<ConvParam::member_tt>("tt");
        
        static const int vectorsize = 8;
        coat::Value pos(uint64_t(0), "pos");
        coat::do_while([&]{
            auto x0 = coat::make_vector<vectorsize>(x[pos]);
            auto y0 = coat::make_vector<vectorsize>(y[pos]);
            x0 += y0;
            x0.store(z[pos]);
            for (size_t i = 0; i < expected.size(); i++) {
                expected[i] = array_x[i] + array_y[i];
            }
            // move to next vector
            pos += vectorsize;
        }, pos < size);

        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    func_t foo = fn.finalize();

    // execute the generated function
    ConvParam param;
    param.src = array_x.data();
    param.weight = array_y.data();
    param.dst = array_z.data();
    param.size = array_x.size();
    param.bin[1].p = array_x.data();
    foo(&param);
    // print result
    if(array_z == expected) {
        printf("correct \n");
    } else {
        printf("wrong result\n");
    }
}

int main(){
    test_struct();
}
