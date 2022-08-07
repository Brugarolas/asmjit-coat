#include <cstdio>
#include <vector>
#include <numeric>

#include <coat/Function.h>
#include <coat/ControlFlow.h>

#define ENABLE_DUMP 0

enum class compute_type {
    add,
    sub,
    mul,
    div
};
void test_compute(compute_type type) {
    printf("start test float type %d...\n", (int)type);
    std::vector<float> array(5);
    std::iota(array.begin(), array.end(), 0.0f);
    float expected;
    using func_t = float (*)(float *data, uint64_t size);

    // context object representing the generated function
    auto fn = coat::createFunction<func_t>();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        // get function arguments as "meta-variables"
        auto [data, size] = fn.getArguments("data", "size");

        // reg for sum
        coat::Value sum(float(0), "sum");
        switch (type)
        {
        case compute_type::add:
            sum += data[0] + data[1] + data[2] + data[3] + data[4];
            expected = array[0] + array[1] + array[2] + array[3] + array[4];
            break;
        case compute_type::sub:
            sum -= 1.0f;
            sum -= data[0] - data[1] - data[2] - data[3] - data[4];
            expected = -1.0f - (array[0] - array[1] - array[2] - array[3] - array[4]);
            break;
        case compute_type::mul:
            sum += 1.0f;
            sum *= 2.0f;
            sum *= (data[0] - data[1]) * (data[2] + data[3]) - data[4];
            expected = 2.0f * ((array[0] - array[1]) * (array[2] + array[3]) - array[4]);
            break;
        case compute_type::div:
            sum = 6.0f;
            sum /= 2.0f;
            sum /= (data[0] / data[1]) / (data[2] / data[3]) - data[4];
            expected = 3.0f / ((array[0] / array[1]) / (array[2] / array[3]) - array[4]);
            break;
        default:
            break;
        }

        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    func_t foo = fn.finalize();

    // execute the generated function
    auto result = foo(array.data(), array.size());

    // print result
    if(result == expected) {
        printf("type: %d correct result: %f\n", (int)type, result);
    } else {
        printf("type: %d wrong result:\nresult: %f; expected: %f\n", (int)type, result, expected);
    }
}

enum class cond_type {
    less,
    less_equal,
    great,
    great_equal,
    equal
};
void test_cond(cond_type type) {
    printf("start test float cond type %d...\n", (int)type);
    float x = 5.0f;
    float expected;
    using func_t = float (*)(float data);

    // context object representing the generated function
    auto fn = coat::createFunction<func_t>();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        // get function arguments as "meta-variables"
        auto [data] = fn.getArguments("data");

        // reg for sum
        coat::Value sum(float(0), "sum");
        switch (type)
        {
        case cond_type::less:
            coat::if_then(data < 10.0f, [&] {
                sum = 1.0f;
            });
            expected = 1.0f;
            coat::if_then_else(data < sum + sum,
            [&] {

            },
            [&] {
                sum += 2.0f;
            });
            expected += 2.0f;
            break;
        case cond_type::less_equal:
            coat::if_then(data <= 5.0f, [&]{
                sum = 1.0f;
            });
            expected = 1.0f;
            coat::if_then_else(data <= sum + sum,
            [&] {

            },
            [&] {
                sum += 2.0f;
            });            
            expected += 2.0f;
            break;
        case cond_type::great:
            coat::if_then(data > 4.0f, [&]{
                sum = 1.0f;
            });
            expected = 1.0f;
            coat::if_then_else(sum > data,
            [&] {

            },
            [&] {
                sum += 2.0f;
            });            
            expected += 2.0f;
            break;
        case cond_type::great_equal:
            coat::if_then(data >= 5.0f, [&]{
                sum = 1.0f;
            });
            expected = 1.0f;
            coat::if_then_else(sum >= data,
            [&] {

            },
            [&] {
                sum += 2.0f;
            });            
            expected += 2.0f;
            break;
        case cond_type::equal:
            coat::if_then(data == 5.0f, [&]{
                sum = 5.0f;
            });            
            expected = 5.0f;
            coat::if_then_else(data != 5.0f,
            [&]{
            },
            [&]{
                sum += 5.0f;
            });            
            expected += 5.0f;
            break;
        default:
            break;
        }

        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    func_t foo = fn.finalize();

    // execute the generated function
    auto result = foo(x);

    // print result
    if(result == expected) {
        printf("float cond type: %d correct result: %f\n", (int)type, result);
    } else {
        printf("float cond type: %d wrong result:\nresult: %f; expected: %f\n", (int)type, result, expected);
    }
}

int main(){
    test_compute(compute_type::add);
    test_compute(compute_type::sub);
    test_compute(compute_type::mul);
    test_compute(compute_type::div);

    test_cond(cond_type::less);
    test_cond(cond_type::less_equal);
    test_cond(cond_type::great);
    test_cond(cond_type::great_equal);
    test_cond(cond_type::equal);
}
