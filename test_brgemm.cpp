#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iostream>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>

using namespace std::chrono;

#define ENABLE_DUMP 0

////////////////////////////////////////////////////
// post ops setting params, interface
enum class AlgType {
    // Unary: x = f(x)
    Abs,
    // Binary: x = f(x, y), x and y are variable
    Add,
    Sub,
    Mul,
    ReLU,
    BatchNorm,
    // BinaryConst: x = f(x, c1, c2), x is varible and c1/c2 is const
    Add_C,
    Sub_C,
    Mul_C,
};

struct UnaryParam {
    float x1;
    float x2;
    float x3;
    float x4;
};

enum class BinaryDataLayout {
    PerTensor,
    PerChannel,
    PerElement
};
struct BinaryParam {
    BinaryDataLayout layout;
};

struct PostOp {
    AlgType alg_type;
    union {
        UnaryParam unary_param;
        BinaryParam binary_param;
    };
};

#define MAX_POSTOPS_NUM 10
struct PostOps {
    int num = 0;
    PostOp ops[MAX_POSTOPS_NUM];
};

////////////////////////////////////////////////////
// jit kernel param when calling kernels, private
struct JitParam {
    COAT_NAME("JitParam");
    #define MEMBERS(x)    \
        x(float*, right_addr)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS
    // int8_t* right_addr; // second param address
};

// array should use alias to workaround macro
using JitParamArr = JitParam[MAX_POSTOPS_NUM];
struct JitPostOps {
    COAT_NAME("JitPostOps");
    #define MEMBERS(x)    \
        x(JitParamArr, params)

    COAT_DECLARE_PRIVATE(MEMBERS)
    #undef MEMBERS    
    // JitParam params[MAX_POSTOPS_NUM];
};

////////////////////////////////////////////////////
// generate jit kernel needed param when injecting kernels, private
struct JitInjectPostOp {
    BinaryDataLayout layout;
    std::vector<coat::Ref<coat::Value<float>>> right_addrs;
};

struct JitInjectPostOps {
    JitInjectPostOp params[MAX_POSTOPS_NUM];
};

template <unsigned width>
void inject_postops(std::vector<coat::Vec<float, width>*> vecs, PostOps* ops_param, JitInjectPostOps* inject_ops_param) {
    for (auto i = 0; i < ops_param->num; i++) {
        switch (ops_param->ops[i].alg_type) {
        case AlgType::Abs: {
            coat::Vec<float, width> tmp;
            std::for_each(vecs.begin(), vecs.end(), [&] (coat::Vec<float, width>* vec) {
                tmp = -0.f;
                tmp -= *vec;
                vec->max(tmp);
            });
            break;
        }
        case AlgType::Add: {
            if (inject_ops_param->params[i].layout == BinaryDataLayout::PerTensor) {
                coat::Vec<float, width> tmp;
                tmp.load(inject_ops_param->params[i].right_addrs[0], true);
                std::for_each(vecs.begin(), vecs.end(), [&] (coat::Vec<float, width>* vec) {
                    *vec += tmp;
                });
            } else if (inject_ops_param->params[i].layout == BinaryDataLayout::PerChannel) {
                coat::Vec<float, width> tmp;
                for (size_t j = 0; j < vecs.size(); j++) {
                    // TODO
                    auto addr = inject_ops_param->params[i].right_addrs[j];
                    tmp.load(addr);
                    *vecs[j] += tmp;
                }
            }
        }
    }
    }
}

// size should be runtime const
template<int vectorsize>
void jit_memset0(coat::Ptr<coat::Value<int8_t>> p, int size) {
    int offset = 0;
    int tail = size % (vectorsize * sizeof(float));
    if (size > vectorsize * (int)sizeof(float)) {
        coat::Vec<float, vectorsize> zero(true, "zero");
        const int size_4 = size / vectorsize / sizeof(float) / 4 * 4 * sizeof(float) * vectorsize;
        if (size_4 > 2 * 4 * (int)sizeof(float) * vectorsize) {
            coat::Value pos(int(0), "pos");
            coat::loop_while(pos < size_4, [&] {
                // p[pos + 1 * vectorsize * sizeof(float)]
                zero.store(p.index(pos, 0 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 1 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 2 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 3 * vectorsize * sizeof(float)));
                pos += 4 * vectorsize * sizeof(float);
            });
            offset += size_4;
        }
        for (; offset < (int)(size / (sizeof(float) * vectorsize) * sizeof(float) * vectorsize); offset += sizeof(float) * vectorsize) {
            zero.store(p[offset]);
        }
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * (int)sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    } else if (tail >= 4 * (int)sizeof(float)) {
        coat::Vec<float, vectorsize> zero(0, "zero");
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * (int)sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * (int)sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    }
    if (tail) {
        coat::Value<int64_t> zero;
        zero = 0;
        if (tail >= 8) {
            p.cast<int64_t>()[offset / 8] = zero;
            offset += 8;
            tail -= 8;
        }
        if (tail >= 4) {
            p.cast<int32_t>()[offset / 4] = coat::Value<int32_t>(zero.reg.r32());
            offset += 4;
            tail -= 4;
        }
        if (tail >= 2) {
            p.cast<int16_t>()[offset / 2] = coat::Value<int16_t>(zero.reg.r16());
            offset += 2;
            tail -= 2;
        }
        if (tail >= 1) {
            p.cast<int8_t>()[offset] = coat::Value<int8_t>(zero.reg.r8());
        }
    }
}

static void matmul_ref(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc) {
#define A(i, j) a[(j) + (i) * lda]
#define B(i, j) b[(j) + (i) * ldb]
#define C(i, j) c[(j) + (i) * ldc]

    int i, j, p;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C(i, j) = 0;//j; // post ops, per-channel
            for (p = 0; p < K; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

//using func_t = void (*)(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc);
using func_t = void (*)(float* a, float* b, float* c, JitPostOps* param);
template <unsigned width>
func_t make_brgemm(int M, int N, int K, int lda, int ldb, int ldc) {
    auto fn = coat::createFunction<func_t>();
    if constexpr (width == 16)
        fn.funcNode->frame().setAvx512Enabled();
    else if  constexpr (width == 8)
        fn.funcNode->frame().setAvxEnabled();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        lda /= sizeof(float);
        ldb /= sizeof(float);
        ldc /= sizeof(float);
        auto [j_a, j_b, j_c, j_ops] = fn.getArguments("a", "b", "c", "ops");
        const int oc_num = 3;
        const int ur_num = 8; // hardcode 8 rows
        using share_vec = std::shared_ptr<coat::Vec<float, width>>;
        std::vector<share_vec> j_weight(oc_num);
        std::vector<share_vec> j_result;
        for (int i = 0; i < oc_num; i++ ) {
            j_weight[i] = std::make_shared<share_vec::element_type>();
        }
        for (int i = 0; i < oc_num * ur_num; i++ ) {
            j_result.push_back(std::make_shared<share_vec::element_type>());
        }
        coat::Vec<float, width> j_data;

        coat::Value<int> j_m(int(0), "m");
        //for (m = 0; m < M; m += 8) {
        coat::for_loop(j_m < M / ur_num * ur_num,
            [&] {
                    j_m += ur_num;
                    j_a += ur_num * lda;
                    j_c += ur_num * ldc;
                },
            [&] {
            for (int i = 0; i < oc_num * ur_num; i++ ) {
                (*j_result[i]) = 0;
            }
            coat::Value<int> j_k(int(0), "k");
            auto j_b_row = j_b;
            auto j_a_row = j_a;
            //for (k = 0; k < K; k += width) {
            coat::for_loop(j_k < K, // TODO handle tail
                [&] {
                        j_k += width;
                        j_b_row += width * ldb;
                        j_a_row += width;
                    },
                [&] {
                for (int j = 0; j < width; j++) {
                    for (int n = 0; n < oc_num; n++) {
                        j_weight[n]->load(j_b_row[j * ldb + n * width]);
                    }
                    for (int m = 0; m < ur_num; m++) {
                        j_data.load(j_a_row[m * lda + j], true);
                        for (int n = 0; n < oc_num; n++) {
                            j_result[m * oc_num + n]->fma231(*j_weight[n], j_data);
                        }
                    }
                }
            });

            for (int m = 0; m < ur_num; m++) {
                for (int n = 0; n < oc_num; n++) {
                    j_result[m * oc_num + n]->store(j_c[m * ldc + n * width]);
                }
            }
        });

        // M tail TODO, remove repeated code
        if (M % ur_num) {
            auto ur = M % ur_num;
            for (int i = 0; i < oc_num * ur; i++ ) {
                (*j_result[i]) = 0;
            }
            coat::Value<int> j_k(int(0), "k");
            auto j_b_row = j_b;
            auto j_a_row = j_a;
            //for (k = 0; k < K; k += width) {
            coat::for_loop(j_k < K, // TODO handle tail
                [&] {
                        j_k += width;
                        j_b_row += width * ldb;
                        j_a_row += width;
                    },
                [&] {
                for (int j = 0; j < width; j++) {
                    for (int n = 0; n < oc_num; n++) {
                        j_weight[n]->load(j_b_row[j * ldb + n * width]);
                    }
                    for (int m = 0; m < ur; m++) {
                        j_data.load(j_a_row[m * lda + j], true);
                        for (int n = 0; n < oc_num; n++) {
                            j_result[m * oc_num + n]->fma231(*j_weight[n], j_data);
                        }
                    }
                }
            });

            for (int m = 0; m < ur; m++) {
                for (int n = 0; n < oc_num; n++) {
                    j_result[m * oc_num + n]->store(j_c[m * ldc + n * width]);
                }
            }
        }
        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    auto foo = fn.finalize();
    return foo;
}

// typical convolution layout
// a: aBcd16b
// b: OIhw16i16o
// c: aBcd16b
// for gemm:
// a: Km16k
// b: Nk16n
// c: Mn16M
template <unsigned width>
func_t make_block(int M, int N, int K, int lda, int ldb, int ldc) {
    auto fn = coat::createFunction<func_t>();
    if constexpr (width == 16)
        fn.funcNode->frame().setAvx512Enabled();
    else if  constexpr (width == 8)
        fn.funcNode->frame().setAvxEnabled();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        lda /= sizeof(float);
        ldb /= sizeof(float);
        ldc /= sizeof(float);
        auto [j_a, j_b, j_c, j_ops] = fn.getArguments("a", "b", "c", "ops");
        const int oc_num = 3;
        const int ur_num = 8; // hardcode 8 rows
        using share_vec = std::shared_ptr<coat::Vec<float, width>>;
        std::vector<share_vec> j_weight(oc_num);
        std::vector<share_vec> j_result;
        for (int i = 0; i < oc_num; i++ ) {
            j_weight[i] = std::make_shared<share_vec::element_type>();
        }
        for (int i = 0; i < oc_num * ur_num; i++ ) {
            j_result.push_back(std::make_shared<share_vec::element_type>());
        }
        coat::Vec<float, width> j_data;

        coat::Value<int> j_m(int(0), "m");
        //for (m = 0; m < M; m += 8) {
        coat::for_loop(j_m < M / ur_num * ur_num,
            [&] {
                    j_m += ur_num;
                    j_a += ur_num * lda;
                    j_c += ur_num * ldc;
                },
            [&] {
            for (int i = 0; i < oc_num * ur_num; i++ ) {
                (*j_result[i]) = 0;
            }
            coat::Value<int> j_k(int(0), "k");
            auto j_b_row = j_b;
            auto j_a_row = j_a;
            //for (k = 0; k < K; k += width) {
            coat::for_loop(j_k < K, // TODO handle tail
                [&] {
                        j_k += width;
                        j_b_row += width * ldb;
                        j_a_row += width * M;
                    },
                [&] {
                for (int j = 0; j < width; j++) {
                    for (int n = 0; n < oc_num; n++) {
                        j_weight[n]->load(j_b_row[j * ldb + n * width * K]);
                    }
                    for (int m = 0; m < ur_num; m++) {
                        j_data.load(j_a_row[m * lda + j], true);
                        for (int n = 0; n < oc_num; n++) {
                            j_result[m * oc_num + n]->fma231(*j_weight[n], j_data);
                        }
                    }
                }
            });

            for (int m = 0; m < ur_num; m++) {
                for (int n = 0; n < oc_num; n++) {
                    j_result[m * oc_num + n]->store(j_c[m * ldc + n * width * M]);
                }
            }
        });

        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    auto foo = fn.finalize();
    return foo;
}

void test_brgemm() {
    int M, N, K;
    M = 25600, N = 48, K = 288;
    // postops, perchannel
    std::vector<float> d(N, 0);
    std::iota(d.begin(), d.end(), 0.0f);

    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N), c_ref(M * N);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);

    PostOps post_ops;
    post_ops.num = 0;
    post_ops.ops[0].alg_type = AlgType::Abs;
    post_ops.ops[1].alg_type = AlgType::Add;
    post_ops.ops[1].binary_param.layout = BinaryDataLayout::PerChannel;
    auto f = make_brgemm<16>(M, N, K, K * 4, N * 4, N * 4);
    JitPostOps ops;
    ops.params[1].right_addr = d.data();
    f(a.data(), b.data(), c.data(), nullptr);
    auto beg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
        f(a.data(), b.data(), c.data(), nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
    std::cout << "nhwc cost " << diff / 1000.0f << " ms" << std::endl;    
    matmul_ref(a.data(), b.data(), c_ref.data(), M, N, K, K, N, N);
    if(c == c_ref) {
        printf("correct \n");
    } else {
        bool error = false;
        for (int i = 0; i < (int)c.size(); i++) {
            if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                error = true;
                printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                break;
            }
        }
        if (error)
            printf("wrong result\n");
        else
            printf("correct with minor error\n");
    }
    coat::getJitRuntimeEnv().release_func(f);
}

void reorder_nhwc2block(const std::vector<float>& in, std::vector<float>& out, int m, int n, int stride) {
    out.resize(in.size());
    for (int i = 0; i < n / stride; i++) {
        for (int j = 0; j < m; j++) {
            for (int x = 0; x < stride; x++) {
                out[i * m * stride + j * stride + x] = in[i * stride + x + j * n];
            }
        }
    }
}

void reorder_block2nhwc(const std::vector<float>& in, std::vector<float>& out, int m, int n, int stride) {
    out.resize(in.size());
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n / stride; i++) {
            for (int x = 0; x < stride; x++) {
                out[i * stride + x + j * n] = in[i * m * stride + j * stride + x];
            }
        }
    }
}

void test_block() {
    int M, N, K;
    M = 25600, N = 48, K = 288;
    // postops, perchannel
    std::vector<float> d(N, 0);
    std::iota(d.begin(), d.end(), 0.0f);

    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N), c_ref(M * N);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);

    PostOps post_ops;
    post_ops.num = 0;
    post_ops.ops[0].alg_type = AlgType::Abs;
    post_ops.ops[1].alg_type = AlgType::Add;
    post_ops.ops[1].binary_param.layout = BinaryDataLayout::PerChannel;
    auto f = make_block<16>(M, N, K, 16 * 4, 16 * 4, 16 * 4);
    JitPostOps ops;
    ops.params[1].right_addr = d.data();
    std::vector<float> a_block, b_block, c_block(M * N);
    reorder_nhwc2block(a, a_block, M, K, 16);
    reorder_nhwc2block(b, b_block, K, N, 16);
    f(a_block.data(), b_block.data(), c_block.data(), nullptr);
    auto beg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
        f(a_block.data(), b_block.data(), c_block.data(), nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
    std::cout << "block cost " << diff / 1000.0f << " ms" << std::endl;

    reorder_block2nhwc(c_block, c, M, N, 16);
    matmul_ref(a.data(), b.data(), c_ref.data(), M, N, K, K, N, N);
    if(c == c_ref) {
        printf("correct \n");
    } else {
        bool error = false;
        for (int i = 0; i < (int)c.size(); i++) {
            if (std::abs(c[i] - c_ref[i]) > 0.00001f * std::abs(c[i])) {
                error = true;
                printf("first error at %d, cur %f ref %f\n", i, c[i], c_ref[i]);
                break;
            }
        }
        if (error)
            printf("wrong result\n");
        else
            printf("correct with minor error\n");
    }
    coat::getJitRuntimeEnv().release_func(f);
}

int main() {
    test_brgemm();
    test_block();
}
