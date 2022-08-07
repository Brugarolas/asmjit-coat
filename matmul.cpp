#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>

#define ENABLE_DUMP 1

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
            C(i, j) = j; // post ops, per-channel
            for (p = 0; p < K; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

//using func_t = void (*)(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc);
using func_t = void (*)(float* a, float* b, float* c, JitPostOps* param);
template <unsigned width>
func_t make_matmul(int M, int N, int K, int lda, int ldb, int ldc, PostOps* post_ops_param) {
    // initialize backend, AsmJit in this case
    // context object representing the generated function
    auto fn = coat::createFunction<func_t>();
    if constexpr (width == 16)
        fn.funcNode->frame().setAvx512Enabled();
    else if  constexpr (width == 8)
        fn.funcNode->frame().setAvxEnabled();
#if ENABLE_DUMP
    fn.enableCodeDump();
#endif
    {
        auto [a, b, c, jit_post_ops_param] = fn.getArguments("a", "b", "c", "ops");
        jit_memset0<width>(c.cast<int8_t>(), M * N * (int)sizeof(float));
        coat::Vec<float, width> regCi0, regCi1;
        coat::Vec<float, width> regA0i0, regA0i1, regB0;
        coat::Vec<float, width> regA1i0, regA1i1, regB1;
        coat::Vec<float, width> regA2i0, regA2i1, regB2;
        coat::Vec<float, width> regA3i0, regA3i1, regB3;
        JitInjectPostOps inject_postops_param;
        using share_p = std::shared_ptr<coat::Ptr<coat::Value<float>>>;
        std::vector<share_p> post_ops_addrs;
        for (auto i = 0; i < post_ops_param->num; i++) {
            if (post_ops_param->ops[i].alg_type >= AlgType::Add) {
                auto params = jit_post_ops_param.get_value<JitPostOps::member_params>("params");
                auto addr = params[i].get_value<JitParam::member_right_addr>("addr");
                // TODO: ptr has no 'operator= addr'
                auto op = std::make_shared<share_p::element_type>(addr);
                post_ops_addrs.push_back(op);
            } else {
                auto op = std::make_shared<share_p::element_type>();
                post_ops_addrs.push_back(op);
            }
        }

        auto prepare_inject_param = [&] (const coat::Value<int>& n, int vec_num) {
            for (auto i = 0; i < post_ops_param->num; i++) {
                if (post_ops_param->ops[i].alg_type >= AlgType::Add && post_ops_param->ops[i].binary_param.layout != BinaryDataLayout::PerTensor) {
                    auto& ptr = *post_ops_addrs[i];
                    inject_postops_param.params[i].right_addrs.clear();
                    inject_postops_param.params[i].layout = post_ops_param->ops[i].binary_param.layout;
                    for (int j = 0; j < vec_num; j++)
                        inject_postops_param.params[i].right_addrs.push_back(ptr.index(n, width * 0 * sizeof(float)));
                }
            }
        };

        coat::Value i(int(0), "i");
        auto m_a = a;
        auto m_c = c;
        //for (i = 0; i < params_c.m / 4 * 4; i += 4) {
        coat::for_loop(i < M, [&] {
                i += 2;
                m_a += 2 * lda / sizeof(float);
                m_c += 2 * ldc / sizeof(float);
            },
            [&] {
            coat::Value<int> p(int(0), "p");
            auto m_b = b;
            auto p_a = m_a;
            //for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
            coat::for_loop(p < K,
                [&] {
                        p += 4;
                        m_b += 4 * ldb / sizeof(float);
                        p_a += 4;
                    },
                [&] {
                regA0i0.load(p_a[0], true);
                regA1i0.load(p_a[1], true);
                regA2i0.load(p_a[2], true);
                regA3i0.load(p_a[3], true);
                regA0i1.load(p_a[0 + 1 * lda / sizeof(float)], true);
                regA1i1.load(p_a[1 + 1 * lda / sizeof(float)], true);
                regA2i1.load(p_a[2 + 1 * lda / sizeof(float)], true);
                regA3i1.load(p_a[3 + 1 * lda / sizeof(float)], true);
                coat::Value<int> j(int(0), "j");
                //for (j = 0; j < params_c.n; j += width) { // TODO: handle tail
                coat::for_loop(j < N, [&] { j += width; }, [&] {
                    regCi0.load(m_c.index(j, 0));
                    regCi1.load(m_c.index(j, 1 * ldc));

                    regB0.load(m_b.index(j, 0));
                    regB1.load(m_b.index(j, 1 * ldb));
                    regB2.load(m_b.index(j, 2 * ldb));
                    regB3.load(m_b.index(j, 3 * ldb));
                    coat::Vec<float, width> tmp0(regA0i0);
                    coat::Vec<float, width> tmp1(regA1i0);
                    tmp0 *= regB0;
                    tmp1 *= regB1;
                    regCi0 += tmp0;
                    regCi0 += tmp1;
                    tmp0 = regA2i0;
                    tmp1 = regA3i0;
                    tmp0 *= regB2;
                    tmp1 *= regB3;
                    regCi0 += tmp0;
                    regCi0 += tmp1;
                    // regCi0 += (regA0i0 * regB0 + regA1i0 * regB1) + 
                    //         (regA2i0 * regB2 + regA3i0 * regB3);
                    tmp0 = regA0i1;
                    tmp1 = regA1i1;
                    tmp0 *= regB0;
                    tmp1 *= regB1;
                    regCi1 += tmp0;
                    regCi1 += tmp1;
                    tmp0 = regA2i1;
                    tmp1 = regA3i1;
                    tmp0 *= regB2;
                    tmp1 *= regB3;
                    regCi1 += tmp0;
                    regCi1 += tmp1;
                    // regCi1 += (regA0i1 * regB0 + regA1i1 * regB1) + 
                    //         (regA2i1 * regB2 + regA3i1 * regB3);
                    regCi0.store(m_c.index(j, 0));
                    regCi1.store(m_c.index(j, 1 * ldc));
                });
            });
        });
        /*for (; i < params_c.m; i++) {
            for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
                regA0i0 = b_t<Type, Arch>(A(i, p + 0));
                regA1i0 = b_t<Type, Arch>(A(i, p + 1));
                regA2i0 = b_t<Type, Arch>(A(i, p + 2));
                regA3i0 = b_t<Type, Arch>(A(i, p + 3));
                for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                    ptrC0 = &C(i, j);
                    regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                    regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                    regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                    regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                    regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                    regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                            regA2i0 * regB2 + regA3i0 * regB3;
                    regCi0.store_unaligned(ptrC0);
                }
            }
        }*/

        // inject binary postops
        m_c = c;
        i = 0;
        //for (i = 0; i < params_c.m / 4 * 4; i += 4) {
        coat::for_loop(i < M, [&] {
                i += 2;
                m_c += 2 * ldc / sizeof(float);
            },
            [&] {
            coat::Value<int> j(int(0), "j");
            //for (j = 0; j < params_c.n; j += width) { // TODO: handle tail
            coat::for_loop(j < N, [&] { j += width; }, [&] {
                regCi0.load(m_c.index(j, 0));
                regCi1.load(m_c.index(j, 1 * ldc));
                prepare_inject_param(j, 2);
                inject_postops<width>({ &regCi0, &regCi1 }, post_ops_param, &inject_postops_param);
                regCi0.store(m_c.index(j, 0));
                regCi1.store(m_c.index(j, 1 * ldc));
            });
        });
        // specify return value
        coat::ret();
    }

    // finalize code generation and get function pointer to the generated function
    func_t foo = fn.finalize();
    return foo;
}
/*
void matmulT(Arch*, const MatmulConstParam params_c, const MatmulMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m) {
    constexpr std::size_t inc = b_t<Type, Arch>::size;

#define A(i, j) a[(j) + (i) * params_c.x_stride]
#define B(i, j) b[(j) + (i) * params_c.y_stride]
#define C(i, j) c[(j) + (i) * params_c.dst_stride]

    const auto *a = (Type*)params_m.src_x;
    const auto *b = (Type*)params_m.src_y;
    auto *c = (Type*)params_m.dst_d;
    int i, j, p;
    memset(c, 0, params_c.m * params_c.n * sizeof(Type));

    Type *ptrC0, *ptrC1, *ptrC2, *ptrC3;
    b_t<Type, Arch> regCi0, regCi1, regCi2, regCi3;
    b_t<Type, Arch> regA0i0, regA0i1, regA0i2, regA0i3, regB0;
    b_t<Type, Arch> regA1i0, regA1i1, regA1i2, regA1i3, regB1;
    b_t<Type, Arch> regA2i0, regA2i1, regA2i2, regA2i3, regB2;
    b_t<Type, Arch> regA3i0, regA3i1, regA3i2, regA3i3, regB3;

    for (i = 0; i < params_c.m / 4 * 4; i += 4) {
        for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));
            regA0i1 = b_t<Type, Arch>(A(i + 1, p + 0));
            regA1i1 = b_t<Type, Arch>(A(i + 1, p + 1));
            regA2i1 = b_t<Type, Arch>(A(i + 1, p + 2));
            regA3i1 = b_t<Type, Arch>(A(i + 1, p + 3));
            regA0i2 = b_t<Type, Arch>(A(i + 2, p + 0));
            regA1i2 = b_t<Type, Arch>(A(i + 2, p + 1));
            regA2i2 = b_t<Type, Arch>(A(i + 2, p + 2));
            regA3i2 = b_t<Type, Arch>(A(i + 2, p + 3));
            regA0i3 = b_t<Type, Arch>(A(i + 3, p + 0));
            regA1i3 = b_t<Type, Arch>(A(i + 3, p + 1));
            regA2i3 = b_t<Type, Arch>(A(i + 3, p + 2));
            regA3i3 = b_t<Type, Arch>(A(i + 3, p + 3));
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                ptrC1 = &C(i + 1, j);
                ptrC2 = &C(i + 2, j);
                ptrC3 = &C(i + 3, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regCi1 = b_t<Type, Arch>::load(ptrC1, xsimd::unaligned_mode());
                regCi2 = b_t<Type, Arch>::load(ptrC2, xsimd::unaligned_mode());
                regCi3 = b_t<Type, Arch>::load(ptrC3, xsimd::unaligned_mode());

                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3;
                regCi1 += regA0i1 * regB0 + regA1i1 * regB1 + 
                          regA2i1 * regB2 + regA3i1 * regB3;
                regCi2 += regA0i2 * regB0 + regA1i2 * regB1 + 
                          regA2i2 * regB2 + regA3i2 * regB3;
                regCi3 += regA0i3 * regB0 + regA1i3 * regB1 + 
                          regA2i3 * regB2 + regA3i3 * regB3;
                regCi0.store_unaligned(ptrC0);
                regCi1.store_unaligned(ptrC1);
                regCi2.store_unaligned(ptrC2);
                regCi3.store_unaligned(ptrC3);
            }
        }
    }
    for (; i < params_c.m; i++) {
        for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
            regA0i0 = b_t<Type, Arch>(A(i, p + 0));
            regA1i0 = b_t<Type, Arch>(A(i, p + 1));
            regA2i0 = b_t<Type, Arch>(A(i, p + 2));
            regA3i0 = b_t<Type, Arch>(A(i, p + 3));
            for (j = 0; j < params_c.n; j += inc) { // TODO: handle tail
                ptrC0 = &C(i, j);
                regCi0 = b_t<Type, Arch>::load(ptrC0, xsimd::unaligned_mode());
                regB0 = b_t<Type, Arch>::load(&B(p + 0, j), xsimd::unaligned_mode());
                regB1 = b_t<Type, Arch>::load(&B(p + 1, j), xsimd::unaligned_mode());
                regB2 = b_t<Type, Arch>::load(&B(p + 2, j), xsimd::unaligned_mode());
                regB3 = b_t<Type, Arch>::load(&B(p + 3, j), xsimd::unaligned_mode());
                regCi0 += regA0i0 * regB0 + regA1i0 * regB1 + 
                          regA2i0 * regB2 + regA3i0 * regB3;
                regCi0.store_unaligned(ptrC0);
            }
        }
    }

    // fuse operation
    const auto *dst_d = (uint8_t*)params_m.dst_d;
    i = 0;
    const auto size = params_c.m * params_c.n;
    for (; i < size / inc; i++) {
        // TODO: add type convert here
        auto x = b_t<Type, Arch>::load((Type*)(dst_d + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto d = seq_fuse<Type, Arch>(x, i, fuse_params_m, fuse_params_c);
        // TODO: add type convert here
        d.store_unaligned((Type*)(dst_d + i * inc * sizeof(Type)));
    }
    if (size % inc) {
        Type buf[inc];
        // TODO: FIXME: malloc more data at least align simd width
        auto x = b_t<Type, Arch>::load((Type*)(dst_d + i * inc * sizeof(Type)), xsimd::unaligned_mode());
        auto d = seq_fuse<Type, Arch>(x, i, fuse_params_m, fuse_params_c);
        d.store_unaligned(buf);
        memcpy((void*)(dst_d + i * inc * sizeof(Type)), (void*)buf, (size % inc) * sizeof(Type));
    }
#undef A
#undef B
#undef C
}
*/

void test_matmul() {
    int M, N, K;
    M = N = K = 640;
    // postops, perchannel
    std::vector<float> d(N, 0);
    std::iota(d.begin(), d.end(), 0.0f);

    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N), c_ref(M * N);
    PostOps post_ops;
    post_ops.num = 2;
    post_ops.ops[0].alg_type = AlgType::Abs;
    post_ops.ops[1].alg_type = AlgType::Add;
    post_ops.ops[1].binary_param.layout = BinaryDataLayout::PerChannel;
    auto f = make_matmul<8>(M, N, K, M * 4, N * 4, M * 4, &post_ops);
    JitPostOps ops;
    ops.params[1].right_addr = d.data();
    f(a.data(), b.data(), c.data(), &ops);
    matmul_ref(a.data(), b.data(), c_ref.data(), M, N, K, M, N, M);
    if(c == c_ref) {
        printf("correct \n");
    } else {
        printf("wrong result\n");
    }
    coat::getJitRuntimeEnv().release_func(f);
}
int main(){
    test_matmul();
}
