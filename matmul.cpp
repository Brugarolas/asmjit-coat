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
		x(float*, xx)	\
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
		x(float, tt)	  \
		x(float*, src)    \
		x(float*, weight) \
		x(float*, dst)    \
		x(size_t, size)   \
		x(BinayParamArr, bin) \
        x(float*, buf)

	COAT_DECLARE_PRIVATE(MEMBERS)
	#undef MEMBERS
};

// size should be runtime const
template<int vectorsize>
void jit_memset0(asmjit::x86::Compiler &cc, coat::Ptr<coat::Value<int8_t>> p, size_t size) {
    size_t offset = 0;
    size_t tail = size % (vectorsize * sizeof(float));
    //auto p = ptr.cast<float>();
    if (size > vectorsize * sizeof(float)) {
        coat::Vec<float, vectorsize> zero(cc, true, "zero");
        const auto size_4 = size / vectorsize / sizeof(float) / 4 * 4 * sizeof(float) * vectorsize;
        if (size_4 > 2 * 4 * sizeof(float) * vectorsize) {
            coat::Value pos(cc, uint64_t(0), "pos");
            coat::loop_while(cc, pos < size_4, [&] {
                // p[pos + 1 * vectorsize * sizeof(float)]
                zero.store(p.index(pos, 0 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 1 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 2 * vectorsize * sizeof(float)));
                zero.store(p.index(pos, 3 * vectorsize * sizeof(float)));
                pos += 4 * vectorsize * sizeof(float);
            });
            offset += size_4;
        }
        for (; offset < size / (sizeof(float) * vectorsize) * sizeof(float) * vectorsize; offset += sizeof(float) * vectorsize) {
            zero.store(p[offset]);
        }
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.cc, zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.cc, zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.cc, zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    } else if (tail >= 4 * sizeof(float)) {
        coat::Vec<float, vectorsize> zero(cc, 0, "zero");
        if constexpr(vectorsize >= 16) {
            if (tail >= 8 * sizeof(float)) {
                coat::Vec<float, 8> zero_y(zero.cc, zero.reg.half());
                zero_y.store(p[offset]);
                offset += 8 * sizeof(float);
                tail -= 8 * sizeof(float);
            }
            if (tail >= 4 * sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.cc, zero.reg.half().half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        } else if constexpr(vectorsize >= 8) {
            if (tail >= 4 * sizeof(float)) {
                coat::Vec<float, 4> zero_y(zero.cc, zero.reg.half());
                zero_y.store(p[offset]);
                offset += 4 * sizeof(float);
                tail -= 4 * sizeof(float);
            }
        }
    }
    if (tail) {
        coat::Value<int64_t> zero(cc);
        zero = 0;
        if (tail >= 8) {
            p.cast<int64_t>()[offset / 8] = zero;
            offset += 8;
            tail -= 8;
        }
        if (tail >= 4) {
            p.cast<int32_t>()[offset / 4] = coat::Value<int32_t>(zero.cc, zero.reg.r32());
            offset += 4;
            tail -= 4;
        }
        if (tail >= 2) {
            p.cast<int16_t>()[offset / 2] = coat::Value<int16_t>(zero.cc, zero.reg.r16());
            offset += 2;
            tail -= 2;
        }
        if (tail >= 1) {
            p.cast<int8_t>()[offset] = coat::Value<int8_t>(zero.cc, zero.reg.r8());
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
            C(i, j) = 0;
            for (p = 0; p < K; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

coat::runtimeasmjit asmrt;
//using func_t = void (*)(float* a, float* b, float* c, int M, int N, int K, int lda, int ldb, int ldc);
using func_t = void (*)(float* a, float* b, float* c);
template <unsigned width>
func_t makeMatmul(int M, int N, int K, int lda, int ldb, int ldc) {
	// initialize backend, AsmJit in this case
	// context object representing the generated function
	auto fn = asmrt.createFunction<func_t>();
#if ENABLE_DUMP
	fn.enableCodeDump();
#endif
	{
		auto [a, b, c] = fn.getArguments("a", "b", "c");
        jit_memset0<width>(fn.cc, c.cast<int8_t>(), M * N * sizeof(float));
        coat::Vec<float, width> regCi0(fn.cc), regCi1(fn.cc), regCi2(fn.cc), regCi3(fn.cc);
        coat::Vec<float, width> regA0i0(fn.cc), regA0i1(fn.cc), regA0i2(fn.cc), regA0i3(fn.cc), regB0(fn.cc);
        coat::Vec<float, width> regA1i0(fn.cc), regA1i1(fn.cc), regA1i2(fn.cc), regA1i3(fn.cc), regB1(fn.cc);
        coat::Vec<float, width> regA2i0(fn.cc), regA2i1(fn.cc), regA2i2(fn.cc), regA2i3(fn.cc), regB2(fn.cc);
        coat::Vec<float, width> regA3i0(fn.cc), regA3i1(fn.cc), regA3i2(fn.cc), regA3i3(fn.cc), regB3(fn.cc);

        coat::Value i(fn.cc, int(0), "i");
        auto m_a = a;
        auto m_c = c;
        coat::loop_while(fn.cc, i < M, [&] {
        //for (i = 0; i < params_c.m / 4 * 4; i += 4) {
            coat::Value p(fn.cc, int(0), "p");
            auto m_b = b;
            auto p_a = m_a;
            coat::loop_while(fn.cc, p < K, [&] {
            //for (p = 0; p < params_c.k; p += 4) { // TODO: handle tail
                // p[pos + 1 * vectorsize * sizeof(float)]
                //zero.store(p.index(pos, 0 * vectorsize * sizeof(float)));

                regA0i0.load(p_a[0], true);
                regA1i0.load(p_a[1], true);
                regA2i0.load(p_a[2], true);
                regA3i0.load(p_a[3], true);
                regA0i1.load(p_a[0 + 1 * lda / sizeof(float)], true);
                regA1i1.load(p_a[1 + 1 * lda / sizeof(float)], true);
                regA2i1.load(p_a[2 + 1 * lda / sizeof(float)], true);
                regA3i1.load(p_a[3 + 1 * lda / sizeof(float)], true);
                // regA0i2.load(m_a.index(p, 0 * sizeof(float) + 2 * lda), true);
                // regA1i2.load(m_a.index(p, 1 * sizeof(float) + 2 * lda), true);
                // regA2i2.load(m_a.index(p, 2 * sizeof(float) + 2 * lda), true);
                // regA3i2.load(m_a.index(p, 3 * sizeof(float) + 2 * lda), true);
                // regA0i3.load(m_a.index(p, 0 * sizeof(float) + 3 * lda), true);
                // regA1i3.load(m_a.index(p, 1 * sizeof(float) + 3 * lda), true);
                // regA2i3.load(m_a.index(p, 2 * sizeof(float) + 3 * lda), true);
                // regA3i3.load(m_a.index(p, 3 * sizeof(float) + 3 * lda), true);
                coat::Value j(fn.cc, int(0), "j");
                coat::loop_while(fn.cc, j < N, [&] {
                //for (j = 0; j < params_c.n; j += width) { // TODO: handle tail
                    // ptrC0 = &C(i, j);
                    // ptrC1 = &C(i + 1, j);
                    // ptrC2 = &C(i + 2, j);
                    // ptrC3 = &C(i + 3, j);
                    regCi0.load(m_c.index(j, 0));
                    regCi1.load(m_c.index(j, 1 * ldc));
                    // regCi2.load(m_c.index(j, 2 * ldc));
                    // regCi3.load(m_c.index(j, 3 * ldc));

                    regB0.load(m_b.index(j, 0));
                    regB1.load(m_b.index(j, 1 * ldb));
                    regB2.load(m_b.index(j, 2 * ldb));
                    regB3.load(m_b.index(j, 3 * ldb));
                    regCi0 += (regA0i0 * regB0 + regA1i0 * regB1) + 
                            (regA2i0 * regB2 + regA3i0 * regB3);
                    regCi1 += (regA0i1 * regB0 + regA1i1 * regB1) + 
                            (regA2i1 * regB2 + regA3i1 * regB3);
                    // regCi2 += regA0i2 * regB0 + regA1i2 * regB1 + 
                    //         regA2i2 * regB2 + regA3i2 * regB3;
                    // regCi3 += regA0i3 * regB0 + regA1i3 * regB1 + 
                    //         regA2i3 * regB2 + regA3i3 * regB3;
                    regCi0.store(m_c.index(j, 0));
                    regCi1.store(m_c.index(j, 1 * ldc));
                    // regCi2.store(m_c.index(j, 2 * ldc));
                    // regCi3.store(m_c.index(j, 3 * ldc));
                    j += width;
                });
                p += 4;
                m_b += 4 * ldb / sizeof(float);
                p_a += 4;
            });
            i += 2;
            m_a += 2 * lda / sizeof(float);
            m_c += 2 * ldc / sizeof(float);
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
		// specify return value
		coat::ret(fn);
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
void test_struct() {
	printf("start test struct...\n");
	std::vector<float> array_x(16);
	std::vector<float> array_y(16);
	std::vector<float> array_z(16);
    std::vector<float> buf(128+63, 1);
	std::iota(array_x.begin(), array_x.end(), 0.0f);
	std::iota(array_y.begin(), array_y.end(), 1.0f);
	std::vector<float> expected(16);
	using func_t = void (*)(ConvParam*);

	// initialize backend, AsmJit in this case
	coat::runtimeasmjit asmrt;
	// context object representing the generated function
	auto fn = asmrt.createFunction<func_t>();
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
        auto buf_p = param.get_value<ConvParam::member_buf>("buf");
		
        jit_memset0<8>(fn.cc, buf_p.cast<int8_t>(), buf.size() * 4);
		static const int vectorsize = 8;
		coat::Value pos(fn, uint64_t(0), "pos");
		coat::do_while(fn, [&]{
			auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
			auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
			x0 += y0;
			x0.store(z[pos]);
			for (size_t i = 0; i < expected.size(); i++) {
				expected[i] = array_x[i] + array_y[i];
			}
			// move to next vector
			pos += vectorsize;
		}, pos < size);

		// specify return value
		coat::ret(fn);
	}

	// finalize code generation and get function pointer to the generated function
	func_t foo = fn.finalize();

	// execute the generated function
	ConvParam param;
	param.src = array_x.data();
	param.weight = array_y.data();
	param.dst = array_z.data();
	param.size = array_x.size();
    param.buf = buf.data();
	param.bin[1].p = array_x.data();
	foo(&param);
	// print result
	if(array_z == expected) {
		printf("correct \n");
	} else {
		printf("wrong result\n");
	}
}

void test_matmul() {
    int M, N, K;
    M = N = K = 400;
    std::vector<float> a(M * K, 2), b(K * N, 1), c(M * N), c_ref(M * N);
    auto f = makeMatmul<8>(M, N, K, M * 4, N * 4, M * 4);
    f(a.data(), b.data(), c.data());
    matmul_ref(a.data(), b.data(), c_ref.data(), M, N, K, M, N, M);
    if(c == c_ref) {
		printf("correct \n");
	} else {
		printf("wrong result\n");
	}
}
int main(){
	test_matmul();
}
