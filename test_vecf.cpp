#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

#include <coat/Function.h>
#include <coat/ControlFlow.h>
#include <coat/Vec.h>

#define ENABLE_DUMP 1

enum class compute_type {
	add,
	sub,
	mul,
	div,
	fma
};
void test_compute(compute_type type) {
	printf("start test vec float type %d...\n", (int)type);
	std::vector<float> array_x(16);
	std::vector<float> array_y(16);
	std::vector<float> array_z(16);
	std::iota(array_x.begin(), array_x.end(), 0.0f);
	std::iota(array_y.begin(), array_y.end(), 1.0f);
	std::vector<float> expected(16);
	using func_t = void (*)(float* x, float* y, uint64_t size, float* z);

	// initialize backend, AsmJit in this case
	coat::runtimeasmjit asmrt;
	// context object representing the generated function
	auto fn = asmrt.createFunction<func_t>();
#if ENABLE_DUMP
	fn.enableCodeDump();
#endif
	{
		// get function arguments as "meta-variables"
		auto [x, y, size, z] = fn.getArguments("x", "y", "size", "z");

		static const int vectorsize = 8;
		coat::Value pos(fn, uint64_t(0), "pos");
		coat::do_while(fn, [&]{
			switch (type)
			{
			case compute_type::add:
			{
				auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
				auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
				x0 += y0;
				x0.store(z[pos]);
				for (size_t i = 0; i < expected.size(); i++) {
					expected[i] = array_x[i] + array_y[i];
				}
				break;
			}
			case compute_type::sub:
			{
				auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
				auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
				x0 -= y0;
				x0.store(z[pos]);
				for (size_t i = 0; i < expected.size(); i++) {
					expected[i] = array_x[i] - array_y[i];
				}
				break;
			}
			case compute_type::mul:
			{
				auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
				auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
				x0 *= y0;
				x0.store(z[pos]);
				for (size_t i = 0; i < expected.size(); i++) {
					expected[i] = array_x[i] * array_y[i];
				}
				break;
			}
			case compute_type::div:
			{
				auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
				auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
				x0 /= y0;
				x0.store(z[pos]);
				for (size_t i = 0; i < expected.size(); i++) {
					expected[i] = array_x[i] / array_y[i];
				}
				break;
			}
			case compute_type::fma:
			{
				auto s = coat::Vec<float, vectorsize>(fn);
				s = 10.0f;
				auto x0 = coat::make_vector<vectorsize>(fn, x[pos]);
				auto y0 = coat::make_vector<vectorsize>(fn, y[pos]);
				s.fma231(x0, y0);
				s.store(z[pos]);
				for (size_t i = 0; i < expected.size(); i++) {
					expected[i] = array_x[i] * array_y[i] + 10.0f;
				}
				break;
			}
			default:
				break;
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
	foo(array_x.data(), array_y.data(), array_z.size(), array_z.data());

	// print result
	if(array_z == expected) {
		printf("type: %d correct \n", (int)type);
	} else {
		printf("type: %d wrong result\n", (int)type);
	}
}

int main(){
	test_compute(compute_type::add);
	test_compute(compute_type::sub);
	test_compute(compute_type::mul);
	test_compute(compute_type::div);
	test_compute(compute_type::fma);
}
