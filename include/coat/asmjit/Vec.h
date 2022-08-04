#ifndef COAT_ASMJIT_VECTOR_H_
#define COAT_ASMJIT_VECTOR_H_

#include <cassert>

#include "Ptr.h"


namespace coat {

template<typename T, unsigned width>
struct Vec final {
	using F = ::asmjit::x86::Compiler;
	using value_type = T;
	
	static_assert(sizeof(T)==1 || sizeof(T)==2 || sizeof(T)==4 || sizeof(T)==8,
		"only plain arithmetic types supported of sizes: 1, 2, 4 or 8 bytes");
	static_assert(std::is_signed_v<T> || std::is_unsigned_v<T>,
		"only plain signed or unsigned arithmetic types supported");
	static_assert(sizeof(T)*width == 128/8 || sizeof(T)*width == 256/8,
		"only 128-bit and 256-bit vector instructions supported at the moment");

	//FIXME: not a good idea when AVX512 comes into play
	using reg_type = std::conditional_t<sizeof(T)*width==128/8,
						::asmjit::x86::Xmm,
						::asmjit::x86::Ymm // only these two are allowed
					>;
	reg_type reg;


	F &cc; //FIXME: pointer stored in every value type

	Vec(F &cc, const char *name="") : cc(cc){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			reg = cc.newXmm(name);
		}else{
			// 256 bit AVX
			reg = cc.newYmm(name);
		}
	}

	inline unsigned getWidth() const { return width; }

	// load Vec from memory, always unaligned load
	Vec &operator=(Ref<Value<T>>&& src){ load(std::move(src)); return *this; }
	// load Vec from memory, always unaligned load
	void load(Ref<Value<T>>&& src){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			src.mem.setSize(16); // change to xmmword
			cc.movdqu(reg, src);
		}else{
			// 256 bit AVX
			src.mem.setSize(32); // change to ymmword
			cc.vmovdqu(reg, src);
		}
	}

	// unaligned store
	void store(Ref<Value<T>>&& dest) const {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			dest.mem.setSize(16); // change to xmmword
			cc.movdqu(dest, reg);
		}else{
			// 256 bit AVX
			dest.mem.setSize(32); // change to ymmword
			cc.vmovdqu(dest, reg);
		}
	}
	//TODO: aligned load & store

	Vec &operator+=(const Vec &other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			switch(sizeof(T)){
				case 1: cc.paddb(reg, other.reg); break;
				case 2: cc.paddw(reg, other.reg); break;
				case 4: cc.paddd(reg, other.reg); break;
				case 8: cc.paddq(reg, other.reg); break;
			}
		}else{
			// 256 bit AVX
			switch(sizeof(T)){
				case 1: cc.vpaddb(reg, reg, other.reg); break;
				case 2: cc.vpaddw(reg, reg, other.reg); break;
				case 4: cc.vpaddd(reg, reg, other.reg); break;
				case 8: cc.vpaddq(reg, reg, other.reg); break;
			}
		}
		return *this;
	}
	Vec &operator+=(Ref<Value<T>>&& other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			other.mem.setSize(16); // change to xmmword
			switch(sizeof(T)){
				case 1: cc.paddb(reg, other); break;
				case 2: cc.paddw(reg, other); break;
				case 4: cc.paddd(reg, other); break;
				case 8: cc.paddq(reg, other); break;
			}
		}else{
			// 256 bit AVX
			other.mem.setSize(32); // change to ymmword
			switch(sizeof(T)){
				case 1: cc.vpaddb(reg, reg, other); break;
				case 2: cc.vpaddw(reg, reg, other); break;
				case 4: cc.vpaddd(reg, reg, other); break;
				case 8: cc.vpaddq(reg, reg, other); break;
			}
		}
		return *this;
	}

	Vec &operator-=(const Vec &other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			switch(sizeof(T)){
				case 1: cc.psubb(reg, other.reg); break;
				case 2: cc.psubw(reg, other.reg); break;
				case 4: cc.psubd(reg, other.reg); break;
				case 8: cc.psubq(reg, other.reg); break;
			}
		}else{
			// 256 bit AVX
			switch(sizeof(T)){
				case 1: cc.vpsubb(reg, reg, other.reg); break;
				case 2: cc.vpsubw(reg, reg, other.reg); break;
				case 4: cc.vpsubd(reg, reg, other.reg); break;
				case 8: cc.vpsubq(reg, reg, other.reg); break;
			}
		}
		return *this;
	}
	Vec &operator-=(Ref<Value<T>>&& other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			other.mem.setSize(16); // change to xmmword
			switch(sizeof(T)){
				case 1: cc.psubb(reg, other); break;
				case 2: cc.psubw(reg, other); break;
				case 4: cc.psubd(reg, other); break;
				case 8: cc.psubq(reg, other); break;
			}
		}else{
			// 256 bit AVX
			other.mem.setSize(32); // change to ymmword
			switch(sizeof(T)){
				case 1: cc.vpsubb(reg, reg, other); break;
				case 2: cc.vpsubw(reg, reg, other); break;
				case 4: cc.vpsubd(reg, reg, other); break;
				case 8: cc.vpsubq(reg, reg, other); break;
			}
		}
		return *this;
	}

	Vec &operator/=(int amount){
		if(is_power_of_two(amount)){
			operator>>=(clog2(amount));
		}else{
			//TODO
			assert(false);
		}
		return *this;
	}

	Vec &operator<<=(int amount){
		static_assert(sizeof(T) > 1, "shift does not support byte element size");
		// shift left same for signed and unsigned types
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			switch(sizeof(T)){
				case 2: cc.psllw(reg, amount); break;
				case 4: cc.pslld(reg, amount); break;
				case 8: cc.psllq(reg, amount); break;
			}
		}else{
			// 256 bit AVX
			switch(sizeof(T)){
				case 2: cc.vpsllw(reg, reg, amount); break;
				case 4: cc.vpslld(reg, reg, amount); break;
				case 8: cc.vpsllq(reg, reg, amount); break;
			}
		}
		return *this;
	}
	Vec &operator<<=(const Vec &other){
		static_assert(sizeof(T) > 1, "shift does not support byte element size");
		// shift left same for signed and unsigned types
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			switch(sizeof(T)){
				case 2: cc.psllw(reg, other); break;
				case 4: cc.pslld(reg, other); break;
				case 8: cc.psllq(reg, other); break;
			}
		}else{
			// 256 bit AVX
			switch(sizeof(T)){
				case 2: cc.vpsllw(reg, reg, other); break;
				case 4: cc.vpslld(reg, reg, other); break;
				case 8: cc.vpsllq(reg, reg, other); break;
			}
		}
		return *this;
	}

	Vec &operator>>=(int amount){
		static_assert(sizeof(T) > 1, "shift does not support byte element size");
		static_assert(!(std::is_signed_v<T> && sizeof(T) == 8), "no arithmetic shift right for 64 bit values");
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			if constexpr(std::is_signed_v<T>){
				switch(sizeof(T)){
					case 2: cc.psraw(reg, amount); break;
					case 4: cc.psrad(reg, amount); break;
				}
			}else{
				switch(sizeof(T)){
					case 2: cc.psrlw(reg, amount); break;
					case 4: cc.psrld(reg, amount); break;
					case 8: cc.psrlq(reg, amount); break;
				}
			}
		}else{
			// 256 bit AVX
			if constexpr(std::is_signed_v<T>){
				switch(sizeof(T)){
					case 2: cc.vpsraw(reg, reg, amount); break;
					case 4: cc.vpsrad(reg, reg, amount); break;
				}
			}else{
				switch(sizeof(T)){
					case 2: cc.vpsrlw(reg, reg, amount); break;
					case 4: cc.vpsrld(reg, reg, amount); break;
					case 8: cc.vpsrlq(reg, reg, amount); break;
				}
			}
		}
		return *this;
	}
	Vec &operator>>=(const Vec &other){
		static_assert(sizeof(T) > 1, "shift does not support byte element size");
		static_assert(!(std::is_signed_v<T> && sizeof(T) == 8), "no arithmetic shift right for 64 bit values");
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			if constexpr(std::is_signed_v<T>){
				switch(sizeof(T)){
					case 2: cc.psraw(reg, other); break;
					case 4: cc.psrad(reg, other); break;
				}
			}else{
				switch(sizeof(T)){
					case 2: cc.psrlw(reg, other); break;
					case 4: cc.psrld(reg, other); break;
					case 8: cc.psrlq(reg, other); break;
				}
			}
		}else{
			// 256 bit AVX
			if constexpr(std::is_signed_v<T>){
				switch(sizeof(T)){
					case 2: cc.vpsraw(reg, reg, other); break;
					case 4: cc.vpsrad(reg, reg, other); break;
				}
			}else{
				switch(sizeof(T)){
					case 2: cc.vpsrlw(reg, reg, other); break;
					case 4: cc.vpsrld(reg, reg, other); break;
					case 8: cc.vpsrlq(reg, reg, other); break;
				}
			}
		}
		return *this;
	}

	Vec &operator&=(const Vec &other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			cc.pand(reg, other);
		}else{
			// 256 bit AVX
			cc.vpand(reg, reg, other);
		}
		return *this;
	}
	Vec &operator|=(const Vec &other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			cc.por(reg, other);
		}else{
			// 256 bit AVX
			cc.vpor(reg, reg, other);
		}
		return *this;
	}
	Vec &operator^=(const Vec &other){
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			cc.pxor(reg, other);
		}else{
			// 256 bit AVX
			cc.vpxor(reg, reg, other);
		}
		return *this;
	}
};

template<unsigned width>
struct Vec<float, width> final {
	using F = ::asmjit::x86::Compiler;
	using T = float;

	static_assert(sizeof(T) * width == 128 / 8 || sizeof(T) * width == 256 / 8 ||
		sizeof(T) * width == 512 / 8,
		"only 128-bit, 256-bit or 512-bit vector instructions supported at the moment");

	using reg_type = std::conditional_t<sizeof(T) * width == 128 / 8,
						asmjit::x86::Xmm, std::conditional_t<sizeof(T) * width == 256 / 8,
						asmjit::x86::Ymm,
						asmjit::x86::Zmm>
					>;
	F &cc;
	reg_type reg;

	Vec(F &cc, bool zero = false, const char *name="") : cc(cc) {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>) {
			// 128 bit SSE
			reg = cc.newXmm(name);
			if (zero)
				cc.pxor(reg, reg);
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			// 256 bit AVX
			reg = cc.newYmm(name);
			if (zero)
				cc.vpxor(reg, reg, reg);
		} else {
			reg = cc.newZmm(name);
			if (zero)
				cc.vpxor(reg, reg);
		}
	}
	Vec(const Vec& other) : cc(other.cc), reg(other.reg) {}
	Vec(F &cc, reg_type reg) : cc(cc), reg(reg) {}
	inline unsigned getWidth() const { return width; }

	Vec &operator=(float v) {
		if (v == 0) {
			if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>) {
				cc.pxor(reg, reg);
			} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
				cc.vpxor(reg, reg, reg);
			} else {
				cc.vpxor(reg, reg, reg);
			}
		} else {
			auto src = cc.newFloatConst(asmjit::ConstPool::kScopeLocal, v);
			if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>) {
				cc.movss(reg, src);
				cc.shufps(reg, reg, 0);
			} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
				cc.vbroadcastss(reg, src);
			} else {
				cc.vbroadcastss(reg, src);
			}
		}
		return *this;
	}
	// load Vec from memory, always unaligned load
	Vec &operator=(Ref<Value<T>>&& src) { load(std::move(src)); return *this; }
	// TODO: support mask
	void load(Ref<Value<T>>&& src, bool broadcast = false) {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>) {
			if (broadcast) {
				cc.movss(reg, src);
				cc.shufps(reg, reg, 0);
			} else {
				src.mem.setSize(16); // change to xmmword
				cc.movdqu(reg, src);
			}
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			if (broadcast) {
				cc.vbroadcastss(reg, src);
			} else {
				src.mem.setSize(32); // change to ymmword
				cc.vmovdqu(reg, src);
			}
		} else {
			if (broadcast) {
				cc.vbroadcastss(reg, src);
			} else {
				src.mem.setSize(64); // change to zmmword
				cc.vmovdqu(reg, src);
			}
		}
	}
	// unaligned store
	void store(Ref<Value<T>>&& dest) const {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			dest.mem.setSize(16); // change to xmmword
			cc.movdqu(dest, reg);
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			// 256 bit AVX
			dest.mem.setSize(32); // change to ymmword
			cc.vmovdqu(dest, reg);
		} else {
			dest.mem.setSize(64); // change to zmmword
			cc.vmovdqu(dest, reg);			
		}
	}
	void store(Ref<Value<int8_t>>&& dest) const {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			dest.mem.setSize(16); // change to xmmword
			cc.movdqu(dest, reg);
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			// 256 bit AVX
			dest.mem.setSize(32); // change to ymmword
			cc.vmovdqu(dest, reg);
		} else {
			dest.mem.setSize(64); // change to zmmword
			cc.vmovdqu(dest, reg);			
		}
	}	
	void load_aligned(Ref<Value<T>>&& src, bool broadcast = false) {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>) {
			if (broadcast) {
				cc.movss(reg, src);
				cc.shufps(reg, reg, 0);
			} else {
				src.mem.setSize(16); // change to xmmword
				cc.movdqa(reg, src);
			}
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			if (broadcast) {
				cc.vbroadcastss(reg, src);
			} else {
				src.mem.setSize(32); // change to ymmword
				cc.vmovdqa(reg, src);
			}
		} else {
			if (broadcast) {
				cc.vbroadcastss(reg, src);
			} else {
				src.mem.setSize(64); // change to zmmword
				cc.vmovdqa(reg, src);
			}
		}
	}
	void store_aligned(Ref<Value<T>>&& dest) const {
		if constexpr(std::is_same_v<reg_type,::asmjit::x86::Xmm>){
			// 128 bit SSE
			dest.mem.setSize(16); // change to xmmword
			cc.movdqa(dest, reg);
		} else if constexpr(std::is_same_v<reg_type,::asmjit::x86::Ymm>) {
			// 256 bit AVX
			dest.mem.setSize(32); // change to ymmword
			cc.vmovdqa(dest, reg);
		} else {
			dest.mem.setSize(64); // change to zmmword
			cc.vmovdqa(dest, reg);			
		}
	}	
	Vec& add(const Vec& other) {
		cc.vaddps(reg, reg, other.reg);
		return *this;
	}
	Vec& add(Ref<Value<T>>&& other) {
		cc.vaddps(reg, reg, other);
		return *this;
	}
	Vec& sub(const Vec& other){
		cc.vsubps(reg, reg, other.reg);
		return *this;
	}
	Vec& sub(Ref<Value<T>>&& other) {
		cc.vsubps(reg, reg, other);
		return *this;
	}
	Vec& mul(const Vec& other) {
		cc.vmulps(reg, reg, other.reg);
		return *this;
	}
	Vec& mul(Ref<Value<T>>&& other) {
		cc.vmulps(reg, reg, other);
		return *this;
	}
	Vec& div(const Vec& other) {
		cc.vdivps(reg, reg, other.reg);
		return *this;
	}
	Vec& div(Ref<Value<T>>&& other) {
		cc.vdivps(reg, reg, other);
		return *this;
	}
	Vec& fma231(const Vec& x, const Vec& y) {
		cc.vfmadd231ps(reg, x.reg, y.reg);
		return *this;
	}
	Vec& fma231(const Vec& x, const Ref<Value<T>>&& y) {
		cc.vfmadd231ps(reg, x.reg, y);
		return *this;
	}	
	Vec& operator+=(const Vec& other) {
		return add(other);
	}
	Vec& operator+=(Ref<Value<T>>&& other) {
		return add(other);
	}
	Vec& operator-=(const Vec& other){
		return sub(other);
	}
	Vec& operator-=(Ref<Value<T>>&& other) {
		return sub(other);
	}
	Vec& operator*=(const Vec& other) {
		return mul(other);
	}
	Vec& operator*=(Ref<Value<T>>&& other) {
		return mul(other);
	}
	Vec& operator/=(const Vec& other) {
		return div(other);
	}
	Vec& operator/=(Ref<Value<T>>&& other) {
		return div(other);
	}
};

template<int width, typename T>
Vec<T, width> make_vector(::asmjit::x86::Compiler &cc, Ref<Value<T>>&& src){
	Vec<T, width> v(cc);
	v = std::move(src);
	return v;
}

} // namespace

#endif
