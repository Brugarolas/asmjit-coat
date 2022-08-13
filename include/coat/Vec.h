#pragma once

#include <cassert>
#include "Global.h"
#include "Ptr.h"

namespace coat {

// Note:
// 1, Constructor: 
//    Default constructor: allocate a Xmm register
//    Construct from Xmm: _reuse_/share the Xmm -- base need add
//    Copy constructor: allocate a Xmm register, copy value -- base need add
// 2, Assign:
//    Assign from Ref
//    Assign from int/float -- base need add
//    Assign from Vec -- base need add
template<typename T, unsigned width>
struct Vec final {
    using value_type = T;
    
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
        "only plain arithmetic types supported of sizes: 1, 2, 4 or 8 bytes");
    static_assert(std::is_signed_v<T> || std::is_unsigned_v<T>,
        "only plain signed or unsigned arithmetic types supported");
    static_assert(sizeof(T) * width == 128 / 8 || sizeof(T) * width == 256 / 8,
        "only 128-bit and 256-bit vector instructions supported at the moment");

    //FIXME: not a good idea when AVX512 comes into play
    using reg_type = std::conditional_t<sizeof(T) * width == 128 / 8,
                        asmjit::x86::Xmm,
                        asmjit::x86::Ymm // only these two are allowed
                    >;
    reg_type reg;

    Vec(const char* name="") {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            reg = _CC.newXmm(name);
        } else {
            // 256 bit AVX
            reg = _CC.newYmm(name);
        }
    }

    inline unsigned getWidth() const { return width; }

    // load Vec from memory, always unaligned load
    Vec& operator=(Ref<Value<T>>&& src) { load(std::move(src)); return *this; }
    // load Vec from memory, always unaligned load
    void load(Ref<Value<T>>&& src) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            src.mem.setSize(16); // change to xmmword
            _CC.movdqu(reg, src);
        } else {
            // 256 bit AVX
            src.mem.setSize(32); // change to ymmword
            _CC.vmovdqu(reg, src);
        }
    }
    // load Vec from memory, always unaligned load
    Vec& operator=(Ref<Value<T>>& src) { load(src); return *this; }
    // load Vec from memory, always unaligned load
    void load(Ref<Value<T>>& src) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            src.mem.setSize(16); // change to xmmword
            _CC.movdqu(reg, src);
        } else {
            // 256 bit AVX
            src.mem.setSize(32); // change to ymmword
            _CC.vmovdqu(reg, src);
        }
    }

    // unaligned store
    void store(Ref<Value<T>>&& dest) const {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            dest.mem.setSize(16); // change to xmmword
            _CC.movdqu(dest, reg);
        } else {
            // 256 bit AVX
            dest.mem.setSize(32); // change to ymmword
            _CC.vmovdqu(dest, reg);
        }
    }
    //TODO: aligned load&  store

    Vec& operator+=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            switch(sizeof(T)) {
                case 1: _CC.paddb(reg, other.reg); break;
                case 2: _CC.paddw(reg, other.reg); break;
                case 4: _CC.paddd(reg, other.reg); break;
                case 8: _CC.paddq(reg, other.reg); break;
            }
        } else {
            // 256 bit AVX
            switch(sizeof(T)) {
                case 1: _CC.vpaddb(reg, reg, other.reg); break;
                case 2: _CC.vpaddw(reg, reg, other.reg); break;
                case 4: _CC.vpaddd(reg, reg, other.reg); break;
                case 8: _CC.vpaddq(reg, reg, other.reg); break;
            }
        }
        return *this;
    }
    Vec& operator+=(Ref<Value<T>>&& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            other.mem.setSize(16); // change to xmmword
            switch(sizeof(T)) {
                case 1: _CC.paddb(reg, other); break;
                case 2: _CC.paddw(reg, other); break;
                case 4: _CC.paddd(reg, other); break;
                case 8: _CC.paddq(reg, other); break;
            }
        } else {
            // 256 bit AVX
            other.mem.setSize(32); // change to ymmword
            switch(sizeof(T)) {
                case 1: _CC.vpaddb(reg, reg, other); break;
                case 2: _CC.vpaddw(reg, reg, other); break;
                case 4: _CC.vpaddd(reg, reg, other); break;
                case 8: _CC.vpaddq(reg, reg, other); break;
            }
        }
        return *this;
    }

    Vec& operator-=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            switch(sizeof(T)) {
                case 1: _CC.psubb(reg, other.reg); break;
                case 2: _CC.psubw(reg, other.reg); break;
                case 4: _CC.psubd(reg, other.reg); break;
                case 8: _CC.psubq(reg, other.reg); break;
            }
        } else {
            // 256 bit AVX
            switch(sizeof(T)) {
                case 1: _CC.vpsubb(reg, reg, other.reg); break;
                case 2: _CC.vpsubw(reg, reg, other.reg); break;
                case 4: _CC.vpsubd(reg, reg, other.reg); break;
                case 8: _CC.vpsubq(reg, reg, other.reg); break;
            }
        }
        return *this;
    }
    Vec& operator-=(Ref<Value<T>>&& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            other.mem.setSize(16); // change to xmmword
            switch(sizeof(T)) {
                case 1: _CC.psubb(reg, other); break;
                case 2: _CC.psubw(reg, other); break;
                case 4: _CC.psubd(reg, other); break;
                case 8: _CC.psubq(reg, other); break;
            }
        } else {
            // 256 bit AVX
            other.mem.setSize(32); // change to ymmword
            switch(sizeof(T)) {
                case 1: _CC.vpsubb(reg, reg, other); break;
                case 2: _CC.vpsubw(reg, reg, other); break;
                case 4: _CC.vpsubd(reg, reg, other); break;
                case 8: _CC.vpsubq(reg, reg, other); break;
            }
        }
        return *this;
    }

    Vec& operator/=(int amount) {
        if(is_power_of_two(amount)) {
            operator>>=(clog2(amount));
        } else {
            //TODO
            assert(false);
        }
        return *this;
    }

    Vec& operator<<=(int amount) {
        static_assert(sizeof(T) > 1, "shift does not support byte element size");
        // shift left same for signed and unsigned types
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            switch(sizeof(T)) {
                case 2: _CC.psllw(reg, amount); break;
                case 4: _CC.pslld(reg, amount); break;
                case 8: _CC.psllq(reg, amount); break;
            }
        } else {
            // 256 bit AVX
            switch(sizeof(T)) {
                case 2: _CC.vpsllw(reg, reg, amount); break;
                case 4: _CC.vpslld(reg, reg, amount); break;
                case 8: _CC.vpsllq(reg, reg, amount); break;
            }
        }
        return *this;
    }
    Vec& operator<<=(const Vec& other) {
        static_assert(sizeof(T) > 1, "shift does not support byte element size");
        // shift left same for signed and unsigned types
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            switch(sizeof(T)) {
                case 2: _CC.psllw(reg, other); break;
                case 4: _CC.pslld(reg, other); break;
                case 8: _CC.psllq(reg, other); break;
            }
        } else {
            // 256 bit AVX
            switch(sizeof(T)) {
                case 2: _CC.vpsllw(reg, reg, other); break;
                case 4: _CC.vpslld(reg, reg, other); break;
                case 8: _CC.vpsllq(reg, reg, other); break;
            }
        }
        return *this;
    }

    Vec& operator>>=(int amount) {
        static_assert(sizeof(T) > 1, "shift does not support byte element size");
        static_assert(!(std::is_signed_v<T> && sizeof(T) == 8), "no arithmetic shift right for 64 bit values");
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            if constexpr(std::is_signed_v<T>) {
                switch(sizeof(T)) {
                    case 2: _CC.psraw(reg, amount); break;
                    case 4: _CC.psrad(reg, amount); break;
                }
            } else {
                switch(sizeof(T)) {
                    case 2: _CC.psrlw(reg, amount); break;
                    case 4: _CC.psrld(reg, amount); break;
                    case 8: _CC.psrlq(reg, amount); break;
                }
            }
        } else {
            // 256 bit AVX
            if constexpr(std::is_signed_v<T>) {
                switch(sizeof(T)) {
                    case 2: _CC.vpsraw(reg, reg, amount); break;
                    case 4: _CC.vpsrad(reg, reg, amount); break;
                }
            } else {
                switch(sizeof(T)) {
                    case 2: _CC.vpsrlw(reg, reg, amount); break;
                    case 4: _CC.vpsrld(reg, reg, amount); break;
                    case 8: _CC.vpsrlq(reg, reg, amount); break;
                }
            }
        }
        return *this;
    }
    Vec& operator>>=(const Vec& other) {
        static_assert(sizeof(T) > 1, "shift does not support byte element size");
        static_assert(!(std::is_signed_v<T> && sizeof(T) == 8), "no arithmetic shift right for 64 bit values");
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            if constexpr(std::is_signed_v<T>) {
                switch(sizeof(T)) {
                    case 2: _CC.psraw(reg, other); break;
                    case 4: _CC.psrad(reg, other); break;
                }
            } else {
                switch(sizeof(T)) {
                    case 2: _CC.psrlw(reg, other); break;
                    case 4: _CC.psrld(reg, other); break;
                    case 8: _CC.psrlq(reg, other); break;
                }
            }
        } else {
            // 256 bit AVX
            if constexpr(std::is_signed_v<T>) {
                switch(sizeof(T)) {
                    case 2: _CC.vpsraw(reg, reg, other); break;
                    case 4: _CC.vpsrad(reg, reg, other); break;
                }
            } else {
                switch(sizeof(T)) {
                    case 2: _CC.vpsrlw(reg, reg, other); break;
                    case 4: _CC.vpsrld(reg, reg, other); break;
                    case 8: _CC.vpsrlq(reg, reg, other); break;
                }
            }
        }
        return *this;
    }

    Vec& operator&=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            _CC.pand(reg, other);
        } else {
            // 256 bit AVX
            _CC.vpand(reg, reg, other);
        }
        return *this;
    }
    Vec& operator|=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            _CC.por(reg, other);
        } else {
            // 256 bit AVX
            _CC.vpor(reg, reg, other);
        }
        return *this;
    }
    Vec& operator^=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            _CC.pxor(reg, other);
        } else {
            // 256 bit AVX
            _CC.vpxor(reg, reg, other);
        }
        return *this;
    }
};

template<unsigned width>
struct Vec<float, width> final {
    using T = float;

    static_assert(sizeof(T) * width == 128 / 8 || sizeof(T) * width == 256 / 8 ||
        sizeof(T) * width == 512 / 8,
        "only 128-bit, 256-bit or 512-bit vector instructions supported at the moment");

    using reg_type = std::conditional_t<sizeof(T) * width == 128 / 8,
                        asmjit::x86::Xmm, std::conditional_t<sizeof(T) * width == 256 / 8,
                        asmjit::x86::Ymm,
                        asmjit::x86::Zmm>
                    >;
    reg_type reg;

    Vec(bool zero = false, const char* name="") {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            reg = _CC.newXmm(name);
            if (zero)
                _CC.pxor(reg, reg);
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            // 256 bit AVX
            reg = _CC.newYmm(name);
            if (zero)
                _CC.vpxor(reg, reg, reg);
        } else {
            reg = _CC.newZmm(name);
            if (zero)
                _CC.vpxor(reg, reg, reg);
        }
    }
    Vec(const Vec& other) : Vec() {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>)
            _CC.movaps(reg, other.reg);
        else
            _CC.vmovaps(reg, other.reg);
    }
    Vec(Vec&& other) : reg(other.reg) {}
    Vec(reg_type reg) : reg(reg) {}

    Vec& operator=(float v) {
        if (v == 0) {
            if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
                _CC.pxor(reg, reg);
            } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
                _CC.vpxor(reg, reg, reg);
            } else {
                _CC.vpxor(reg, reg, reg);
            }
        } else {
            auto src = _CC.newFloatConst(asmjit::ConstPoolScope::kLocal, v);
            if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
                _CC.movss(reg, src);
                _CC.shufps(reg, reg, 0);
            } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
                _CC.vbroadcastss(reg, src);
            } else {
                _CC.vbroadcastss(reg, src);
            }
        }
        return *this;
    }
    // load Vec from memory, always unaligned load
    Vec& operator=(Ref<Value<T>>&& src) { load(std::move(src)); return *this; }
    Vec& operator=(const Vec& other) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>)
            _CC.movaps(reg, other.reg);
        else
            _CC.vmovaps(reg, other.reg);
        return *this;
    }

    inline unsigned getWidth() const { return width; }

    // TODO: support mask
    void load(Ref<Value<T>>&& src, bool broadcast = false) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            if (broadcast) {
                _CC.movss(reg, src);
                _CC.shufps(reg, reg, 0);
            } else {
                src.mem.setSize(16); // change to xmmword
                _CC.movups(reg, src);
            }
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(32); // change to ymmword
                _CC.vmovups(reg, src);
            }
        } else {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(64); // change to zmmword
                _CC.vmovups(reg, src);
            }
        }
    }
    Vec& operator=(Ref<Value<T>>& src) { load(src); return *this; }
    void load(Ref<Value<T>>& src, bool broadcast = false) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            if (broadcast) {
                _CC.movss(reg, src);
                _CC.shufps(reg, reg, 0);
            } else {
                src.mem.setSize(16); // change to xmmword
                _CC.movups(reg, src);
            }
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(32); // change to ymmword
                _CC.vmovups(reg, src);
            }
        } else {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(64); // change to zmmword
                _CC.vmovups(reg, src);
            }
        }
    }

    // unaligned store
    void store(Ref<Value<T>>&& dest) const {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            dest.mem.setSize(16); // change to xmmword
            _CC.movups(dest, reg);
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            // 256 bit AVX
            dest.mem.setSize(32); // change to ymmword
            _CC.vmovups(dest, reg);
        } else {
            dest.mem.setSize(64); // change to zmmword
            _CC.vmovups(dest, reg);            
        }
    }
    void store(Ref<Value<int8_t>>&& dest) const {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            dest.mem.setSize(16); // change to xmmword
            _CC.movups(dest, reg);
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            // 256 bit AVX
            dest.mem.setSize(32); // change to ymmword
            _CC.vmovups(dest, reg);
        } else {
            dest.mem.setSize(64); // change to zmmword
            _CC.vmovups(dest, reg);            
        }
    }    
    void load_aligned(Ref<Value<T>>&& src, bool broadcast = false) {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            if (broadcast) {
                _CC.movss(reg, src);
                _CC.shufps(reg, reg, 0);
            } else {
                src.mem.setSize(16); // change to xmmword
                _CC.movdqa(reg, src);
            }
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(32); // change to ymmword
                _CC.vmovdqa(reg, src);
            }
        } else {
            if (broadcast) {
                _CC.vbroadcastss(reg, src);
            } else {
                src.mem.setSize(64); // change to zmmword
                _CC.vmovdqa(reg, src);
            }
        }
    }
    void store_aligned(Ref<Value<T>>&& dest) const {
        if constexpr(std::is_same_v<reg_type,asmjit::x86::Xmm>) {
            // 128 bit SSE
            dest.mem.setSize(16); // change to xmmword
            _CC.movdqa(dest, reg);
        } else if constexpr(std::is_same_v<reg_type,asmjit::x86::Ymm>) {
            // 256 bit AVX
            dest.mem.setSize(32); // change to ymmword
            _CC.vmovdqa(dest, reg);
        } else {
            dest.mem.setSize(64); // change to zmmword
            _CC.vmovdqa(dest, reg);            
        }
    }    
    Vec& add(const Vec& other) {
        _CC.vaddps(reg, reg, other.reg);
        return *this;
    }
    Vec& add(Ref<Value<T>>&& other) {
        _CC.vaddps(reg, reg, other);
        return *this;
    }
    Vec& sub(const Vec& other) {
        _CC.vsubps(reg, reg, other.reg);
        return *this;
    }
    Vec& sub(Ref<Value<T>>&& other) {
        _CC.vsubps(reg, reg, other);
        return *this;
    }
    Vec& mul(const Vec& other) {
        _CC.vmulps(reg, reg, other.reg);
        return *this;
    }
    Vec& mul(Ref<Value<T>>&& other) {
        _CC.vmulps(reg, reg, other);
        return *this;
    }
    Vec& div(const Vec& other) {
        _CC.vdivps(reg, reg, other.reg);
        return *this;
    }
    Vec& div(Ref<Value<T>>&& other) {
        _CC.vdivps(reg, reg, other);
        return *this;
    }
    Vec& fma231(const Vec& x, const Vec& y) {
        _CC.vfmadd231ps(reg, x.reg, y.reg);
        return *this;
    }
    Vec& fma231(const Vec& x, const Ref<Value<T>>&& y) {
        _CC.vfmadd231ps(reg, x.reg, y);
        return *this;
    }    
    Vec& operator+=(const Vec& other) {
        return add(other);
    }
    Vec& operator+=(Ref<Value<T>>&& other) {
        return add(other);
    }
    Vec& operator-=(const Vec& other) {
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
    Vec& max(const Vec& other) {
        _CC.vmaxps(reg, reg, other.reg);
        return *this;
    }
    Vec& min(const Vec& other) {
        _CC.vminps(reg, reg, other.reg);
        return *this;
    }
};

template<int width, typename T>
Vec<T, width> make_vector(Ref<Value<T>>&& src) {
    Vec<T, width> v;
    v = std::move(src);
    return v;
}

} // namespace
