#ifndef COAT_LLVMJIT_REF_H_
#define COAT_LLVMJIT_REF_H_


namespace coat {

template<class T>
struct Ref<::llvm::IRBuilder<>,T> {
	using F = ::llvm::IRBuilder<>;
	using inner_type = T;

	llvm::IRBuilder<> &cc;
	llvm::Value *mem;

	llvm::Value *load() const { return cc.CreateLoad(mem, "memload"); }
	//void store(llvm::Value *v) { cc.CreateStore(v, mem); }
	llvm::Type *type() const { return ((llvm::PointerType*)mem->getType())->getElementType(); }


	Ref(llvm::IRBuilder<> &cc, llvm::Value *mem) : cc(cc), mem(mem) {}

	Ref &operator=(const T &other){
		cc.CreateStore(other.load(), mem);
		return *this;
	}
	Ref &operator=(int value){
		cc.CreateStore(llvm::ConstantInt::get(type(), value), mem);
		return *this;
	}

	// operators creating temporary virtual registers
	T operator<<(int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp <<= amount; return tmp; }
	T operator>>(int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp >>= amount; return tmp; }
	T operator+ (int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp  += amount; return tmp; }
	T operator- (int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp  -= amount; return tmp; }
	T operator& (int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp  &= amount; return tmp; }
	T operator| (int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp  |= amount; return tmp; }
	T operator^ (int amount) const { T tmp(cc, "tmp"); tmp = *this; tmp  ^= amount; return tmp; }

	//T operator*(const T &other) const { T tmp(cc, "tmp"); tmp = *this; tmp *= other; return tmp; }
	//T operator/(const T &other) const { T tmp(cc, "tmp"); tmp = *this; tmp /= other; return tmp; }
	//T operator%(const T &other) const { T tmp(cc, "tmp"); tmp = *this; tmp %= other; return tmp; }

	// comparisons
	// swap sides of operands and comparison, not needed for assembly, but avoids code duplication in wrapper
	Condition<F> operator==(const T &other) const { return other==*this; }
	Condition<F> operator!=(const T &other) const { return other!=*this; }
	Condition<F> operator< (const T &other) const { return other>=*this; }
	Condition<F> operator<=(const T &other) const { return other> *this; }
	Condition<F> operator> (const T &other) const { return other<=*this; }
	Condition<F> operator>=(const T &other) const { return other< *this; }
	//TODO: possible without temporary: cmp m32 imm32, complicates Condition
	Condition<F> operator==(int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp==constant; }
	Condition<F> operator!=(int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp!=constant; }
	Condition<F> operator< (int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp< constant; }
	Condition<F> operator<=(int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp<=constant; }
	Condition<F> operator> (int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp> constant; }
	Condition<F> operator>=(int constant) const { T tmp(cc, "tmp"); tmp = *this; return tmp>=constant; }
	// not possible in instruction, requires temporary
	Condition<F> operator==(const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp==other; }
	Condition<F> operator!=(const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp!=other; }
	Condition<F> operator< (const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp< other; }
	Condition<F> operator<=(const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp<=other; }
	Condition<F> operator> (const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp> other; }
	Condition<F> operator>=(const Ref &other) const { T tmp(cc, "tmp"); tmp = *this; return tmp>=other; }
};

} // namespace

#endif
