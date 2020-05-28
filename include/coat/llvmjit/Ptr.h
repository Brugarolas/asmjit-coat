#ifndef COAT_LLVMJIT_PTR_H_
#define COAT_LLVMJIT_PTR_H_

#include "ValueBase.h"
//#include "Ref.h"


namespace coat {

template<class T>
struct Ptr<::llvm::IRBuilder<>,T> {
	using F = ::llvm::IRBuilder<>;
	using value_type = std::remove_pointer_t<T>;
	using value_base_type = ValueBase<F>;
//	using mem_type = Ref<F,T>;
    // TODO *pptr2 = ptr1 should result in **pptr2 = *ptr1
    //  A solution can be adding an specific Ref for pointer type.
    //  This is necessary because pointers don't support <, <=, >, >=
    using mem_type = std::conditional_t<std::is_pointer_v<value_type>,
            Ptr<F, value_type>, Ref<F,Value<F, value_type>>
    >;

//	static_assert(std::is_base_of_v<value_base_type,T>, "pointer type only of register wrappers");
    // Assert that T is a pointer
    // TODO assert that is pointer to integer
    static_assert(std::is_pointer<T>::value, "Only pointer types supported");

	llvm::IRBuilder<> &cc;
	llvm::Value *memreg;

	llvm::Value *load() const { return cc.CreateLoad(memreg, "load"); }
	void store(llvm::Value *v) { cc.CreateStore(v, memreg); }
	llvm::Type *type() const { return ((llvm::PointerType*)memreg->getType())->getElementType(); }

	Ptr(F &cc, const char *name="") : cc(cc) {
		// llvm IR has no types for unsigned/signed integers
		switch(sizeof(value_type)){
			case 1: memreg = allocateStackVariable(cc, llvm::Type::getInt8PtrTy (cc.getContext()), name); break;
			case 2: memreg = allocateStackVariable(cc, llvm::Type::getInt16PtrTy(cc.getContext()), name); break;
			case 4: memreg = allocateStackVariable(cc, llvm::Type::getInt32PtrTy(cc.getContext()), name); break;
			case 8: memreg = allocateStackVariable(cc, llvm::Type::getInt64PtrTy(cc.getContext()), name); break;
		}
	}
	Ptr(F &cc, value_type *val, const char *name="") : Ptr(cc, name) {
		*this = val;
	}
	Ptr(F &cc, const value_type *val, const char *name="") : Ptr(cc, name) {
		*this = const_cast<value_type*>(val);
	}
	// real copy requires new stack memory and copy of content
	Ptr(const Ptr &other) : Ptr(other.cc) {
		*this = other;
	}
	// move, just take the stack memory
	Ptr(const Ptr &&other) : cc(other.cc), memreg(other.memreg) {}

    // For when initalizing with references
    Ptr(F &cc, llvm::Value *mem) : Ptr(cc) {
        store(cc.CreateLoad(mem, "load"));
    }

	//FIXME: takes any type
	Ptr &operator=(llvm::Value *val){ store( val ); return *this; }

	Ptr &operator=(value_type *value){
		llvm::Constant *int_val = llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), (uint64_t)value);
		store( cc.CreateIntToPtr(int_val, type()) );
		return *this;
	}
	Ptr &operator=(const Ptr &other){ store( other.load() ); return *this; }

	// dereference
	mem_type operator*(){
		return { cc, cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), 0)) };
	}
	// indexing with variable
	mem_type operator[](const value_base_type &idx){
		return { cc, cc.CreateGEP(load(), idx.load()) };
	}
	// indexing with constant -> use offset
	mem_type operator[](size_t idx){
		return { cc, cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), idx)) };
	}
	
	Ptr operator+(const value_base_type &value) const {
		Ptr res(cc);
		res.store( cc.CreateGEP(load(), value.load()) );
		return res;
	}
	Ptr operator+(size_t value) const {
		Ptr res(cc);
		res.store( cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), value)) );
		return res;
	}

	Ptr &operator+=(const value_base_type &value){
		store( cc.CreateGEP(load(), value.load()) );
		return *this;
	}
	Ptr &operator+=(int amount){
		store( cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), amount)) );
		return *this;
	}
	Ptr &operator-=(int amount){
		store( cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), -amount)) );
		return *this;
	}

	// like "+=" without pointer arithmetic
	Ptr &addByteOffset(const value_base_type &value){ //TODO: any integer value should be possible as operand
		llvm::Value *int_reg = cc.CreatePtrToInt(load(), llvm::Type::getInt64Ty(cc.getContext()));
		llvm::Value *int_value = cc.CreatePtrToInt(value.load(), llvm::Type::getInt64Ty(cc.getContext()));
		llvm::Value *int_sum = cc.CreateAdd(int_reg, int_value);
		store( cc.CreateIntToPtr(int_sum, type()) );
		return *this;
	}

	// operators creating temporary
	Value<F,size_t> operator- (const Ptr &other) const {
		Value<F,size_t> ret(cc, "ret");
		llvm::Value *int_reg = cc.CreatePtrToInt(load(), llvm::Type::getInt64Ty(cc.getContext()));
		llvm::Value *int_other = cc.CreatePtrToInt(other.load(), llvm::Type::getInt64Ty(cc.getContext()));
		llvm::Value *bytes = cc.CreateSub(int_reg, int_other);
		// compilers do arithmetic shift...
		llvm::Value *elements = cc.CreateAShr(bytes, clog2(sizeof(value_type)), "", true);
		ret.store(elements);
		return ret;
	}

	// pre-increment, post-increment not provided as it creates temporary
	Ptr &operator++(){
		store( cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), 1)) );
		return *this;
	}
	// pre-decrement
	Ptr &operator--(){
		store( cc.CreateGEP(load(), llvm::ConstantInt::get(llvm::Type::getInt64Ty(cc.getContext()), -1)) );
		return *this;
	}

	// comparisons
	Condition<F> operator==(const Ptr &other) const { return {cc, memreg, other.memreg, ConditionFlag::e};  }
	Condition<F> operator!=(const Ptr &other) const { return {cc, memreg, other.memreg, ConditionFlag::ne}; }
};


template<typename dest_type, typename src_type>
Ptr<::llvm::IRBuilder<>,dest_type>
cast(const Ptr<::llvm::IRBuilder<>,src_type> &src){
	static_assert(std::is_pointer_v<dest_type>, "a pointer type can only be casted to another pointer type");

	// create new pointer
	Ptr<::llvm::IRBuilder<>,dest_type> res(src.cc);
	// cast between pointer types
	res.store(
		src.cc.CreateBitCast(
			src.load(),
			res.type()
		)
	);
	// return new pointer
	return res;
}

} // namespace

#endif
