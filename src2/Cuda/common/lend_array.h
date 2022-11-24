#ifndef LEND_ARRAY_H
#define LEND_ARRAY_H

#include "macro.h"
#include "array_t.h"

template<typename T>
struct lend_array
{
private:
    T* store_;
    int size_;
public:
    GPU CPU explicit lend_array(const array_t<T>* arr)
    {
        this->size_ = arr->size();
        this->store_ = arr->arr();
    }
    GPU CPU explicit lend_array(T* store, int size)
    {
        this->size_ = size;
        this->store_ = store;
    }
    GPU CPU T* at(int i) const
    {
        // if(i < 0 || i >= this->size_) return nullptr;
        return &this->store_[i];
    }

    GPU CPU T get(int i) const
    {
        if(i < 0 || i >= this->size_) return NULL;
        return this->store_[i];
    }
    
    GPU CPU int size() const
    {
        return this->size_;
    }
};

#endif