#pragma once
#ifndef COMMON_INCLUDE_FILE
#define COMMON_INCLUDE_FILE

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <list>


#define GPU __device__
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__


template<typename T>
struct array_t
{
private:
    T* store_;
    int size_;
public:
    explicit array_t(int size)
    {
        this->size_ = size;
        this->store_ = size > 0
        ? static_cast<T*>(malloc(sizeof(T)*size))
        : nullptr;
    }
    explicit array_t(T* store, int size)
    {
        this->size_ = size;
        this->store_ = store;
    }

    T* at(int i) const
    {
        if(i < 0 || i >= this->size_)
            return nullptr;
        return &this->store_[i];
    }

    T* arr() const
    {
        return this->store_;
    }

    int size() const
    {
        return this->size_;
    }

    // ReSharper disable once CppMemberFunctionMayBeConst
    void free_array()
    {
        free(this->store_);
    }
};

template<typename T>
struct lend_array
{
private:
    T* store_;
    int size_;
public:
    explicit lend_array(array_t<T>* arr)
    {
        this->size_ = arr->size();
        this->store_ = arr->arr();
    }
    explicit lend_array(T* store, int size)
    {
        this->size_ = size;
        this->store_ = store;
    }
    T* at(int i) const
    {
        if(i < 0 || i >= this->size_)
            return nullptr;
        return &this->store_[i];
    }

    int size() const
    {
        return this->size_;
    }
};


template<typename T>
array_t<T> to_array(std::list<T>* list)
{
    int size = static_cast<int>(list->size());
    T* arr = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0;
    for(T item : *list)
    {
        arr[i] = item;
        i++;
    }

    array_t<T> info = array_t<T>(arr, size);
    return info;
}

template<typename T>
array_t<T> to_array_as_pointers(std::list<T*>* list)
{
    int size = static_cast<int>(list->size());
    T* arr = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0;
    for(T* item : *list)
    {
        arr[i] = *item;
        i++;
    }

    array_t<T> info = array_t<T>(arr, size);
    return info;
}

#endif