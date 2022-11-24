#ifndef ARRAY_T_H
#define ARRAY_T_H

#include "macro.h"
#include <list>
#include "allocation_helper.h"

template<typename T>
struct array_t
{
private:
    T* store_;
    int size_;
public:
    GPU CPU explicit array_t(int size)
    {
        this->size_ = size;
        this->store_ = size > 0
        ? static_cast<T*>(malloc(sizeof(T)*size))
        : nullptr;
    }
    GPU CPU explicit array_t(T* store, int size)
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

    GPU CPU T* arr() const
    {
        return this->store_;
    }

    GPU CPU int size() const
    {
        return this->size_;
    }

    // ReSharper disable once CppMemberFunctionMayBeConst
    GPU CPU void free_internal()
    {
        free(this->store_);
    }
};

template<typename T>
array_t<T> to_array(std::list<T>* list)
{
    int size = static_cast<int>(list->size());
    if(size == 0) return array_t<T>(0);
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
array_t<T> cuda_to_array(std::list<T>* list, allocation_helper* helper)
{
    int size = static_cast<int>(list->size());
    if(size == 0) return array_t<T>(0);
    T* cuda_arr = nullptr;
    T* local_arr = static_cast<T*>(malloc(sizeof(T)*size));
    helper->allocate(&cuda_arr, sizeof(T) * size);
    
    int i = 0;
    for(T item : *list)
    {
        local_arr[i] = item;
        i++;
    }

    cudaMemcpy(cuda_arr, local_arr, sizeof(T) * size, cudaMemcpyHostToDevice);
    array_t<T> info = array_t<T>(cuda_arr, size);
    free(local_arr);
    return info;
}

#endif