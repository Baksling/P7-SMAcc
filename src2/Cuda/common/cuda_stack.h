#pragma once

#ifndef CUDA_STACK_H
#define CUDA_STACK_H
#include <cstdio>
#include <cstdlib>
#include "macro.h"
#include "allocation_helper.h"

template<typename T>
class cuda_stack
{
    T* store_ = nullptr;
    unsigned int size_ = 0;
    unsigned int stack_pointer_ = 0;
    cuda_stack(T* store, unsigned int size);
public:
    CPU GPU explicit cuda_stack(unsigned int size);

    //SIMULATION methods
    CPU GPU T peak() const;
    CPU GPU T* peak_at() const;
    CPU GPU T pop();
    CPU GPU bool is_empty() const;
    CPU GPU void push(T value);
    CPU GPU unsigned int count() const;
    CPU GPU void clear();
    CPU GPU unsigned int max_size() const;
    CPU GPU void free_internal() const;

    //Host methods
    cuda_stack<T>* cuda_allocate(const allocation_helper* helper);
};

template <typename T>
cuda_stack<T>::cuda_stack(T* store, unsigned size)
{
    this->store_ = store;
    this->size_ = size;
    this->stack_pointer_ = 0;
}

template <typename T>
cuda_stack<T>::cuda_stack(unsigned size)
{
    this->size_ = size;
    this->stack_pointer_ = 0;

    if(size > 0) this->store_ = static_cast<T*>(malloc(sizeof(T)*size));
    else this->store_ = nullptr;
}

template <typename T>
CPU GPU T cuda_stack<T>::peak() const
{
    if(this->stack_pointer_ == 0)
    {
        printf("cant get top element of empty stack");
    }
    return store_[stack_pointer_ - 1];
}

template <typename T>
CPU GPU T* cuda_stack<T>::peak_at() const
{
    if(this->stack_pointer_ == 0)
    {
        printf("cant get top element of empty stack");
    }
    return &store_[stack_pointer_ - 1];
}

template <typename T>
CPU GPU T cuda_stack<T>::pop()
{
    if(stack_pointer_ == 0)
    {
        printf("cant pop empty stack");
    }
    return store_[--stack_pointer_];
}

template <typename T>
CPU GPU bool cuda_stack<T>::is_empty() const
{
    return this->stack_pointer_ == 0;
}

template <typename T>
CPU GPU void cuda_stack<T>::push(T value)
{
    if(this->stack_pointer_ >= this->size_)
    {
        printf("Stack is full");
        return;
    }
    store_[stack_pointer_++] = value;
}

template <typename T>
CPU GPU unsigned cuda_stack<T>::count() const
{
    return this->size_ > this->stack_pointer_ ? this->stack_pointer_ : this->size_;
}

template <typename T>
CPU GPU void cuda_stack<T>::clear()
{
    this->stack_pointer_ = 0;
}

template <typename T>
unsigned cuda_stack<T>::max_size() const
{
    return this->size_;
}

template <typename T>
void cuda_stack<T>::free_internal() const
{
    free(this->store_);
}

template <typename T>
cuda_stack<T>* cuda_stack<T>::cuda_allocate(const allocation_helper* helper)
{
    cuda_stack<T>* cuda = nullptr;
    cudaMalloc(&cuda, sizeof(cuda_stack<T>));
    helper->free_list->push_back(cuda);

    T* cuda_store = nullptr;
    cudaMalloc(&cuda_store, sizeof(T)*this->size_);
    helper->free_list->push_back(cuda_store);

    //heap allocated, as it is a template, and results in segment fault if not.
    cuda_stack<T>* copy = new cuda_stack<T>(cuda_store, this->size_);
    
    cudaMemcpy(cuda, copy, sizeof(cuda_stack<T>), cudaMemcpyHostToDevice);    
    free(copy); //free heap allocated copy
    return cuda;
}

#endif


