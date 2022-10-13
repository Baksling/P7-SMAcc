#pragma once

#ifndef CUDA_STACK_H
#define CUDA_STACK_H
#include <cstdio>
#include <cstdlib>

template<typename T>
class cuda_stack
{
    T* store_ = nullptr;
    unsigned int size_ = 0;
    unsigned int stack_pointer_ = 0;
public:
    explicit cuda_stack(unsigned int size);
    T peak() const;
    T pop();
    void push(T value);
    unsigned int count() const;
};

template <typename T>
cuda_stack<T>::cuda_stack(unsigned size)
{
    this->size_ = size;
    this->stack_pointer_ = static_cast<unsigned int>(-1);

    if(size > 0)
    {
        this->store_ = static_cast<T*>(malloc(sizeof(T)*size));
    }
}

template <typename T>
T cuda_stack<T>::peak() const
{
    if(this->size_ < stack_pointer_)
    {
        printf("cant get top element of empty stack");
    }
    return store_[stack_pointer_];
}

template <typename T>
T cuda_stack<T>::pop()
{
    if(this->size_ < stack_pointer_)
    {
        printf("cant pop empty stack");
    }
    return store_[stack_pointer_--];
}

template <typename T>
void cuda_stack<T>::push(T value)
{
    return store_[++stack_pointer_] = value;
}

template <typename T>
unsigned cuda_stack<T>::count() const
{
    return this->size_ > this->stack_pointer_ ? this->stack_pointer_ : this->size_;
}

#endif
