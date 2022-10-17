#pragma once

#ifndef COMMON_H
#define COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <list>
#include <iostream>
#include <unordered_map>
#include <tuple>

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
        if(i < 0 || i >= this->size_)
            return nullptr;
        return &this->store_[i];
    }

    GPU CPU T get(int i) const
    {
        if(i < 0 || i >= this->size_)
            return nullptr;
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
    GPU CPU void free_array()
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
    GPU CPU explicit lend_array(array_t<T>* arr)
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
        if(i < 0 || i >= this->size_)
            return nullptr;
        return &this->store_[i];
    }

    GPU CPU T get(int i) const
    {
        if(i < 0 || i >= this->size_)
            return nullptr;
        return this->store_[i];
    }
    
    GPU CPU int size() const
    {
        return this->size_;
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
array_t<T> cuda_to_array(std::list<T>* list, std::list<void*>* free_list)
{
    int size = static_cast<int>(list->size());
    if(size == 0) return array_t<T>(0);
    T* cuda_arr = nullptr;
    T* local_arr = static_cast<T*>(malloc(sizeof(T)*size));
    cudaMalloc(&cuda_arr, sizeof(T) * size);
    free_list->push_back(cuda_arr);
    
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

class visitor;
class edge_t;
class node_t;
class constraint_t;
class clock_timer_t;
class update_t;
class stochastic_model_t;
class system_variable;
class update_expression;

class update_expression;
template<typename  T> class cuda_stack;

struct allocation_helper
{
    std::list<void*>* free_list;
    std::unordered_map<node_t*, node_t*>* node_map;
};



#include "UpdateExpressions/cuda_stack.h"

struct simulator_state
{
    cuda_stack<double> value_stack;
    cuda_stack<update_expression*> expression_stack;
    lend_array<system_variable> variables;
    lend_array<clock_timer_t> timers;
};

#include "UpdateExpressions/update_expression.h"
#include "visitor.h"
#include "constraint_t.h"
#include "edge_t.h"
#include "stochastic_model_t.h"
#include "clock_timer_t.h"
#include "node_t.h"
#include "update_t.h"
#include "system_variable.h"




#endif