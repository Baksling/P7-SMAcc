#pragma once
#include <stdlib.h>
#include "common.h"
#include <list>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T, typename V>
struct key_value_pair
{
    T key;
    V value;
};
template<typename T, typename V>
class cuda_map
{
private:
    key_value_pair<T, V>* store_;
    int size_;
    CPU GPU int hash(T item)
    {
        long p_val = reinterpret_cast<long>(item);
        return p_val % this->size_;
    }
    explicit cuda_map(key_value_pair<T, V>* store, int size)
    {
        this->size_ = size;
        this->store_ = store;
    }
public:
    explicit cuda_map(int size)
    {
        this->size_ = size;
        this->store_ = static_cast<key_value_pair<T, V>*>(malloc(sizeof(key_value_pair<T, V>) * size));
        for (int i = 0; i < size; ++i)
        {
            key_value_pair<T,V> pair = {nullptr, nullptr};
            this->store_[i] = pair;
        }
    }
    CPU GPU bool set(T key, V value)
    {
        const int hash_i = hash(key);
        for (int i = 0; i < this->size_; ++i)
        {
            int index = (hash_i + i) % this->size_;
            key_value_pair<T,V> value_pair = this->store_[index];
            if (value_pair.key == nullptr)
            {
                key_value_pair<T,V> pair = {key, value};
                this->store_[index] = pair;
                return true;
            }
            if (value_pair.key == key)
            {
                return false;
            }
        }
        return false;
    }
    CPU GPU V* get(T key)
    {
        const int hash_i = hash(key);
        for (int i = 0; i < this->size_; ++i)
        {
            int index = (hash_i + i) % this->size_;
            key_value_pair<T,V> value_pair = this->store_[index];
            if (value_pair.key == key)
            {
                return value_pair.value;
            }
        }
        return nullptr;
    }

    CPU GPU bool remove(T* key)
    {
        const int hash_i = hash(key);
        for (int i = 0; i < this->size_; ++i)
        {
            int index = (hash_i + i) % this->size_;
            key_value_pair<T,V> value_pair = this->store_[index];
            if (value_pair.key == key)
            {
                key_value_pair<T,V> pair = {nullptr, nullptr};
                this->store_[index] = pair;
                return true;
            }
        }
        return false;
    }

    void cuda_allocate(cuda_map<T, V>** p, std::list<void*>* free_list)
    {
        //move internal store
        key_value_pair<T,V>* store = nullptr; 
        cudaMalloc(&store, sizeof(key_value_pair<T,V>)*this->size_);
        free_list->push_back(store);
        cudaMemcpy(store, this->store_, sizeof(key_value_pair<T,V>)*this->size_, cudaMemcpyHostToDevice);

        
        cudaMalloc(p, sizeof(cuda_map<T, V>));
        free_list->push_back((*p));

        const cuda_map<T, V>* cpy = new cuda_map<T, V>(store, this->size_);
        cudaMemcpy((*p), cpy, sizeof(cuda_map<T, V>), cudaMemcpyHostToDevice);
    }

    ~cuda_map()
    {
        free(this->store_);
    }
};
    