#ifndef ALLOCATION_HELPER_H
#define ALLOCATION_HELPER_H

#include <list>
#include <unordered_map>
#include "macro.h"

class node_t;

struct allocation_helper
{
    const bool use_cuda;
    explicit allocation_helper(const bool use_cuda): use_cuda(use_cuda) {
        this->free_list = std::list<void*>();
        this->node_map = std::unordered_map<node_t*, node_t*>();
        allocated_size = 0;
    }

    std::list<void*> free_list;
    std::unordered_map<node_t*, node_t*> node_map;
    unsigned long long allocated_size;
    
    void add(void* p, const size_t size)
    {
        this->free_list.push_back(p);
        this->allocated_size += size;
    }

    template<typename T>
    void allocate(T** p, const size_t size)
    {
        if(size == 0)
        {
            *p = nullptr;
            return;
        }
        if(this->use_cuda)
        {
            cudaMalloc(p, size);
        }
        else
        {
            *p = static_cast<T*>(malloc(size));
        }
        this->free_list.push_back(*p);
        this->allocated_size += size;
    }

    void free_allocations()
    {
        this->node_map.clear();
        this->allocated_size = 0;
        for (const auto p : this->free_list)
        {
            if(this->use_cuda)
            {
                cudaFree(p);
            }
            else
            {
                free(p);
            }
        }
    }
};

#endif