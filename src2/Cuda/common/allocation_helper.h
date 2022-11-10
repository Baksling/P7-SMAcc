#ifndef ALLOCATION_HELPER_H
#define ALLOCATION_HELPER_H

#include <list>
#include <unordered_map>

class node_t;

struct allocation_helper
{
    explicit allocation_helper()
    {
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
    void allocate_cuda(T** p, const size_t size)
    {
        cudaMalloc(p, size);
        this->free_list.push_back(p);
        this->allocated_size += size;
    }
};

#endif