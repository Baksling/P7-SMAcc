#ifndef ALLOCATION_HELPER_H
#define ALLOCATION_HELPER_H

#include <list>
#include <unordered_map>


//Prototype :-D &*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*

template<typename T>
CPU GPU void allocate(T** ptr, unsigned long long int size, const bool cuda_allocate)
{
    if(cuda_allocate)
    {
        cudaMalloc(ptr, size);
    }
    else
    {
        *ptr = static_cast<T*>(malloc(size));
    }
}

class node_t;

struct allocation_helper
{
    std::list<void*>* free_list;
    std::unordered_map<node_t*, node_t*>* node_map;
};

#endif