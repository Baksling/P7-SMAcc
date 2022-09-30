#ifndef COMMON_INCLUDE_FILE
#define COMMON_INCLUDE_FILE

#include <cuda.h>
#include <cuda_runtime.h>

#include "constraint_d.h"
#include "constraint_d.h"
#include "cuda_map.h"
#include "uneven_list.h"

#define GPU __device__
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__

template<typename T>
array_info<T> to_array(list<T>* list)
{
    int size = list->size();
    T* arr = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0;
    for(T item : *list)
    {
        arr[i] = item;
        i++;
    }

    array_info<T> info = { arr, size };
    return info;
}

template<typename T>
array_info<T> to_flat_array(list<list<T>*>* list2d, cuda_map<int, int> )
{
    int size = 0;
    for(list<T>* lst : list2d)
    {
        size += lst->size();
    }
    
    T* arr = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0, j = 0;
    for(list<T> lst : *list2d)
    {
        
        for(T item : *lst)
        {
            arr[i] = item;
            i++;
        }
    }
    
    
}

#endif