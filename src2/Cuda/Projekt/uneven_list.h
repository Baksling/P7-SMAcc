//
// Created by Patrick on 19-09-2022.
//

#ifndef SRC2_SLN_UNEVEN_LIST_H
#define SRC2_SLN_UNEVEN_LIST_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <list>
#include <stdio.h>

#define GPU __device__
#define CPU __host__

using namespace std;

template <typename T>
struct array_info
{
    T* arr;
    int size;

    CPU GPU void free_arr() const
    {
        free(this->arr);
    }
};

template <class T>
class uneven_list {
private:
    int *index_list_, *index_list_d_;
public:
    T *data_;
    T *data_d_ = nullptr;
    int max_index_;
    int max_elements_;
    uneven_list(list<list<T>>* value_list, int size_of_indexes)
    {
        this->index_list_ = (int*)malloc(sizeof(int) * size_of_indexes);
        this->max_index_ = size_of_indexes;
        list<T> result;
        int items = 0;
        int i = 0;
        for (list<T> item_list : *value_list) {
            this->index_list_[i++] = items;
    
            for(T item : item_list) {
                items++;
                result.emplace_back(item);
            }
        }

        // Convert back to array!
        T* arr = (T*)malloc(sizeof(T) * items);
        i = 0;
        for (T item : result) {
            arr[i] = item;
            i++;
        }

        this->data_ = arr;
        this->max_elements_ = items;
    }
    __host__ __device__ array_info<T> get_index(int index)
    {
        if (index >= this->max_index_ || index < 0)
        {
            printf("YOU FUCKED UP MAN! this index: %d does not exist!", index);
            array_info<T> result;
            result.size = -1;
            return result;
        }
        
        int index_val = this->index_list_d_[index];
        int nr_of_elements = index != this->max_index_ - 1 ? this->index_list_d_[index + 1] - index_val : (this->max_elements_) - index_val;

        // ME DOING BRAIN EXERCICE TO CHECK MY SANITY!
        // printf("DEBUG:: index_val: %d -- Number of elements: %d %d \n", index_val, nr_of_elements, this->max_elements_);


        // index:  [0, 2, 3]
        // values: [0, 1, 2, 3]

        // index 0 -> nr: (2 - 0) = 2
        // index 1 -> nr: (3 - 1) = 2
        // index 2 -> nr: (4 - 3) = 1

        // i 0 = 0 + 0 :: 0 + 1
        // i 2 = 2 + 0
        // i 3 = 3 + 0
        
        T* arr = static_cast<T*>(malloc(sizeof(T) * nr_of_elements));
    
        for (int i = 0; i < nr_of_elements; i++) {
            arr[i] = this->data_d_[(index_val + i)];
        }

        array_info<T> result {arr , nr_of_elements};

        return result;
    }
    void allocate_memory()
    {
        // Allocate Memory
        cudaMalloc((void**)&index_list_d_, sizeof(int) * max_index_);
        cudaMalloc((void**)&data_d_, sizeof(T) * max_elements_);

        //Copy Memory
        cudaMemcpy(index_list_d_, index_list_, sizeof(int) * max_index_, cudaMemcpyHostToDevice);
        cudaMemcpy(data_d_, data_, sizeof(T) * max_elements_, cudaMemcpyHostToDevice);
    }

    void allocate_memory_2(uneven_list<T>** location)
    {
        // Allocate Memory
        cudaMalloc(location, sizeof(uneven_list<T>));
        cudaMalloc(&index_list_d_, sizeof(int) * max_index_);
        cudaMalloc(&data_d_, sizeof(T) * max_elements_);

        //Copy Memory
        cudaMemcpy(index_list_d_, index_list_, sizeof(int) * max_index_, cudaMemcpyHostToDevice);
        cudaMemcpy(data_d_, data_, sizeof(T) * max_elements_, cudaMemcpyHostToDevice);
        cudaMemcpy(location, this, sizeof(uneven_list<T>), cudaMemcpyHostToDevice);
    }
    
    void free_memory() const
    {
        free(index_list_d_);
        free(data_d_);
    }
    void cuda_free_memory() const{
        cudaFree(index_list_d_);
        cudaFree(data_d_);
    }
};


#endif //SRC2_SLN_UNEVEN_LIST_H
