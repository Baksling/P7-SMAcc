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
    int *index_list_;
    T *data_;
    int max_index_;
    int max_elements_;
    uneven_list(int* index_list, T* data, int max_index, int max_elements)
    {
        this->index_list_ = index_list;
        this->data_ = data;
        this->max_index_ = max_index;
        this->max_elements_ = max_elements;
    }
public:
    uneven_list(list<list<T>>* value_list, int size_of_indexes)
    {
        this->index_list_ = static_cast<int*>(malloc(sizeof(int) * size_of_indexes));
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
        T* arr = static_cast<T*>(malloc(sizeof(T) * items));
        i = 0;
        for (T item : result) {
            arr[i] = item;
            i++;
        }

        this->data_ = arr;
        this->max_elements_ = items;
    }
    CPU GPU array_info<T> get_index(int index)
    {
        if (index >= this->max_index_ || index < 0)
        {
            printf("YOU FUCKED UP MAN! this index: %d does not exist!", index);
            array_info<T> result;
            result.size = -1;
            return result;
        }
        
        int index_val = this->index_list_[index];
        int nr_of_elements = index != this->max_index_ - 1
        ? this->index_list_[index + 1] - index_val
        : (this->max_elements_) - index_val;
        
        T* arr = static_cast<T*>(malloc(sizeof(T) * nr_of_elements));
    
        for (int i = 0; i < nr_of_elements; i++) {
            arr[i] = this->data_[(index_val + i)];
        }

        array_info<T> result {arr , nr_of_elements};

        return result;
    }
    void cuda_allocate(uneven_list<T>** location, list<void*>* free_list)
    {
        // Allocate Memory
        cudaMalloc((void**)location, sizeof(uneven_list<T>));
        free_list->push_back((*location));
        
        //move data to cuda, capture cuda pointer
        int* index_lst = nullptr;
        T* data_lst = nullptr;
        cudaMalloc((void**)&index_lst, sizeof(int) * max_index_);
        cudaMalloc((void**)&data_lst, sizeof(T) * max_elements_);
        
        //add cuda pointers to free list
        free_list->push_back(index_lst);
        free_list->push_back(data_lst);

        //create dummy uneven list with cuda pointers
        uneven_list<T>* lst = new uneven_list<T>(index_lst, data_lst,
            this->max_index_, this->max_elements_);
        
        //Copy Memory to cuda
        cudaMemcpy(index_lst, this->index_list_, sizeof(int) * max_index_, cudaMemcpyHostToDevice);
        cudaMemcpy(data_lst, this->data_, sizeof(T) * max_elements_, cudaMemcpyHostToDevice);
        cudaMemcpy((*location), lst, sizeof(uneven_list<T>), cudaMemcpyHostToDevice);
        free(lst);
    }

    CPU GPU int get_index_size() const
    {
        return this->max_index_;
    }

    ~uneven_list()
    {
        free(this->index_list_);
        free(this->data_);
    }
};




#endif //SRC2_SLN_UNEVEN_LIST_H
