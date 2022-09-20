//
// Created by Patrick on 19-09-2022.
//

#ifndef SRC2_SLN_UNEVEN_LIST_H
#define SRC2_SLN_UNEVEN_LIST_H

#include <list>

using namespace std;

template <typename T>
struct array_info
{
    T* arr;
    int size;
};

template <class T>
class uneven_list {
private:
    int* index_list_;
    T* data_;
    int max_index_;
    int max_elements_;
public:
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
    array_info<T> get_index(int index)
    {
        int index_val = this->index_list_[index];
        int nr_of_elements = index != this->max_index_ - 1 ? this->index_list_[index + 1] - index_val : this->max_elements_ - index_val;
    
        array_info<T> result;
        T* arr = (T*)malloc(sizeof(T) * nr_of_elements);
    
        for (int i = 0; i < nr_of_elements; i++) {
            arr[i] = this->data_[i+index_val];
        }
    
        result.arr = arr;
        result.size = nr_of_elements;
    
        return result;
    }
};


#endif //SRC2_SLN_UNEVEN_LIST_H
