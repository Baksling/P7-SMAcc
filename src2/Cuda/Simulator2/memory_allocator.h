#ifndef MEMORY_ALLOCATOR_CU
#define MEMORY_ALLOCATOR_CU

#include <list>
class memory_allocator
{
private:
    std::list<void*> free_list_;
    size_t allocated_size_;
public:
    const bool use_cuda;

    explicit memory_allocator(const bool use_cuda): use_cuda(use_cuda) {
        this->free_list_ = std::list<void*>();
        allocated_size_ = 0;
    }

    void add(void* p, const size_t size)
    {
        this->free_list_.push_back(p);
        this->allocated_size_ += size;
    }

    template<typename T>
    cudaError allocate(T** p, const size_t size)
    {
        cudaError e;
        if(size == 0)
        {
            *p = nullptr;
            return cudaSuccess;
        }
        if(this->use_cuda)
        {
            e = cudaMalloc(p, size);
        }
        else
        {
            *p = static_cast<T*>(malloc(size));
            e = cudaSuccess;
        }
        this->free_list_.push_back(*p);
        this->allocated_size_ += size;
        return e;
    }

    template<typename T>
    cudaError allocate_and_copy(T** dest, T* source, const unsigned amount)
    {
        const size_t size = sizeof(T)*amount;
        const cudaMemcpyKind kind = use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        if(amount == 0)
        {
            *dest = nullptr;
            return cudaSuccess;
        }

        cudaError e = allocate(dest, size);
        if(e != cudaSuccess) return e;

        e = cudaMemcpy(*dest, source, size, kind);
        return e;
    }

    size_t get_memory_util() const
    {
        return this->allocated_size_;
    }

    void free_allocations()
    {
        this->allocated_size_ = 0;
        for (const auto p : this->free_list_)
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