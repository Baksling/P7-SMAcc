
#include <string>
#include "macro.cu"
#include "automata_engine.cu"
#include "./results/outout_writer.h"

#define CHECK(x)             \
do{                          \
if ((x) != cudaSuccess) {    \
    throw std::runtime_error(std::string("Allocation error on line ") +  std::to_string(__LINE__));\
}                             \
}while(0) 


template<typename T>
T* cuda_yeet(T* t, memory_allocator* allocator)
{
    T* tp = nullptr;
    CHECK(allocator->allocate(&tp, sizeof(T)));
    CHECK(cudaMemcpy(tp, t, sizeof(T), cudaMemcpyHostToDevice));
    return tp;
}

int main(int argc, const char* argv[])
{
    CHECK(cudaFree(nullptr));
    memory_allocator helper = memory_allocator(true);
    
    expr e1 = {expr::literal_ee,nullptr, nullptr, {nullptr}};
    e1.value = 1.0;
    
    node n2 = node{
        2,
        cuda_yeet(&e1, &helper),
        arr<edge>::empty(),
        arr<constraint>::empty(),
        false,
        true
    };

    node n3 = node{
        3,
        cuda_yeet(&e1, &helper),
        arr<edge>::empty(),
        arr<constraint>::empty(),
        false,
        true
    };

    edge e12 = edge{
        TAU_CHANNEL,
        cuda_yeet(&e1, &helper),
        cuda_yeet(&n2, &helper),
        arr<constraint>::empty(),
        arr<update>::empty(),
    };

    edge e13 = edge{
        TAU_CHANNEL,
        cuda_yeet(&e1, &helper),
        cuda_yeet(&n3, &helper),
        arr<constraint>::empty(),
        arr<update>::empty(),
    };

    edge* e_store = static_cast<edge*>(malloc(sizeof(edge)*2));
    e_store[0] = e12;
    e_store[1] = e13;
    edge* e_d = nullptr;
    CHECK(cudaMalloc(&e_d, sizeof(edge)*2));
    CHECK(cudaMemcpy(e_d, e_store, sizeof(edge)*2, cudaMemcpyHostToDevice));
    

    node n1 = node{
        1,
        cuda_yeet(&e1, &helper),
        arr<edge>{ e_d, 2 },
        arr<constraint>::empty(),
        false,
        false
    };

    node* n1_d = cuda_yeet(&n1, &helper);
    node** n1_dd = nullptr;
    CHECK(cudaMalloc(&n1_dd, sizeof(node*)));
    CHECK(cudaMemcpy(n1_dd, &n1_d, sizeof(node**), cudaMemcpyHostToDevice));

    automata a = {
        arr<node*>{ n1_dd, 1 },
        arr<clock_var>::empty() 
        };

    automata* a_d = nullptr; 
    CHECK(cudaMalloc(&a_d, sizeof(automata)));
    CHECK(cudaMemcpy(a_d, &a, sizeof(automata), cudaMemcpyHostToDevice));

    sim_config config = {
        4,
        256,
        10,
        10,
        123,
        true,
        100,
        100.0,
        0,
        0,
        1,
        nullptr,
        nullptr,
        1
    };

    result_store store = result_store(config, &helper);
    result_store* store_d = nullptr;
    CHECK(cudaMalloc(&store_d, sizeof(result_store)));
    CHECK(cudaMemcpy(store_d, &store, sizeof(result_store), cudaMemcpyHostToDevice));
    
    CHECK(cudaMalloc(&config.cache, static_cast<unsigned long long>(config.blocks*config.threads)*thread_heap_size(&config)));
    CHECK(cudaMalloc(&config.random_state_arr, static_cast<unsigned long long>(config.blocks*config.threads)*sizeof(curandState)));

    sim_config* config_d = nullptr;
    CHECK(cudaMalloc(&config_d, sizeof(sim_config)));
    CHECK(cudaMemcpy(config_d, &config, sizeof(sim_config), cudaMemcpyHostToDevice));

    printf("pre: %d\n", cudaPeekAtLastError());
    
    simulator_gpu_kernel<<<config.blocks, config.threads>>>(a_d, store_d, config_d);
    cudaDeviceSynchronize();
    
    printf("post: %d\n", cudaPeekAtLastError());
}
