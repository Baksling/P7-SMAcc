
#include <string>

#include "Allocation_visitor.h"
#include "macro.cu"
#include "automata_engine.cu"
#include "../UPPAALXMLParser/uppaal_xml_parser.h"

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

    uppaal_xml_parser xml_parser;
    automata model = xml_parser.parse("./UPPAALXMLParser/XmlFiles/tests/dicebase.xml");

    Allocation_visitor av = Allocation_visitor();
    av.visit(&mode);
    
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
