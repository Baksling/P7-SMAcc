
#include <string>

#include "./allocations/memory_allocator.h"
#include "./allocations/cuda_allocator.h"
#include "macro.h"
#include "automata_engine.cu"
#include "../UPPAALXMLParser/uppaal_xml_parser.h"
#include "./results/output_writer.h"
#include "sim_config.h"

 


template<typename T>
T* cuda_yeet(T* t, memory_allocator* allocator)
{
    T* tp = nullptr;
    CUDA_CHECK(allocator->allocate(&tp, sizeof(T)));
    CUDA_CHECK(cudaMemcpy(tp, t, sizeof(T), cudaMemcpyHostToDevice));
    return tp;
}

int main(int argc, const char* argv[])
{
    CUDA_CHECK(cudaFree(nullptr));
    memory_allocator helper = memory_allocator(true);
    uppaal_xml_parser xml_parser;
    const automata model = xml_parser.parse("./UPPAALXMLParser/XmlFiles/tests/dicebase.xml");
    cuda_allocator av = cuda_allocator(&helper);
    const automata* model_d = av.allocate_automata(&model);
    
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

    const result_store store = result_store(config.blocks, config.threads, config.simulation_amount,
                                            config.variable_count,  config.network_size, &helper);

    std::string path = "./test";
    output_writer writer = output_writer(
        &path,
        config.blocks*config.threads*config.simulation_amount,
        output_writer::parse_mode("c"),
        &model        
        );
    
    result_store* store_d = nullptr;
    CUDA_CHECK(cudaMalloc(&store_d, sizeof(result_store)));
    CUDA_CHECK(cudaMemcpy(store_d, &store, sizeof(result_store), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&config.cache, static_cast<unsigned long long>(config.blocks*config.threads)*thread_heap_size(&config)));
    CUDA_CHECK(cudaMalloc(&config.random_state_arr, static_cast<unsigned long long>(config.blocks*config.threads)*sizeof(curandState)));

    sim_config* config_d = nullptr;
    CUDA_CHECK(cudaMalloc(&config_d, sizeof(sim_config)));
    CUDA_CHECK(cudaMemcpy(config_d, &config, sizeof(sim_config), cudaMemcpyHostToDevice));

    printf("pre: %d\n", cudaPeekAtLastError());
    const std::chrono::steady_clock::time_point global_start = std::chrono::steady_clock::now();

    
    simulator_gpu_kernel<<<config.blocks, config.threads>>>(model_d, store_d, config_d);
    cudaDeviceSynchronize();
    
    printf("post: %d\n", cudaPeekAtLastError());
    const std::chrono::steady_clock::duration sim_duration = std::chrono::steady_clock::now() - global_start;


    writer.write(&store, sim_duration);
    writer.write_summary(sim_duration);

    printf("PULLY PORKY");
}
