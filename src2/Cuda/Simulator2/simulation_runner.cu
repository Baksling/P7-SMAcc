#include "simulation_runner.h"
#include "results/output_writer.h"
#include "allocations/cuda_allocator.h"
#include "allocations/memory_allocator.h"
#include "automata_engine.cu"
#include "common/macro.h"
#include "common/thread_pool.h"

using namespace std::chrono;

void simulation_runner::simulate_gpu(const automata* model, sim_config* config)
{
    memory_allocator allocator = memory_allocator(true);
    
    const size_t n_parallelism = static_cast<size_t>(config->blocks)*config->threads;
    const size_t total_simulations = config->total_simulations();

    cuda_allocator av = cuda_allocator(&allocator);
    const automata* model_d = av.allocate_automata(model);
    
    CUDA_CHECK(allocator.allocate_cuda(&config->cache, n_parallelism*thread_heap_size(config)));
    CUDA_CHECK(allocator.allocate_cuda(&config->random_state_arr, n_parallelism*sizeof(curandState)));

    sim_config* config_d = nullptr;
    CUDA_CHECK(allocator.allocate(&config_d, sizeof(sim_config)));
    CUDA_CHECK(cudaMemcpy(config_d, config, sizeof(sim_config), cudaMemcpyHostToDevice));
    
    const result_store store = result_store(
        static_cast<unsigned>(config->total_simulations()),
        config->tracked_variable_count,
        config->network_size,
        &allocator);
    
    result_store* store_d = nullptr;
    CUDA_CHECK(allocator.allocate(&store_d, sizeof(result_store)));
    CUDA_CHECK(cudaMemcpy(store_d, &store, sizeof(result_store), cudaMemcpyHostToDevice));
    
    output_writer writer = output_writer(
        &config->out_path,
        static_cast<unsigned>(total_simulations),
        config->write_mode,
        model        
        );

    if(config->verbose)  std::cout << "GPU simulation started\n";
    const steady_clock::time_point global_start = steady_clock::now();
    for (unsigned r = 0; r < config->simulation_repetitions; ++r)
    {
        const steady_clock::time_point local_start = steady_clock::now();
        simulator_gpu_kernel<<<config->blocks, config->threads>>>(model_d, store_d, config_d);
        cudaDeviceSynchronize();
        if(cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error("An error was encountered while running simulation. Error: " + std::to_string(cudaPeekAtLastError()) + ".\n" );

        writer.write(
            &store,
            std::chrono::duration_cast<milliseconds>(steady_clock::now() - local_start));
    }
    if(config->verbose) std::cout << "GPU simulation finished\n";
    writer.write_summary(std::chrono::duration_cast<milliseconds>(steady_clock::now() - global_start));

    allocator.free_allocations();
}


void simulation_runner::simulate_cpu(const automata* model, sim_config* config)
{
    memory_allocator allocator = memory_allocator(false);
    
    const size_t n_parallelism = static_cast<size_t>(config->blocks)*config->threads;
    const size_t total_simulations = config->total_simulations();

    const result_store store = result_store(
    static_cast<unsigned>(total_simulations),
    config->tracked_variable_count,
    config->network_size,
    &allocator);

    output_writer writer = output_writer(
        &config->out_path,
        static_cast<unsigned>(total_simulations),
        config->write_mode,
        model        
        );
    
    CUDA_CHECK(allocator.allocate_host(&config->cache, n_parallelism*thread_heap_size(config)));
    CUDA_CHECK(allocator.allocate_host(&config->random_state_arr, n_parallelism*sizeof(curandState)));

    std::cout << "CPU simulation started\n";
    const steady_clock::time_point global_start = steady_clock::now();
    for (unsigned r = 0; r < config->simulation_repetitions; ++r)
    {
        const steady_clock::time_point local_start = steady_clock::now();
        thread_pool pool = {config->cpu_threads};
    
        for (int i = 0; i < n_parallelism; ++i)
        {
            pool.queue_job([model, &store, config, i]()
            {
                simulate_automata(i, model, &store, config);
            });
        }
        pool.await_run();

        writer.write(&store, std::chrono::duration_cast<milliseconds>(steady_clock::now() - local_start));
    }
    writer.write_summary(std::chrono::duration_cast<milliseconds>(steady_clock::now() - global_start));
    
    allocator.free_allocations();
}
