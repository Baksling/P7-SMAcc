// #include "cuda_simulator.h"
//
// void cuda_simulator::simulate(stochastic_model_t* model, simulation_strategy* strategy)
// {
//     //setup start variables
//     const unsigned long total_simulations = strategy->total_simulations();
//     //setup random state
//     curandState* state;
//     cudaMalloc(&state, sizeof(curandState) * strategy->degree_of_parallelism());
//     
//     //setup results array
//     const unsigned long size = sizeof(int)*total_simulations;
//     int* cuda_results = nullptr;
//     const auto r = cudaMalloc(&cuda_results, sizeof(int)*total_simulations);
//     printf("allocated %lu (%lu*%lu) bytes successfully: %s\n" ,
//         size, static_cast<unsigned long>(sizeof(int)), total_simulations, (r == cudaSuccess ? "True" : "False") );
//
//     //prepare allocation helper
//     std::list<void*> free_list;
//     std::unordered_map<node_t*, node_t*> node_map;
//     const allocation_helper allocator = { &free_list, &node_map };
//
//     //allocate model to cuda
//     stochastic_model_t* model_d = nullptr;
//     model->cuda_allocate(&model_d, &allocator);
//     
//     //implement here
//     model_options* options_d = nullptr;
//     const model_options options = {
//         strategy->simulation_amounts,
//         strategy->max_sim_steps,
//         static_cast<unsigned long>(time(nullptr))
//     };
//     cudaMalloc(&options_d, sizeof(model_options));
//     cudaMemcpy(options_d, &options, sizeof(model_options), cudaMemcpyHostToDevice);
//
//
//     //run simulations
//     std::map<int, unsigned long> node_results;
//     const steady_clock::time_point start = steady_clock::now();
//     std::cout << "Started running!\n";
//     if(cudaSuccess != cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8589934592))
//         printf("COULD NOT CHANGE LIMIT");
//     
//     for (int i = 0; i < strategy->sim_count; ++i)
//     {
//         //simulate on device
//         simulate_gpu<<<strategy->block_n, strategy->threads_n>>>(
//         model_d, options_d, state, cuda_results);
//
//         //wait for all processes to finish
//         cudaDeviceSynchronize();
//         if(cudaPeekAtLastError() != cudaSuccess) break;
//
//         //count result unless last sim
//         if(i < strategy->sim_count - 1) 
//         {
//             read_results(cuda_results, total_simulations, &node_results);
//         }
//     }
//     
//     std::cout << "Simulation ran for: " << duration_cast<milliseconds>(steady_clock::now() - start).count() << "[ms] \n";
//
//     const cudaError status = cudaPeekAtLastError();
//     if(cudaPeekAtLastError() == cudaSuccess)
//     {
//         std::cout << "Reading results...\n";
//         read_results(cuda_results, total_simulations, &node_results);
//         print_results(&node_results, total_simulations * strategy->sim_count);
//     }
//     else
//     {
//         printf("An error occured during device execution" );
//         printf("CUDA error code: %d\n", status);
//         exit(status);  // NOLINT(concurrency-mt-unsafe)
//         return;
//     }
//     
//     cudaFree(cuda_results);
//     cudaFree(state);
//     cudaFree(options_d);
//     
//     for (void* it : free_list)
//     {
//         cudaFree(it);
//     }
// }