#include "stochastic_engine.h"
#include "stochastic_simulator.h"
#include "thread_pool.h"
#include "device_launch_parameters.h"
#include <chrono>
#include "simulator_tools.h"
#include "../Domain/edge_t.h"
#include "../Domain/stochastic_model_t.h"
#include "writers/result_manager.h"


using namespace std::chrono;


CPU GPU void run_simulator(simulator_state* state, result_manager* trace_tracker, const model_options* options)
{
    while (true)
    {
        model_state* current_model = state->progress_sim(options);

        if(current_model == nullptr || current_model->reached_goal)
        {
            break;
        }

        do //repeat as long as current node is branch node
        {
            lend_array<edge_t> outgoing_edges =  current_model->current_node->get_edges();
            if(outgoing_edges.size() == 0) break;

            const edge_t* edge = simulator_tools::choose_next_edge_bit(state, &outgoing_edges, state->random);
            if(edge == nullptr)
            {
                break;
            }
            
            current_model->current_node = edge->get_dest();
            edge->execute_updates(state);
            state->broadcast_channel(current_model, edge->get_channel(), trace_tracker);
        }
        while (current_model->current_node->is_branch_point());

        trace_tracker->write_step_trace(current_model, state);
        
        if(current_model->current_node->is_goal_node())
        {
            current_model->reached_goal = true;
        }
    }
}

CPU GPU void simulate_stochastic_model(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* random_states,
    result_manager* output,
    const unsigned long idx,
    void* memory_heap
)
{
    curandState* r_state = &random_states[idx];
    curand_init(options->seed, idx, idx, r_state);
    
    simulator_state state = simulator_state::init(model, options, r_state, memory_heap);
    
    
    for (unsigned i = 0; i < options->simulation_amount; ++i)
    {
        const unsigned int sim_id = i + options->simulation_amount * static_cast<unsigned int>(idx);
        state.reset(sim_id, model);

        //run simulation
        run_simulator(&state, output, options);

        output->write_result(&state);
    }
}

__global__ void gpu_simulate(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    result_manager* output,
    void* total_memory_heap
    )
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned long long int thread_memory_size = options->get_cache_size();
    const unsigned long long int offset = (idx * thread_memory_size) / sizeof(char);
    simulate_stochastic_model(model, options, r_state, output, idx, static_cast<void*>(&static_cast<char*>(total_memory_heap)[offset]));
}

bool stochastic_engine::run_gpu(
    const stochastic_model_t* model,
    const model_options* options,
    result_manager* output,
    const simulation_strategy* strategy,
    void* total_memory_heap)
{
    curandState* random_states = nullptr;
    cudaMalloc(&random_states, sizeof(curandState)*strategy->block_n*strategy->threads_n);
    
    //simulate on device
    gpu_simulate<<<strategy->block_n, strategy->threads_n>>>(model, options, random_states, output, total_memory_heap);
        
    //wait for all processes to finish
    cudaDeviceSynchronize();
    
    const cudaError success = cudaPeekAtLastError();
    if(success != cudaSuccess) printf("\nAn error of code '%d' occured in cuda :( \n", success);
    cudaFree(random_states);

    return success == cudaSuccess;
}

bool stochastic_engine::run_cpu(
    const stochastic_model_t* model,
    const model_options* options,
    result_manager* output,
    const simulation_strategy* strategy,
    void* total_memory_heap)
{
    curandState* random_states = static_cast<curandState*>(malloc(sizeof(curandState)*strategy->degree_of_parallelism()));

    const unsigned long long int thread_memory_size = options->get_cache_size();
    //init thread pool
    thread_pool pool(strategy->cpu_threads_n);

    //add all jobs
    for (unsigned i = 0; i < strategy->degree_of_parallelism(); i++)
    {
        unsigned long long int offset = (i * thread_memory_size) / sizeof(char);
        pool.queue_job([model, options, random_states, output, i, total_memory_heap, offset]()
        {
            simulate_stochastic_model(model, options, random_states, output, i, static_cast<void*>(&static_cast<char*>(total_memory_heap)[offset]));
        });
    }
    //Start processing jobs in pool.
    pool.start();
    
    while(pool.is_busy()) //wait for pool to process all tasks
    {
        std::this_thread::yield();
    }

    //stop pool
    pool.stop();
    free(random_states);
    
    return true;
}
