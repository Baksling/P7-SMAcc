#include "stochastic_engine.h"
#include "stochastic_simulator.h"
#include "thread_pool.h"
#include <map>
#include "device_launch_parameters.h"
#include <chrono>
#include "simulator_tools.h"
#include "../Domain/edge_t.h"
#include "../Domain/stochastic_model_t.h"

using namespace std::chrono;


CPU GPU void run_simulator(simulator_state state, curandState* r_state, const model_options* options)
{
    while (true)
    {
        model_state* current_model = state.progress_sim(options, r_state);

        if(current_model == nullptr || current_model->reached_goal == true)
        {
            break;
        }

        do //repeat as long as current node is branch node
        {
            lend_array<edge_t*> outgoing_edges =  current_model->current_node->get_edges();
            if(outgoing_edges.size() == 0) break;

            const edge_t* edge = simulator_tools::choose_next_edge(&state, &outgoing_edges, r_state);
            if(edge == nullptr)
            {
                break;
            }

            current_model->current_node = edge->get_dest();
            edge->execute_updates(&state);
        }
        while (current_model->current_node->is_branch_point());

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
    simulation_result* output,
    const unsigned long idx
)
{
    curandState* r_state = &random_states[idx];
    curand_init(options->seed, idx, idx, r_state);
    
    simulator_state state = simulator_state::from_multi_model(options->max_expression_depth, model);

    for (unsigned i = 0; i < options->simulation_amount; ++i)
    {
        const unsigned int sim_id = i + options->simulation_amount * static_cast<unsigned int>(idx);
        state.reset(sim_id, model);

        //run simulation
        run_simulator(state, r_state, options);

        state.write_result(output);
    }
    
    state.free_internals();
}

__global__ void gpu_simulate(
    const stochastic_model_t* model,
    const model_options* options,
    curandState* r_state,
    simulation_result* output
    )
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    simulate_stochastic_model(model, options, r_state, output, idx);
}

bool stochastic_engine::run_gpu(
    const stochastic_model_t* model,
    const model_options* options,
    simulation_result* output,
    const simulation_strategy* strategy)
{
    curandState* random_states = nullptr;
    cudaMalloc(&random_states, sizeof(curandState)*strategy->block_n*strategy->threads_n);
    
    if(cudaSuccess != cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8589934592))
    {
        printf("Could not allocate heap space on cuda device\n");
        return false;
    }
    //simulate on device
    gpu_simulate<<<strategy->block_n, strategy->threads_n>>>(model, options, random_states, output);
        
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
    simulation_result* output,
    const simulation_strategy* strategy)
{
    curandState* random_states = static_cast<curandState*>(malloc(sizeof(curandState)*strategy->degree_of_parallelism()));
    
    //init thread pool
    thread_pool pool(strategy->cpu_threads_n);

    //add all jobs
    for (unsigned i = 0; i < strategy->degree_of_parallelism(); i++)
    {
        pool.queue_job([model, options, random_states, output, i]()
        {
            simulate_stochastic_model(model, options, random_states, output, i);
        });
    }
    //Start processing jobs in pool.
    pool.start();
    
    while(pool.is_busy()) //wait for pool to process all tasks
    {
    }

    //stop pool
    pool.stop();
    free(random_states);
    
    return true;
}
