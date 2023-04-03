#ifndef DYNAMIC_ENGINE_CU
#define DYNAMIC_ENGINE_CU

#include "../common/macro.h"
#include "Domain.cu"
#include "device_atomic_functions.h"
#include "../common/sim_config.h"
#include "../results/result_store.h"
#include "device_launch_parameters.h"


unsigned int atomicInc(unsigned int *address, unsigned int val);

struct thread_state
{
    unsigned worker_id;
    unsigned sim_count;
    unsigned workers;

    //i is the index of the state, while idx current process its looking at.
    CPU GPU bool is_done(const unsigned i) const
    {
        return i != worker_id;
    }

    CPU GPU void increment_sim_count(const unsigned idx)
    {
        if(idx == worker_id)
            sim_count++;
    }

    CPU GPU void join()
    {
        atomicInc(&workers, INT_MAX);
    }
};

#define NO_THREAD (-1)
#define THREAD_EXISTS(x) ((x) < 0)
inline int find_next_state(const unsigned idx, const thread_state* thread_array)
{
    for (int i = 0; i < 32; ++i)
    {
        const int index = (i + idx) % 32;
        const thread_state* job = &thread_array[index];
        if(!job->is_done(idx))
            return index;
    }
    return NO_THREAD;
}

#define NO_PROCESS (-1)
#define IS_NO_PROCESS(x) ((x) < 0)
CPU GPU int progress_sim(state* sim_state, const sim_config* config);
CPU GPU edge* pick_next_edge_stack(const arr<edge>& edges, state* state);



#define MUTEX(id, t) ((id) == (t))

CPU GPU void engine(
    const unsigned idx,
    const unsigned thread_id,
    const network* model,
    thread_state* thread,
    state* sim_state,
    const result_store* output,
    const sim_config* config)
{
    for (unsigned i = 0; i < config->sim_pr_thread; ++i)
    {
        const unsigned sim_id = config->sim_pr_thread * thread_id + i;
        sim_state->reset(sim_id, model, config->initial_urgent, config->initial_committed);
            
        while(true)
        {
            const int process = progress_sim(sim_state, config);
            if(IS_NO_PROCESS(process)) break;
            
            do
            {
                const node* current = sim_state->models.store[process];
                const edge* e = pick_next_edge_stack(current->edges, sim_state);
                if(e == nullptr) break;

                if(MUTEX(idx, thread_id))
                {
                    sim_state->traverse_edge(process, e->dest);
                }
                    
                e->apply_updates(sim_state);
                sim_state->broadcast_channel(e->channel, process);
            } while (sim_state->models.store[process]->type == node::branch);
        }
        output->write_output(idx, sim_state);
            
    }
    
    
}

#endif


__global__ void dynamic_gpu_engine(
    const network* model,
    const result_store* output,
    void* state_cache,
    const sim_config* config)
{
    extern __shared__ char shared_mem[];
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(config->seed, idx, idx, &config->random_state_arr[idx]);
    thread_state* thread_array =  static_cast<thread_state*>(config->use_shared_memory ? shared_mem : state_cache);
    // size_t global_size = config->thread_heap_size();
    state* state_array = static_cast<state*>(config->cache);
    void* cache = &state_array[static_cast<int>(config->blocks*config->threads)];

    thread_array[idx] = thread_state{idx, config->sim_pr_thread, 1};
    
    cache = static_cast<void*>(
        &static_cast<char*>(
            static_cast<void*>(cache))
            [(idx*config->thread_heap_size()) / sizeof(char)]);

    state_array[idx] = state::init(cache, &config->random_state_arr[idx], model,
        config->max_expression_depth, config->max_backtrace_depth, config->max_edge_fanout);

    while(true)
    {
        const int id = find_next_state(idx, thread_array);
        if(THREAD_EXISTS(id)) return;

        thread_state* thread = &thread_array[id];
        thread->join();
        state* sim_state = &state_array[id];

        engine(idx, id, model, thread, sim_state, output, config);
    }
}


