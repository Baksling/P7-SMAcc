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

CPU GPU double determine_progress(const node* node, state* state)
{
    double lower, upper = 0.0;
    const bool is_finite = node->progress_bounds(state, &lower, &upper);
    const double random = curand_uniform_double(state->random);
    const double lambda = node->lamda->evaluate_expression(state);

    //TODO throw error
    // if(lower > upper || lower < 0.0 || upper < 0.0)
    // {
    //     asm("TODO kill simulation. Report good error");
    // }
    
    if(is_finite)
    {
        return ((1.0 - random) * (upper - lower)) + lower;
    }
    
    return -log(random) / (lambda - (lambda == 0.0));
}

#define NO_PROCESS (-1)
#define IS_NO_PROCESS(x) ((x) < 0)
CPU GPU int progress_sim(state* sim_state, const sim_config* config)
{
    if(sim_state->global_time >= config->max_global_progression)
        return NO_PROCESS;

    sim_state->steps++;
    
    double max_progress = DBL_MAX;
    int winning_process = NO_PROCESS;

    for (int i = 0; i < sim_state->models.size; ++i)
    {
        const node* current = sim_state->models.store[i];

        const double local_progress = determine_progress(current, sim_state);
        
        if(local_progress >= 0.0 && local_progress < max_progress)
        {
            winning_process = i;
            max_progress = local_progress;
        }
    }
    if(IS_NO_PROCESS(winning_process)) return NO_PROCESS;

    if(sim_state->urgent_count == 0)
    {
        for (int i = 0; i < sim_state->variables.size; ++i)
        {
            sim_state->variables.store[i].add_time(max_progress);
        }
    }

    return winning_process;
}

CPU GPU edge* pick_edge(const arr<edge>& edges, state* state)
{
    const unsigned init = curand(state->random) % edges.size;

    for (int i = 0; i < edges.size; ++i)
    {
        const int index = (i + init) % edges.size;
        if(constraint::evaluate_constraint_set(edges.store[index].guards, state))
        {
            return &edges.store[index];
        }
    }

    //TODO throw error
    return nullptr;
}

CPU GPU node* pick_destination(const arr<edge::branch> branches, state* state)
{
    if(branches.size == 1) return branches.store->dest;
    state->edge_stack.clear();

    double weight_sum = 0.0;
    for (int i = 0; i < branches.size; ++i)
    {
        edge::branch* branch = &branches.store[i];
        const double weight = branch->weight->evaluate_expression(state);
        if(weight <= 0.0) continue;

        state->edge_stack.push_val(state::w_edge{branch, weight});
        weight_sum += weight;
    }

    const double r_val = (1.0 - curand_uniform_double(state->random)) * weight_sum;
    double weight_acc = 0.0;

    node* output = nullptr;
    while (state->edge_stack.count() > 0)
    {
        const state::w_edge e = state->edge_stack.pop();
        output = e.e->dest;
        weight_acc += e.w;

        if(weight_acc >= r_val) break;
    }
    
    return output;
}

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
            
            const node* current = sim_state->models.store[process];
            const edge* e = pick_edge(current->edges, sim_state);
            node* dest = pick_destination(e->branches, sim_state);
            
            if(MUTEX(idx, thread_id))
            {
                sim_state->traverse_edge(process, dest);
            }
                    
            e->apply_updates(sim_state);
            sim_state->broadcast_channel(e->channel, process);
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


