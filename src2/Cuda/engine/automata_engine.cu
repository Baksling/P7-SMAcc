
#include "../common/macro.h"
#include "Domain.cu"
#include "model_oracle.cu"
#include "../common/sim_config.h"
#include "../results/result_store.h" 
#include "device_launch_parameters.h"

CPU GPU size_t thread_heap_size(const sim_config* config)
{
    const size_t size =
          static_cast<size_t>(config->max_expression_depth*2+1) * sizeof(void*) + //this is a expression*, but it doesnt like sizeof(expression*)
          config->max_expression_depth * sizeof(double) +
          config->network_size * sizeof(node) +
          config->variable_count * sizeof(clock_var) +
          config->max_edge_fanout * sizeof(state::w_edge);

    const unsigned long long int padding = (8 - (size % 8));

    return padding < 8 ? size + padding : size;
}  

CPU GPU double determine_progress(const node* node, state* state)
{
    bool is_finite = true;
    const double random_val = curand_uniform_double(state->random);
    const double max = node->max_progression(state, &is_finite);
    const double lambda = node->lamda->evaluate_expression(state);

    if(is_finite)
    {
        return (1.0 - random_val) * max;
    }
    else
    {
        // return lambda > 0.0 ? -log(random_val) / (lambda) : lambda;
        return (-log(random_val)) / (lambda - (lambda == 0.0));
    }
}

CPU GPU inline bool can_progress(const node* n)
{
    //#No brackets gang!
    for (int i = 0; i < n->edges.size; ++i)
        if(!IS_LISTENER(n->edges.store[i].channel))
            return true;
    return false;
} 

CPU GPU node** progress_sim(state* sim_state, const sim_config* config)
{
    //determine if sim is done

    // if(config->use_max_steps * sim_state->steps  >= config->max_steps_pr_sim
    //     + !config->use_max_steps * sim_state->global_time >= config->max_global_progression)
    
    if((config->use_max_steps && sim_state->steps  >= config->max_steps_pr_sim)
        || (!config->use_max_steps && sim_state->global_time >= config->max_global_progression) )
            return nullptr;

    //progress number of steps
    sim_state->steps++;

    // const double max_progression_time = config->use_max_steps
    //                                         ? DBL_MAX
    //                                         : config->max_global_progression - sim_state->global_time;

    const double max_progression_time = ((config->use_max_steps) * DBL_MAX)
                + ((!config->use_max_steps) * (config->max_global_progression - sim_state->global_time));

    double min_progression_time = max_progression_time;
    
    node** winning_model = nullptr;
    for (int i = 0; i < sim_state->models.size; ++i)
    {
        const node* current = sim_state->models.store[i];
        
        //if goal is reached, dont bother
        if(current->is_goal) continue;
        
        //If all channels that are left is listeners, then dont bother
        //This also ensures that current_node has edges
        if(!can_progress(current)) continue;
        
        //if it is not in a valid state, then it is disabled 
        if(!constraint::evaluate_constraint_set(current->invariants, sim_state)) continue;

        
        //determine current models progress
        const double local_progress = determine_progress(current, sim_state);

        // printf("progress %lf\n", local_progress);
        //If negative progression, skip. Represents NO_PROGRESS
        //Set current as winner, if it is the earliest active model.
        if(local_progress >= 0.0 && local_progress < min_progression_time)
        {
            min_progression_time = local_progress;
            winning_model = &sim_state->models.store[i];
        }
    }
    // printf(" I WON! Node: %d \n", winning_model->current_node->get_id());
    if(min_progression_time < max_progression_time)
    {
        for (int i = 0; i < sim_state->variables.size; ++i)
        {
            sim_state->variables.store[i].add_time(min_progression_time);
        }
        sim_state->global_time += min_progression_time;
    }

    return winning_model;
}
#define BIT_IS_SET(n, i) ((n) & (1UL << (i)))
#define SET_BIT(n, i) (n) |= (1UL << (i)) 

CPU GPU edge* pick_next_edge(const arr<edge>& edges, state* state)
{
    //TODO set max nr. of outgoing edges to 64 - done, it is ignored, by takign the last valid, if > 64 edges exists.
    // const int edge_amount = umin(edges->size, sizeof(unsigned long long)*8);
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge* valid_edge = nullptr;
    double weight_sum = 0.0;
    
    for (int i = 0; i < edges.size; ++i)
    {
        edge* e = &edges.store[i];
        if(IS_LISTENER(e->channel)) continue;
        if(!constraint::evaluate_constraint_set(e->guards, state)) continue;
        
        const double weight = e->weight->evaluate_expression(state);
        //only consider edge if it its weight is positive.
        //Negative edge value is semantically equivalent to disabled.
        if(weight <= 0.0) continue;
        SET_BIT(valid_edges_bitarray, i);
        valid_edge = e;
        valid_count++;
        weight_sum += weight; 
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;
    if(static_cast<size_t>(edges.size) > sizeof(size_t)*8) return valid_edge;

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(state->random)) * weight_sum;
    double r_acc = 0.0;

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges.size; ++i)
    {
        if(!BIT_IS_SET(valid_edges_bitarray, i)) continue;
        const double weight = edges.store[i].weight->evaluate_expression(state);
        if(weight <= 0.0) continue;
        
        valid_edge = &edges.store[i];
        r_acc += weight;
        if(r_val < r_acc) break;
    }
    return valid_edge;
}

CPU GPU edge* pick_next_edge_stack(const arr<edge>& edges, state* state)
{
    state->edge_stack.clear();
    int valid_count = 0;
    state::w_edge valid_edge = {nullptr, 0.0};
    double weight_sum = 0.0;
    
    for (int i = 0; i < edges.size; ++i)
    {
        edge* e = &edges.store[i];
        if(IS_LISTENER(e->channel)) continue;
        if(!constraint::evaluate_constraint_set(e->guards, state)) continue;
        
        const double weight = e->weight->evaluate_expression(state);
        //only consider edge if it its weight is positive.
        //Negative edge value is semantically equivalent to disabled.
        if(weight <= 0.0) continue;
        valid_edge = state::w_edge{ e, weight };
        valid_count++;
        weight_sum += weight;
        state->edge_stack.push(valid_edge);
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1) return valid_edge.e;

    const double r_val = (1.0 - curand_uniform_double(state->random)) * weight_sum;
    double r_acc = 0.0;

    //pick the weighted random value.
    valid_edge = { nullptr, 0.0 }; //reset valid edge !IMPORTANT
    for (int i = 0; i < valid_count; ++i)
    {
        valid_edge = state->edge_stack.pop();
        r_acc += valid_edge.w;
        if(r_val < r_acc) break;
    }

    return valid_edge.e;
}

CPU GPU void simulate_automata(
    const unsigned idx,
    const network* model,
    const result_store* output,
    const sim_config* config)
{
    void* cache = static_cast<void*>(&static_cast<char*>(config->cache)[(idx*thread_heap_size(config)) / sizeof(char)]);
    curandState* r_state = &config->random_state_arr[idx];
    curand_init(config->seed, idx, idx, r_state);
    state sim_state = state::init(cache, r_state, model, config->max_expression_depth, config->max_edge_fanout);
    
    for (unsigned i = 0; i < config->simulation_amount; ++i)
    {
        const unsigned int sim_id = i + config->simulation_amount * static_cast<unsigned int>(idx);
        sim_state.reset(sim_id, model);
        
        //run simulation
        while (true)
        {
            node** state = progress_sim(&sim_state, config);
            
            if(state == nullptr || (*state)->is_goal) break;
            
            do
            {
                const edge* e = pick_next_edge_stack((*state)->edges, &sim_state);
                if(e == nullptr) break;

                *state = e->dest;
                e->apply_updates(&sim_state);
                sim_state.broadcast_channel(e->channel, *state);
            } while ((*state)->is_branch_point);
        }
        output->write_output(&sim_state);
    }
}

__global__ void simulator_gpu_kernel(
    const model_oracle* oracle,
    const result_store* output,
    const sim_config* config)
{
    // ReSharper disable once CppTooWideScope
    extern __shared__ char shared_mem[];
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    network* model;
    if(config->use_shared_memory)
    {
        model = oracle->move_to_shared_memory(shared_mem, static_cast<int>(config->threads));
    }
    else
    {
        model = oracle->network_point();
    }
    cuda_SYNCTHREADS();

    simulate_automata(idx, model, output, config);
}