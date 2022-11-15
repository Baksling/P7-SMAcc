#include "simulator_state.h"
#include "node_t.h"
#include "edge_t.h"
#include "../Simulator/writers/result_manager.h"


CPU GPU double simulator_state::determine_progression(const node_t* node)
{
    double local_max_progress = 0.0;
    const bool has_upper_bound = node->max_time_progression(this, &local_max_progress);
    double time_progression;
    
    if(has_upper_bound)
    {
        if(local_max_progress < 0)
        {
            printf("Local max progression of '%lf' is less than 0. PANIC!\n", local_max_progress);
            time_progression = NO_PROGRESS;
        }
        else
        {
            time_progression = (1.0 - curand_uniform_double(this->random)) * local_max_progress;
        }
    }
    else
    {
        const double lambda = node->get_lambda(this);
        if(lambda <= 0)
        {
            //Defined behavior. If lambda <= 0, then just treat as disabled node.
            time_progression = NO_PROGRESS;
        }
        else
        {
            time_progression = (-log(curand_uniform_double(this->random))) / lambda;
        }
    }

    return time_progression;
}


CPU GPU void simulator_state::progress_timers(const double time)
{
    //update all timers by adding time_progression to each
    for(int i = 0; i < this->timers_.size(); i++)
    {
        this->timers_.at(i)->add_time(time);
    }

    //progress global timer.
    this->global_time_ += time;
}

simulator_state::simulator_state(
    curandState* random,
    const cuda_stack<expression*>& expression_stack,
    const cuda_stack<double>& value_stack)
{
    this->sim_id_ = static_cast<unsigned>(-1);
    this->steps_ = 0;
    this->global_time_ = 0.0;
    // this->models_ = models;
    // this->timers_ = timers;
    // this->variables_ = variables;
    this->expression_stack = expression_stack;
    this->value_stack = value_stack;
    this->random = random;
    // this->medium = medium;
}

CPU GPU model_state* simulator_state::progress_sim(const model_options* options)
{
    //determine if sim is done
    if((options->use_max_steps      && this->steps_       >= options->max_steps_pr_sim)
        || (!options->use_max_steps && this->global_time_ >= options->max_global_progression) )
        return nullptr;

    //progress number of steps
    this->steps_++;
    
    double min_progress_time = MAX_DOUBLE;
    if(!options->use_max_steps)
        min_progress_time = options->max_global_progression - this->global_time_;
    
    model_state* winning_model = nullptr;
    for (int i = 0; i < this->models_.size(); ++i)
    {
        model_state* current = this->models_.at(i);
        
        //if goal is reached, dont bother
        if(current->reached_goal) continue;

        
        //If all channels that are left is listeners, then dont bother
        //This also ensures that current_node has edges
        if(!current->current_node->is_progressible()) continue;
        
        //if it is not in a valid state, then it is disabled 
        if(!current->current_node->evaluate_invariants(this)) continue;

        //determine current models progress
        const double local_progress = this->determine_progression(current->current_node);

        //If negative progression, skip. Represents NO_PROGRESS
        if(local_progress < 0) continue;

        //Set current as winner, if it is the earliest active model.
        if(local_progress < min_progress_time)
        {
            min_progress_time = local_progress;
            winning_model = current;
        }
    }
    // printf(" I WON! Node: %d \n", winning_model->current_node->get_id());
    
    this->progress_timers(min_progress_time);
    return winning_model;
}

CPU GPU simulator_state simulator_state::init(
    const stochastic_model_t* multi_model,
    const model_options* options,
    curandState* random,
    void* memory_heap)
{
    const unsigned long long int thread_memory_size = options->get_cache_size();
    void* store = memory_heap;

    expression** expression_store = static_cast<expression**>(store);
    store = static_cast<void*>(&static_cast<expression**>(store)[options->get_expression_size()]);

    double* value_store = static_cast<double*>(store);
    store = static_cast<void*>(&static_cast<double*>(store)[options->max_expression_depth]);

    model_state* state_store = static_cast<model_state*>(store);
    store = static_cast<void*>(&static_cast<model_state*>(store)[options->model_count]);

    clock_variable* variable_store = static_cast<clock_variable*>(store);
    store = static_cast<void*>(&static_cast<clock_variable*>(store)[options->variable_count]);

    clock_variable* clock_store = static_cast<clock_variable*>(store);
    store = static_cast<void*>(&static_cast<clock_variable*>(store)[options->timer_count]);

    if((reinterpret_cast<unsigned long long>(memory_heap) + thread_memory_size) - reinterpret_cast<unsigned long long>(store) > 8)
            printf("Thread cache size not equivalent to utilized size %llu of %llu (diff %llu)",
                reinterpret_cast<unsigned long long>(store),
                (reinterpret_cast<unsigned long long>(memory_heap) + thread_memory_size),
                (reinterpret_cast<unsigned long long>(memory_heap) + thread_memory_size) - reinterpret_cast<unsigned long long>(store));

    //init state itself
    simulator_state state = simulator_state{
        random,
        cuda_stack<expression*>(expression_store, options->get_expression_size()), //needs to fit all each node twice (for left and right evaluation)
        cuda_stack<double>(value_store, options->max_expression_depth)
    };

    //init models
    // model_state* state_store = static_cast<model_state*>(malloc(sizeof(model_state)*multi_model->models_.size()));
    state.models_ = array_t<model_state>(state_store, multi_model->models_.size());
    for (int i = 0; i < multi_model->models_.size(); ++i)
    {
        state_store[i] = model_state{
            multi_model->models_.get(i),
            false
        }; 
    }

    //init clocks
    // clock_variable* clock_store = static_cast<clock_variable*>(malloc(sizeof(clock_variable)*multi_model->timers_.size()));
    state.timers_ = array_t<clock_variable>(clock_store, multi_model->timers_.size());
    for (int i = 0; i < multi_model->timers_.size(); ++i)
    {
        clock_store[i] = multi_model->timers_.at(i)->duplicate();
    }

    //init variables
    // clock_variable* variable_store = static_cast<clock_variable*>(malloc(sizeof(clock_variable)*multi_model->variables_.size()));
    state.variables_ = array_t<clock_variable>(variable_store, multi_model->variables_.size());
    for (int i = 0; i < multi_model->variables_.size(); ++i)
    {
        variable_store[i] = multi_model->variables_.at(i)->duplicate();
    }

    return state;
}

CPU GPU lend_array<clock_variable> simulator_state::get_timers() const
{
    return lend_array<clock_variable>(&this->timers_);
}

CPU GPU lend_array<clock_variable> simulator_state::get_variables() const
{
    return lend_array<clock_variable>(&this->variables_);
}

void simulator_state::broadcast_channel(const model_state* current_state, const unsigned channel_id, const result_manager* results)
{
    if(channel_id == NO_CHANNEL) return;
    
    for (int i = 0; i < this->models_.size(); ++i)
    {
        model_state* state = this->models_.at(i);
        if(current_state == state) continue; //skip current state
        if(state->reached_goal) continue; //skip if goal node
        if(!state->current_node->evaluate_invariants(this)) continue; //skip if node disabled
        
        lend_array<edge_t> edges = state->current_node->get_edges();
        
        //pick random start index in order to simulate random listener, when multiple listeners on same channel present
        const unsigned size = static_cast<unsigned>(edges.size());
        const unsigned start_index = curand(this->random) % size;
        
        for (unsigned j = 0; j < size; ++j)
        {
            const edge_t* edge = edges.at(static_cast<int>((start_index + j) % size)  );
            if(!edge->is_listener()) continue; //skip if not listener
            if(edge->get_channel() != channel_id) continue; //skip if not current channel
            if(!edge->evaluate_constraints(this)) continue; //skip if not valid
            
            node_t* dest = edge->get_dest();
            
            state->current_node = dest;
            state->reached_goal = dest->is_goal_node();
            
            edge->execute_updates(this);

            results->write_node_trace(state, this);
            break;
        }
    }
    
}

CPU GPU void simulator_state::reset(const unsigned int sim_id, const stochastic_model_t* model)
{
    //set sim_id to new sim and reset steps
    this->sim_id_ = sim_id;
    this->steps_ = 0;
    this->global_time_ = 0.0;

    //reset clear
    this->expression_stack.clear();
    this->value_stack.clear();
    
    //validate that timers match in size
    if(this->timers_.size() != model->timers_.size()
        || this->variables_.size() != model->variables_.size() )
    {
        printf("Timers or variable size mismatch in simulation %d", sim_id);
        return;
    }

    //reset models
    for (int i = 0; i < model->models_.size(); ++i)
    {
        this->models_.arr()[i] = model_state{
            model->models_.get(i),
            false
        }; 
    }

    //reset timers
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        const double start_time = model->timers_.at(i)->get_time();
        this->timers_.at(i)->reset_value(start_time);
    }

    //reset variables
    for (int i = 0; i < this->variables_.size(); ++i)
    {
        const double start_time = model->variables_.at(i)->get_time();
        this->variables_.at(i)->reset_value(start_time);
    }

    //reset channels
    // const lend_array<model_state> lend_states = lend_array<model_state>(&this->models_); 
    // this->medium->clear();
    // this->medium->init(&lend_states);
}
