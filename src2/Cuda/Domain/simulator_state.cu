#include "simulator_state.h"

#include "node_t.h"
#include "../Simulator/simulation_strategy.h"
#include "expressions/expression.h"



CPU GPU double simulator_state::determine_progression(const node_t* node, curandState* r_state)
{
    double local_max_progress = 0.0;
    const bool has_upper_bound = node->max_time_progression(this, &local_max_progress);
    double time_progression;
    
    if(has_upper_bound)
    {
        if(local_max_progress < 0)
        {
            printf("Local max progression of '%lf' is less than 0\n", local_max_progress);
            time_progression = 0.0;
        }
        else
        {
            time_progression = (1.0 - curand_uniform_double(r_state)) * local_max_progress;
        }
    }
    else
    {
        const double lambda = node->get_lambda(this);
        if(lambda <= 0)
        {
            printf("Lambda value %lf is negative or zero! PANIC!\n", lambda);
            time_progression = 0.0;
        }
        else
        {
            time_progression = (-log(curand_uniform_double(r_state))) / lambda;
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
    this->global_timer_.add_time(time);
}

simulator_state::simulator_state(
    const cuda_stack<expression*>& expression_stack,
    const cuda_stack<double>& value_stack)
{
    this->sim_id_ = static_cast<unsigned>(-1);
    this->steps_ = 0;
    this->global_timer_ = clock_variable(GLOBAL_TIMER_ID, 0.0);
    // this->models_ = models;
    // this->timers_ = timers;
    // this->variables_ = variables;
    this->expression_stack = expression_stack;
    this->value_stack = value_stack;
}

CPU GPU model_state* simulator_state::progress_sim(const model_options* options, curandState* r_state)
{
    //progress number of steps
    this->steps_++;
    
    //determine if sim is done
    if(this->steps_ > options->max_steps_pr_sim ) return nullptr;

    double min_progress_time = 99999999; //TODO find hardware function for this limit
    model_state* winning_model = nullptr;
    for (int i = 0; i < this->models_.size(); ++i)
    {
        model_state* current = this->models_.at(i);
        
        //if goal is reached, dont bother
        if(current->reached_goal) continue;

        //if edge has no outgoing edges, then dont bother
        if(current->current_node->get_edges().size() == 0) continue;
        
        //if it is not in a valid state, then it is disabled 
        if(!current->current_node->evaluate_invariants(this)) continue;

        //determine current models progress
        const double local_progress = this->determine_progression(current->current_node, r_state);

        if(local_progress < 0)
        {
            printf("local progress of '%lf' found to be less then 0", local_progress);
        }

        //Set current as winner, if it is the earliest active model.
        if(local_progress < min_progress_time)
        {
            min_progress_time = local_progress;
            winning_model = current;
        }
    }

    return winning_model;
}

CPU GPU double simulator_state::evaluate_expression(expression* expr)
{
    this->value_stack.clear();
    this->expression_stack.clear();
    
    expression* current = expr;
    while (true)
    {
        while(current != nullptr)
        {
            this->expression_stack.push(current);
            this->expression_stack.push(current);

            // if(!current->is_leaf()) //only push twice if it has children
            //      this->expression_stack_->push(current);
            current = current->get_left();
        }
        if(this->expression_stack.is_empty())
        {
            break;
        }
        current = this->expression_stack.pop();
        
        if(!this->expression_stack.is_empty() && this->expression_stack.peak() == current)
        {
            current = current->get_right(&this->value_stack);
        }
        else
        {
            current->evaluate(this);
            current = nullptr;
        }
    }

    if(this->value_stack.count() == 0)
    {
        printf("Expression evaluation ended in no values! PANIC!\n");
        return 0;
    }
    return this->value_stack.pop();
}

CPU GPU void simulator_state::write_result(simulation_result* output_array) const
{
    simulation_result* output = &output_array[this->sim_id_];

    output->total_time_progress = this->global_timer_.get_time();
    output->steps = this->steps_;

    for (int i = 0; i < this->models_.size(); ++i)
    {
        output->end_node_id_arr[i] = this->models_.at(i)->reached_goal
            ? this->models_.at(i)->current_node->get_id()
            : HIT_MAX_STEPS;
    }

    // for (int i = 0; i < this->models_.size(); ++i)
    // {
    //     printf("result: %d\n", output->end_node_id_arr[i]);
    // }

    for (int i = 0; i < this->variables_.size(); ++i)
    {
        output->variables_max_value_arr[i] = this->variables_.at(i)->get_max_value();
    }
}

CPU GPU void simulator_state::free_internals()
{
    this->expression_stack.free_internal();
    this->value_stack.free_internal();
    this->timers_.free_array();
    this->variables_.free_array();
    this->models_.free_array();
}

CPU GPU simulator_state simulator_state::from_multi_model(
    const unsigned int expression_max_depth,
    const stochastic_model_t* multi_model)
{
    //TODO! Optimize this function by only calling malloc once!
    
    //init state itself
    simulator_state state = {
        cuda_stack<expression*>(expression_max_depth*2+1), //needs to fit all each node twice (for left and right evaluation)
        cuda_stack<double>(expression_max_depth)
    };

    //init models
    model_state* state_store = static_cast<model_state*>(malloc(sizeof(model_state)*multi_model->models_.size()));
    state.models_ = array_t<model_state>(state_store, multi_model->models_.size());
    for (int i = 0; i < multi_model->models_.size(); ++i)
    {
        state_store[i] = model_state{
            multi_model->models_.at(i),
            false
        }; 
    }

    //init clocks
    clock_variable* clock_store = static_cast<clock_variable*>(malloc(sizeof(clock_variable)*multi_model->timers_.size()));
    state.timers_ = array_t<clock_variable>(clock_store, multi_model->timers_.size());
    for (int i = 0; i < multi_model->timers_.size(); ++i)
    {
        clock_store[i] = multi_model->timers_.at(i)->duplicate();
    }

    //init variables
    clock_variable* variable_store = static_cast<clock_variable*>(malloc(sizeof(clock_variable)*multi_model->variables_.size()));
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

CPU GPU void simulator_state::reset(const unsigned int sim_id,
                            const stochastic_model_t* model)
{
    //set sim_id to new sim and reset steps
    this->sim_id_ = sim_id;
    this->steps_ = 0;
    this->global_timer_.set_time(0);

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
            model->models_.at(i),
            false
        }; 
    }

    //reset timers
    for (int i = 0; i < this->timers_.size(); ++i)
    {
        const double start_time = this->timers_.at(i)->get_time();
        this->timers_.at(i)->set_time(start_time);
    }

    //reset variables
    for (int i = 0; i < this->variables_.size(); ++i)
    {
        const double start_time = this->variables_.at(i)->get_time();
        this->variables_.at(i)->set_time(start_time);
    }
}
