#include "Domain.h"


//Please do not change the argument names, they are required for JIT compilation 
CPU GPU double evaluate_compiled_expression(const expr* ex, state* state)
{
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    //__SEARCH_MARKER_FOR_JIT_EXPRESSION__
    
    return 0.0;
}

//Please do not change the argument names, they are required for JIT compilation 
CPU GPU bool evaluate_compiled_constraint(const constraint* con, state* state)
{
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    //__SEARCH_MARKER_FOR_JIT_CONSTRAINT__
    
    return false;
}

//Please do not change the argument names, they are required for JIT compilation 
CPU GPU double evaluate_compiled_constraint_upper_bound(const constraint* con, state* state, bool* is_finite)
{
    //If this variable is marked const, then JIT compilation will not work.
    // ReSharper disable once CppLocalVariableMayBeConst
    double v0 = DBL_MAX;
    
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    //__SEARCH_MARKER_FOR_JIT_INVARIANTS__

    *is_finite = v0 < DBL_MAX; 
    return v0;
}

CPU GPU double evaluate_expression_node(const expr* expr, state* state)
{
    double v1, v2;
    switch (expr->operand) {
    case expr::literal_ee:
        return expr->value;
    case expr::clock_variable_ee:
        return state->variables.store[expr->variable_id].value;
    case expr::random_ee:
        v1 = state->value_stack.pop();
        return (1.0 - curand_uniform_double(state->random)) * v1;
    case expr::plus_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 + v2;
    case expr::minus_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 - v2;
    case expr::multiply_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 * v2;
    case expr::division_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 / v2;
    case expr::power_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return pow(v1, v2);
    case expr::negation_ee: 
        v1 = state->value_stack.pop();
        return -v1;
    case expr::sqrt_ee: 
        v1 = state->value_stack.pop();
        return sqrt(v1);
    case expr::modulo_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return static_cast<double>(static_cast<int>(v1) % static_cast<int>(v2));
    case expr::less_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 <= v2;
    case expr::greater_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 >= v2;
    case expr::less_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 < v2;
    case expr::greater_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 > v2;
    case expr::equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return abs(v1 - v2) <= DBL_EPSILON;
    case expr::not_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return abs(v1 - v2) > DBL_EPSILON;
    case expr::not_ee:
        v1 = state->value_stack.pop();
        return (abs(v1) < DBL_EPSILON);
    case expr::conditional_ee:
        v1 = state->value_stack.pop();
        state->value_stack.pop();
        return v1;
    case expr::compiled_ee: return 0.0; break;
    }
    return 0.0;
}

CPU GPU double expr::evaluate_expression(state* state)
{
    if(this->operand == literal_ee)
        return this->value;
    if(this->operand == clock_variable_ee)
        return state->variables.store[this->variable_id].value;
    if(this->operand == compiled_ee)
        return evaluate_compiled_expression(this, state);

    state->expr_stack.clear();
    state->value_stack.clear();
    expr* current = this;
    while (true)
    {
        while(current != nullptr)
        {
            state->expr_stack.push(current);
            
            if(!IS_LEAF(current->operand)) //only push twice if it has children
                state->expr_stack.push(current);
            
            current = current->left;
        }
        if(state->expr_stack.count() == 0)
        {
            break;
        }
        current = state->expr_stack.pop();
        
        if(state->expr_stack.count() > 0 && state->expr_stack.peak() == current)
        {
            current = (current->operand == conditional_ee && abs(state->value_stack.peak()) < DBL_EPSILON)
                ? current->conditional_else
                : current->right;
        }
        else
        {
            double val = evaluate_expression_node(current, state);
            state->value_stack.push(val);
            current = nullptr;
        }
    }

    if(state->value_stack.count() == 0)
    {
        // printf("Expression evaluation ended in no values! PANIC!\n");
        return 0.0;
    }
    
    return state->value_stack.pop();
}


CPU GPU double node::max_progression(state* state, bool* is_finite) const
{
    double max_bound = DBL_MAX;

    for (int i = 0; i < this->invariants.size; ++i)
    {
        const constraint con = this->invariants.store[i];
        double limit;
        
        if(IS_INVARIANT(con.operand))
        {
            if(!con.uses_variable) continue;
            const clock_var var = state->variables.store[con.variable_id];
            if(var.rate == 0) continue;
            limit = (con.expression->evaluate_expression(state) - var.value) / var.rate;
        }
        else if(con.operand == constraint::compiled_c)
        {
            bool finite = false;
            limit = evaluate_compiled_constraint_upper_bound(&con, state, &finite);

            if(!finite) continue;
        }
        else continue;
        max_bound = fmin(max_bound,  limit); //rate is >0.
    }
    *is_finite = max_bound < DBL_MAX;
    return max_bound;
}

CPU GPU bool constraint::evaluate_constraint(state* state) const
{
    if(this->operand == compiled_c)
        return evaluate_compiled_constraint(this, state);
    const double left = this->uses_variable
        ? state->variables.store[this->variable_id].value
        : this->value->evaluate_expression(state);
    const double right = this->expression->evaluate_expression(state);

    switch (this->operand)
    {
    case less_equal_c: return left <= right;
    case less_c: return left < right;
    case greater_equal_c: return left >= right;
    case greater_c: return left > right;
    case equal_c: return left == right;  // NOLINT(clang-diagnostic-float-equal)
    case not_equal_c: return left != right;  // NOLINT(clang-diagnostic-float-equal)
    case compiled_c: return false;
    }
    return false;
}


CPU GPU bool constraint::evaluate_constraint_set(const arr<constraint>& con_arr, state* state)
{
    for (int i = 0; i < con_arr.size; ++i)
    {
        if(!con_arr.store[i].evaluate_constraint(state))
            return false;
    }
    return true;
}

void clock_var::add_time(const double time)
{
    this->value += time*this->rate;
    this->max_value = fmax(this->max_value, this->value);
}

void clock_var::set_value(const double val)
{
    this->value = val;
    this->max_value = fmax(this->max_value, this->value);
}

CPU GPU inline void update::apply_update(state* state) const
{
    const double value = this->expression->evaluate_expression(state);
    state->variables.store[this->variable_id].set_value(value);
}

CPU GPU inline void edge::apply_updates(state* state) const
{
    for (int i = 0; i < this->updates.size; ++i)
    {
        this->updates.store[i].apply_update(state);
    }
}

CPU GPU inline bool edge::edge_enabled(state* state) const
{
    for (int i = 0; i < this->guards.size; ++i)
    {
        if(!this->guards.store[i].evaluate_constraint(state))
            return false;
    }
    return true;
}

CPU GPU void state::traverse_edge(const int process_id, node* dest)
{
    const node* current = this->models.store[process_id];

    this->urgent_count -= IS_URGENT(current->type);
    this->committed_count -= current->type == node::committed;

    this->urgent_count += IS_URGENT(dest->type);
    this->committed_count += dest->type == node::committed;
    
    this->models.store[process_id] = dest;
}

void inline state::broadcast_channel(const int channel, const int process)
{
    if(!IS_BROADCASTER(channel)) return;
    
    for (int p = 0; p < this->models.size; ++p)
    {
        const node* current = this->models.store[p];
        
        if (p == process) continue;
        if(current->type == node::goal) continue;
        if(!constraint::evaluate_constraint_set(current->invariants, this)) continue;
        
        const unsigned offset = curand(this->random) % current->edges.size;
        
        for (int e = 0; e < current->edges.size; ++e)
        {
            const edge current_e = current->edges.store[(e + offset) % current->edges.size];
            if(!IS_LISTENER(current_e.channel)) continue;
            if(!CAN_SYNC(channel, current_e.channel)) continue;

            this->traverse_edge(p, current_e.dest);

            current_e.apply_updates(this);
            break;
        }
    }
}

state state::init(void* cache, curandState* random, const network* model, const unsigned expr_depth, const unsigned fanout)
{
    node** nodes = static_cast<node**>(cache);
    cache = static_cast<void*>(&nodes[model->automatas.size]);
        
    clock_var* vars = static_cast<clock_var*>(cache);
    cache = static_cast<void*>(&vars[model->variables.size]);
        
    expr** exp = static_cast<expr**>(cache);
    cache = static_cast<void*>(&exp[expr_depth*2+1]);
        
    double* val_store = static_cast<double*>(cache);
    cache = static_cast<void*>(&val_store[expr_depth]);

    state::w_edge* fanout_store = static_cast<state::w_edge*>(cache);
    // cache = static_cast<void*>(&cache[fanout]);
    
    
    return state{
        0,
        0,
        0,
        0,
        0.0,
        arr<node*>{ nodes, model->automatas.size },
        arr<clock_var>{ vars, model->variables.size },
        random,
        my_stack<expr*>(exp, static_cast<int>(expr_depth*2+1)),
        my_stack<double>(val_store, static_cast<int>(expr_depth)),
        my_stack<state::w_edge>(fanout_store, static_cast<int>(fanout))
    };
}

void state::reset(const unsigned sim_id, const network* model)
{
    this->simulation_id = sim_id;
    this->steps = 0;
    this->global_time = 0.0;
    this->urgent_count = 0;
    this->committed_count = 0;
    for (int i = 0; i < model->automatas.size; ++i)
    {
        this->models.store[i] = model->automatas.store[i];
    }

    for (int i = 0; i < model->variables.size; ++i)
    {
        this->variables.store[i] = model->variables.store[i];
    }
}
