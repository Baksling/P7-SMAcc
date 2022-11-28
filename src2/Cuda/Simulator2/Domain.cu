#include "Domain.h"

#define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0

CPU GPU double evaluate_expression_node(const expr* expr, state* state)
{
    double v1, v2;
    switch (expr->operand) {
    case expr::literal_ee:
        return expr->value;
    case expr::clock_variable_ee:
        return state->variables.store[expr->variable_id].temp_value;
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
        return abs(v1) < DBL_EPSILON * 1.0;
    case expr::conditional_ee:
        v1 = state->value_stack.pop();
        state->value_stack.pop();
        return v1;
    default: return 0.0;
    }
}

CPU GPU double expr::evaluate_expression(state* state)
{
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
        printf("Expression evaluation ended in no values! PANIC!\n");
    }
    
    return state->value_stack.pop();
}


CPU GPU double node::max_progression(state* state, bool* is_finite) const
{
    double max_bound = INFINITY;

    for (int i = 0; i < this->invariants.size; ++i)
    {
        const constraint* con = &this->invariants.store[i];
        if(!IS_INVARIANT(con->operand)) continue;
        if(!con->uses_variable) continue;
        const clock_var var = state->variables.store[con->variable_id];
        if(var.rate == 0) continue;
        const double time = state->variables.store[con->variable_id].value;
        const double expr_value = con->expression->evaluate_expression(state);
        
        max_bound = fmin(max_bound,  (expr_value - time) / var.rate); //rate is >0.
    }
    *is_finite = isfinite(max_bound);
    return max_bound;
}

CPU GPU bool constraint::evaluate_constraint(state* state) const
{
    const double left = state->variables.store[this->variable_id].value;
    const double right = this->expression->evaluate_expression(state);

    switch (this->operand)
    {
    case constraint::less_equal_c: return left <= right;
    case constraint::less_c: return left < right;
    case constraint::greater_equal_c: return left >= right;
    case constraint::greater_c: return left < right;
    case constraint::equal_c: return left == right;  // NOLINT(clang-diagnostic-float-equal)
    case constraint::not_equal_c: return left != right;  // NOLINT(clang-diagnostic-float-equal)
    default: return false;
    }
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

CPU GPU inline void update::apply_temp_update(state* state) const
{
    const double value = this->expression->evaluate_expression(state);
    state->variables.store[this->variable_id].temp_value = value;
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

    for (int i = 0; i < this->updates.size; ++i)
    {
        this->updates.store[i].apply_temp_update(state);
    }

    bool is_valid = true;
        
    for (int i = 0; i < this->dest->invariants.size; ++i)
    {
        if(this->dest->invariants.store[i].evaluate_constraint(state))
        {
            is_valid = false;
            break;
        }
    }

    //this is always <= state.variables.size
    for (int i = 0; i < this->updates.size; ++i)
    {
        const int id = this->updates.store[i].variable_id;
        state->variables.store[id].reset_temp();
    }

    return is_valid;
}

CPU GPU void inline state::broadcast_channel(const int channel, const node* source)
{
    if(!IS_BROADCASTER(channel)) return;

    for (int i = 0; i < this->models.size; ++i)
    {
        const node* current = this->models.store[i];
        if(current->id == source->id) continue;
        if(current->is_goal) continue;
        if(!constraint::evaluate_constraint_set(current->invariants, this)) continue;
        
        const unsigned offset = curand(this->random) % current->edges.size;
        for (int j = 0; j < current->edges.size; ++j)
        {
            const edge* current_e = &current->edges.store[(i + offset) % current->edges.size];
            if(!IS_LISTENER(current_e->channel)) continue;
            if(!CAN_SYNC(channel, current_e->channel)) continue;

            node* dest = current_e->dest;
            
            this->models.store[i] = dest;

            current_e->apply_updates(this);
            break;
        }
    }
}
