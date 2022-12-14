#include "constraint_t.h"
#include "../simulator_state.h"


void throw_on_timer_expression(const constraint_value left, const constraint_value right)
{
    const bool left_clock_expr = !left.is_clock && left.expr->contains_clock_expression();
    const bool right_clock_expr = !right.is_clock && right.expr->contains_clock_expression();
    
    if(left_clock_expr || right_clock_expr)
        throw std::runtime_error(std::string(
            "A constraint was instantiated using a clock expression. This is not allowed"));
}

GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}

GPU CPU double get_constraint_value(const constraint_value* con, simulator_state* state)
{
    if(con->is_clock)
    {
        return state->get_timer(con->clock_id)->get_temp_time();
    }
    return con->expr->evaluate(state);
}



constraint_t::constraint_t(const logical_operator_t type, const constraint_value left, const constraint_value right,
    const bool validate)
{
    this->type_ = type;
    this->left_ = left;
    this ->right_ = right;
    if(validate)
    {
        throw_on_timer_expression(left, right);
    }
}

bool constraint_t::evaluate(simulator_state* state) const
{
    const double v1 = get_constraint_value(&this->left_, state);
    const double v2 = get_constraint_value(&this->right_, state);
    
    switch(this->type_)
    {
        case logical_operator_t::less_equal_t: return v1 <= v2;
        case logical_operator_t::greater_equal_t: return v1 >= v2;
        case logical_operator_t::less_t: return v1 < v2;
        case logical_operator_t::greater_t: return v1 > v2;
        case logical_operator_t::equal_t: return cuda_abs(v1 - v2) < 0.005; //v1 == v2;
        case logical_operator_t::not_equal_t: return cuda_abs(v1 - v2) >= 0.005; //v1 != v2;
    }
    return false;
}

CPU GPU bool constraint_t::check_max_time_progression(simulator_state* state, double* out_max_progression) const
{
    (*out_max_progression) = 0.0;
    if(this->right_.is_clock && this->left_.is_clock || !this->right_.is_clock && !this->left_.is_clock)
        return false;
    
    if(this->left_.is_clock &&
        (this->type_ == logical_operator_t::less_t || this->type_ == logical_operator_t::less_equal_t))
    {
        const double time = state->get_timer(this->left_.clock_id)->get_time();
        const double value = this->right_.expr->evaluate(state);

        const double diff = value - time;
        (*out_max_progression) = diff; //TODO rethink this. What to do if a diff is negative.
        return true;
    }
    
    if(this->right_.is_clock &&
        (this->type_ == logical_operator_t::greater_t || this->type_ == logical_operator_t::greater_equal_t))
    {
        const double time = state->get_timer(this->right_.clock_id)->get_time();
        const double value = this->left_.expr->evaluate(state);

        const double diff = value - time;
        (*out_max_progression) = diff; //TODO rethink this. What to do if a diff is negative.
        return true;
    }
    
    return false;
}

void constraint_t::accept(visitor* v)
{
    if (!this->left_.is_clock) v->visit(this->left_.expr);
    if (!this->right_.is_clock) v->visit(this->right_.expr);
}

void constraint_t::pretty_print(std::ostream& os) const
{
    std::string left, right;
    
    if (this->left_.is_clock) left = "(Clock " + std::to_string(this->left_.clock_id);
    else left = "(" + this->left_.expr->to_string();
    if (this->right_.is_clock) right = "Clock " + std::to_string(this->right_.clock_id) + ")";
    else right = this->right_.expr->to_string() + ")";

    os << left + " " + constraint_t::logical_operator_to_string(this->type_) + " " + right + "\n";
    
    // printf("%s %s %s\n", left.c_str(), constraint_t::logical_operator_to_string(this->type_).c_str(),
    // right.c_str());
}

void constraint_t::cuda_allocate(constraint_t* pointer, allocation_helper* helper) const
{
    constraint_value left;
    if(!this->left_.is_clock)
    {
        expression* left_expr = nullptr;
        helper->allocate(&left_expr, sizeof(expression));
        this->left_.expr->cuda_allocate(left_expr, helper);
        left = constraint_value::from_expression(left_expr);
    }
    else left = constraint_value::from_timer(this->left_.clock_id);
    
    constraint_value right;
    if(!this->right_.is_clock)
    {
        expression* right_expr = nullptr;
        helper->allocate(&right_expr, sizeof(expression));
        this->right_.expr->cuda_allocate(right_expr, helper);
        right = constraint_value::from_expression(right_expr);
    }
    else right = constraint_value::from_timer(this->right_.clock_id);

    const constraint_t con = constraint_t(this->type_, left, right, false);
    cudaMemcpy(pointer, &con, sizeof(constraint_t), cudaMemcpyHostToDevice);
}

std::string constraint_t::logical_operator_to_string(const logical_operator_t op)
{
    switch (op)
    {
    case logical_operator_t::less_equal_t:
        return "<=";
    case logical_operator_t::greater_equal_t:
        return ">=";
    case logical_operator_t::less_t:
        return "<";
    case logical_operator_t::greater_t:
        return ">";
    case logical_operator_t::equal_t:
        return "==";
    case logical_operator_t::not_equal_t:
        return "!=";
    default:
        return "not a boolean operator";
    }
}





//! LESS THAN OR EQUAL
constraint_t* constraint_t::less_equal_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::less_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}

constraint_t* constraint_t::less_equal_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::less_equal_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::less_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::less_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}

//! GREATER THAN OR EQUAL
constraint_t* constraint_t::greater_equal_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::greater_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}


constraint_t* constraint_t::greater_equal_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::greater_equal_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::greater_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::greater_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}

//! LESS THAN
constraint_t* constraint_t::less_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::less_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}

constraint_t* constraint_t::less_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::less_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::less_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::less_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}

//! GREATER THAN
constraint_t* constraint_t::greater_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::greater_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}

constraint_t* constraint_t::greater_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::greater_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::greater_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::greater_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}

//! equal
constraint_t* constraint_t::equal_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::equal_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}

constraint_t* constraint_t::equal_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::equal_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::equal_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}

//! NOT EQUAL
constraint_t* constraint_t::not_equal_v(const int timer_id, expression* value_expr)
{
    return new constraint_t{logical_operator_t::not_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_expression(value_expr)};
}

constraint_t* constraint_t::not_equal_e(expression* value_expr1, expression* value_expr2)
{
    return new constraint_t{logical_operator_t::not_equal_t, constraint_value::from_expression(value_expr1), constraint_value::from_expression(value_expr2)};
}

constraint_t* constraint_t::not_equal_t(const int timer_id, const int timer_id2)
{
    return new constraint_t{logical_operator_t::not_equal_t, constraint_value::from_timer(timer_id), constraint_value::from_timer(timer_id2)};
}
