#include "update_expression.h"

#include <string>

update_expression::update_expression(const expression_type type, update_expression* left,
                                     update_expression* right, const int value, const unsigned variable_id)
{
    this->type_ = type;
    this->left_ = left;
    this->right_= right;
    this->value_ = value;
    this->variable_id_ = variable_id;
}

void update_expression::evaluate(cuda_stack<int>* stack, const lend_array<clock_timer_t>* timers,
                                   const lend_array<system_variable>* variables) const
{
    int v1, v2;
    switch(this->type_)
    {
    case literal_e:
        stack->push(this->value_);
        break;
    case clock_variable_e:
        stack->push(static_cast<int>(timers->at(static_cast<int>(this->variable_id_))->get_time()));
        break;
    case system_variable_e:
        stack->push(variables->at(static_cast<int>(this->variable_id_))->get_value());
        break;
    case plus_e:
        if(stack->count() < 2) printf("stack not big enough to evaluate plus expression");
        v1 = stack->pop();
        v2 = stack->pop();
        stack->push( v1 + v2 );
        break;
    case minus_e:
        if(stack->count() < 2) printf("stack not big enough to evaluate minus expression");
        v1 = stack->pop();
        v2 = stack->pop();
        stack->push( v1 - v2 );
        break;
    case multiply_e:
        if(stack->count() < 2) printf("stack not big enough to evaluate multiply expression");
        v1 = stack->pop();
        v2 = stack->pop();
        stack->push( v1 * v2 );
        break;
    case division_e:
        if(stack->count() < 2) printf("stack not big enough to evaluate division expression");
        v1 = stack->pop();
        v2 = stack->pop();
        if(v2 == 0) printf("Division by zero");
        stack->push( v1 / v2 );
        break;
    }
}

std::string update_expression::type_to_string()
{
    std::string result;
    switch (this->type_)
    {
    case literal_e:
        result = "literal";
        break;
    case clock_variable_e:
        result = "clock variable";
        break;
    case system_variable_e:
        result = "system variable";
        break;
    case plus_e:
        result = "+";
        break;
    case minus_e:
        result = "-";
        break;
    case multiply_e:
        result = "*";
        break;
    case division_e:
        result = "/";
        break;
    default:
        result = "Not implemented yet";
    }
    return result;
}

std::string update_expression::to_string()
{
    std::string left, right;
    
    if (this->get_left() != nullptr) left = this->get_left()->type_to_string();
    else left = "nullptr";
    if (this->get_right() != nullptr) right = this->get_right()->type_to_string();
    else right = "nullptr";

    return "    Type: " + this->type_to_string() + " | value: " + std::to_string(this->get_value()) + " | left: " + left + " | right: " + right + "\n";
}

int update_expression::get_value()
{
    return this->value_;
}

GPU CPU update_expression* update_expression::get_left() const
{
    return this->left_;
}

GPU CPU update_expression* update_expression::get_right() const
{
    return this->right_;
}

void update_expression::accept(visitor* v) const
{
    if (this->left_ != nullptr) v->visit(this->left_);
    if (this->right_ != nullptr) v->visit(this->right_);
}

unsigned update_expression::get_depth() const
{
    const unsigned left = this->left_ != nullptr ? this->left_->get_depth() : 0;
    const unsigned right = this->right_ != nullptr ? this->right_->get_depth() : 0;

    return (left > right ? left : right)+1;
}

void update_expression::cuda_allocate(update_expression* cuda_p, const allocation_helper* helper) const
{
    update_expression* left_cuda = nullptr;
    if(this->left_)
    {
        cudaMalloc(&left_cuda, sizeof(update_expression));
        helper->free_list->push_back(left_cuda);
        this->left_->cuda_allocate(left_cuda, helper);
    }

    update_expression* right_cuda = nullptr;
    if(this->left_)
    {
        cudaMalloc(&right_cuda, sizeof(update_expression));
        helper->free_list->push_back(right_cuda);
        this->right_->cuda_allocate(right_cuda, helper);
    }

    const update_expression copy = update_expression(
        this->type_, left_cuda, right_cuda, this->value_, this->variable_id_);
    cudaMemcpy(cuda_p, &copy, sizeof(update_expression), cudaMemcpyHostToDevice);
}


//FACTORY CONSTRUCTORS
update_expression* update_expression::literal_expression(const int value)
{
    return new update_expression(literal_e, nullptr, nullptr, value, NO_V_ID);
}

update_expression* update_expression::clock_expression(const unsigned clock_id)
{
    return new update_expression(clock_variable_e, nullptr, nullptr, NO_VALUE, clock_id);
}

update_expression* update_expression::variable_expression(unsigned variable)
{
    return new update_expression(system_variable_e, nullptr, nullptr, NO_VALUE, variable);
}

update_expression* update_expression::plus_expression(update_expression* left, update_expression* right)
{
    return new update_expression(plus_e, left, right, NO_VALUE, NO_V_ID);
}

update_expression* update_expression::minus_expression(update_expression* left, update_expression* right)
{
    return new update_expression(minus_e, left, right, NO_VALUE, NO_V_ID);
}

update_expression* update_expression::multiply_expression(update_expression* left, update_expression* right)
{
    return new update_expression(multiply_e, left, right, NO_VALUE, NO_V_ID);
}

update_expression* update_expression::division_expression(update_expression* left, update_expression* right)
{
    return new update_expression(division_e, left, right, NO_VALUE, NO_V_ID);
}
