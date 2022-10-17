#include "update_expression.h"
#include <string>

update_expression::update_expression(const expression_type type, const double value, const unsigned variable_id,
                                             update_expression* left, update_expression* right, update_expression* condition)
{
    this->type_        = type;
    this->left_        = left;
    this->condition_   = condition;
    this->right_       = right;
    this->value_       = value;
    this->variable_id_ = variable_id;
}

void update_expression::evaluate(simulator_state* state) const
{
    //! The stack has values in reverse order. 
    double v1, v2;
    switch(this->type_)
    {
    case literal_e:
        state->value_stack.push(this->value_);
        break;
    case clock_variable_e:
        state->value_stack.push(static_cast<int>(state->timers.at(static_cast<int>(this->variable_id_))->get_time()));
        break;
    case system_variable_e:
        state->value_stack.push(state->variables.at(static_cast<int>(this->variable_id_))->get_value());
        break;
    case plus_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate plus expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 + v2 );
        break;
    case minus_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate minus expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 - v2 );
        break;
    case multiply_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate multiply expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 * v2 );
        break;
    case division_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate division expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        if(v2 == 0) printf("Division by zero");  // NOLINT(clang-diagnostic-float-equal)
        state->value_stack.push( v1 / v2 );
        break;
    case power_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate power expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( pow(v1, v2) );
        break;
    case sqrt_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate sqrt expression\n");
        v1 = state->value_stack.pop();
        state->value_stack.push( sqrt(v1) );
        break;
    case negation_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate negation expression\n");
        v1 = state->value_stack.pop();
        state->value_stack.push( -v1 );
        break;
    case conditional_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate negation expression\n");
        v2 = state->value_stack.pop();
        state->value_stack.pop();
        state->value_stack.push(v2);
        break;
    case less_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate less_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 <= v2 );
        break;
    case greater_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate greater_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 >= v2 );
        break;
    case less_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate less expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 < v2 );
        break;
    case greater_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate greater expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 > v2 );
        break;
    case equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 == v2 );  // NOLINT(clang-diagnostic-float-equal)
        break;
    case not_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate not_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 != v2 );  // NOLINT(clang-diagnostic-float-equal)
        break;
    case not_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate not expression\n");
        v1 = state->value_stack.pop();
        state->value_stack.push( v1 == 0 ? 1.0 : 0.0   );  // NOLINT(clang-diagnostic-float-equal)
        break;
    // default:
    //     printf("EXPRESSION EVALUATION REACHED UNEXPECTED DEFAULT CASE. Returning 0");
    //     stack->push(0);
    //     break;
    }
}

std::string update_expression::type_to_string() const
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
    case power_e: break;
    case negation_e: break;
    case sqrt_e: break;
    case less_equal_e: break;
    case greater_equal_e: break;
    case less_e: break;
    case greater_e: break;
    case equal_e: break;
    case not_equal_e: break;
    case not_e: break;
    case conditional_e: break;
    default:
        result = "Not implemented yet";
    }
    return result;
}

std::string update_expression::to_string()
{
    std::string str = "Hello, i broke update_expression::to_string :=). Plz fix\n"; //TODO do this :)
    return str;
    // std::string left, right;
    //
    // if (this->get_left() != nullptr) left = this->get_left()->type_to_string();
    // else left = "nullptr";
    // if (this->get_right() != nullptr) right = this->get_right()->type_to_string();
    // else right = "nullptr";
    //
    // return "    Type: " + this->type_to_string() + " | value: " + std::to_string(this->get_value()) + " | left: " + left + " | right: " + right + "\n";
}

int update_expression::get_value()
{
    return this->value_;
}

GPU CPU bool update_expression::is_leaf() const
{
    return this->left_ == nullptr && this->right_ == nullptr && this->condition_ == nullptr;
}

GPU CPU update_expression* update_expression::get_left() const
{
    //The left node is dependent on the type. The condition is the switch
    return this->type_ == conditional_e ? this->condition_ : this->left_;
}


GPU CPU update_expression* update_expression::get_right(const cuda_stack<double>* value_stack) const
{
    if(this->type_ == conditional_e)
    {
        const double con = value_stack->peak();
        //if true, check right, else check middle
        return con != 0 ? this->left_ : this->right_; // NOLINT(clang-diagnostic-float-equal)
    }
    return this->right_;
}

void update_expression::accept(visitor* v) const
{
    if (this->left_ != nullptr) v->visit(this->left_);
    if (this->right_ != nullptr) v->visit(this->right_);
}

unsigned update_expression::get_depth() const
{
    const unsigned conditional = this->condition_ != nullptr ? this->condition_->get_depth() : 0;
    const unsigned left = this->left_ != nullptr ? this->left_->get_depth() : 0;
    const unsigned right = this->right_ != nullptr ? this->right_->get_depth() : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}

void update_expression::cuda_allocate(update_expression* cuda_p, const allocation_helper* helper) const
{
    update_expression* left_cuda = nullptr;
    if(this->left_ != nullptr)
    {
        cudaMalloc(&left_cuda, sizeof(update_expression));
        helper->free_list->push_back(left_cuda);
        this->left_->cuda_allocate(left_cuda, helper);
    }

    update_expression* right_cuda = nullptr;
    if(this->right_ != nullptr)
    {
        cudaMalloc(&right_cuda, sizeof(update_expression));
        helper->free_list->push_back(right_cuda);
        this->right_->cuda_allocate(right_cuda, helper);
    }

    update_expression* condition_cuda = nullptr;
    if(this->condition_ != nullptr)
    {
        cudaMalloc(&condition_cuda, sizeof(update_expression));
        helper->free_list->push_back(condition_cuda);
        this->condition_->cuda_allocate(condition_cuda, helper);
    }

    const update_expression copy = update_expression(this->type_,
        this->value_, this->variable_id_, left_cuda, right_cuda, condition_cuda);
    cudaMemcpy(cuda_p, &copy, sizeof(update_expression), cudaMemcpyHostToDevice);
}


//FACTORY CONSTRUCTORS
update_expression* update_expression::literal_expression(const double value)
{
    return new update_expression(literal_e,  value, NO_V_ID, nullptr, nullptr, nullptr);
}

update_expression* update_expression::clock_expression(const unsigned clock_id)
{
    return new update_expression(clock_variable_e,  NO_VALUE, clock_id, nullptr, nullptr, nullptr);
}

update_expression* update_expression::variable_expression(const unsigned variable_id)
{
    return new update_expression(system_variable_e, NO_VALUE, variable_id, NO_VALUE, nullptr, nullptr);
}

update_expression* update_expression::plus_expression(update_expression* left, update_expression* right)
{
    return new update_expression(plus_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::minus_expression(update_expression* left, update_expression* right)
{
    return new update_expression(minus_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::multiply_expression(update_expression* left, update_expression* right)
{
    return new update_expression(multiply_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::division_expression(update_expression* left, update_expression* right)
{
    return new update_expression(division_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::power_expression(update_expression* left, update_expression* right)
{
    return new update_expression(power_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::negate_expression(update_expression* expression)
{
    return new update_expression(negation_e, NO_VALUE, NO_V_ID, expression, nullptr, nullptr);
}

update_expression* update_expression::sqrt_expression(update_expression* expression)
{
    return new update_expression(sqrt_e, NO_VALUE, NO_V_ID, expression, nullptr, nullptr);
}

update_expression* update_expression::less_equal_expression(update_expression* left, update_expression* right)
{
    return new update_expression(less_equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::less_expression(update_expression* left, update_expression* right)
{
    return new update_expression(less_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::greater_equal_expression(update_expression* left, update_expression* right)
{
    return new update_expression(greater_equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::greater_expression(update_expression* left, update_expression* right)
{
    return new update_expression(greater_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::equal_expression(update_expression* left, update_expression* right)
{
    return new update_expression(equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::not_equal_expression(update_expression* left, update_expression* right)
{
    return new update_expression(equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

update_expression* update_expression::not_expression(update_expression* expression)
{
    return new update_expression(equal_e, NO_VALUE, NO_V_ID, expression, nullptr, nullptr);
}

update_expression* update_expression::conditional_expression(update_expression* condition, update_expression* left,
    update_expression* right)
{
    return new update_expression(conditional_e, NO_VALUE, NO_V_ID, left, right, condition);
}
