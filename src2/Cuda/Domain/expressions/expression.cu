#include "expression.h"
#include "../simulator_state.h"

expression::expression(const expression_type type, const double value, const unsigned variable_id,
                                             expression* left, expression* right, expression* condition)
{
    this->type_        = type;
    this->left_        = left;
    this->condition_   = condition;
    this->right_       = right;
    this->value_       = value;
    this->variable_id_ = variable_id;
}

void expression::evaluate(simulator_state* state) const
{
    //! The stack has values in reverse order. 
    double v1, v2;
    switch(this->type_)
    {
    case literal_e:
        state->value_stack.push(this->value_);
        break;
    case clock_variable_e:
        state->value_stack.push(state->timers.at(static_cast<int>(this->variable_id_))->get_time());
        break;
    case system_variable_e:
        state->value_stack.push(state->variables.at(static_cast<int>(this->variable_id_))->get_time());
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

std::string expression::type_to_string() const
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
    case power_e:
        result = "^";
        break;
    case negation_e:
        result = "~";
        break;
    case sqrt_e:
        result = "sqrt";
        break;
    case less_equal_e:
        result = "<=";
        break;
    case greater_equal_e:
        result = ">=";
        break;
    case less_e:
        result = "<";
        break;
    case greater_e:
        result = ">";
        break;
    case equal_e:
        result = "==";
        break;
    case not_equal_e:
        result = "!=";
        break;
    case not_e:
        result = "!";
        break;
    case conditional_e:
        result = "if";
        break;
    default:
        result = "Not implemented yet";
    }
    return result;
}

std::string expression::to_string() const
{
    std::string left, right, con;
    if (this->condition_ != nullptr) con = this->condition_->type_to_string();
    else con = "nullptr";
    if (this->left_ != nullptr) left = this->left_->type_to_string();
    else left = "nullptr";
    if (this->right_ != nullptr) right = this->right_->type_to_string();
    else right = "nullptr";
    if (this->type_ == expression_type::literal_e)
    {
        return "Type: " + this->type_to_string() + " | Value: " + std::to_string(this->value_) + " | Condition: " + con + " | Left: " + left + " | Right: " + right + "\n";
    }
    
    return "Type: " + this->type_to_string() + " | Condition: " + con + " | Left: " + left + " | Right: " + right + "\n";
    
}

GPU CPU bool expression::is_leaf() const
{
    return this->left_ == nullptr && this->right_ == nullptr && this->condition_ == nullptr;
}

GPU CPU expression* expression::get_left() const
{
    //The left node is dependent on the type. The condition is the switch
    return this->type_ == conditional_e ? this->condition_ : this->left_;
}


GPU CPU expression* expression::get_right(const cuda_stack<double>* value_stack) const
{
    if(this->type_ == conditional_e)
    {
        const double con = value_stack->peak();
        //if true, check right, else check middle
        return con != 0 ? this->left_ : this->right_; // NOLINT(clang-diagnostic-float-equal)
    }
    return this->right_;
}

void expression::accept(visitor* v) const
{
    if (this->condition_ != nullptr) v->visit(this->condition_);
    if (this->left_ != nullptr) v->visit(this->left_);
    if (this->right_ != nullptr) v->visit(this->right_);
}

unsigned expression::get_depth() const
{
    const unsigned conditional = this->condition_ != nullptr ? this->condition_->get_depth() : 0;
    const unsigned left = this->left_ != nullptr ? this->left_->get_depth() : 0;
    const unsigned right = this->right_ != nullptr ? this->right_->get_depth() : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}

bool expression::contains_clock_expression() const
{
    if(this->type_ == clock_variable_e) return true;
    
    const bool con = this->condition_ != nullptr && this->condition_->contains_clock_expression();
    const bool left = this->left_ != nullptr && this->left_->contains_clock_expression();
    const bool right = this->right_ != nullptr && this->right_->contains_clock_expression();

    return con || left || right;
}

void expression::cuda_allocate(expression* cuda_p, const allocation_helper* helper) const
{
    expression* left_cuda = nullptr;
    if(this->left_ != nullptr)
    {
        cudaMalloc(&left_cuda, sizeof(expression));
        helper->free_list->push_back(left_cuda);
        this->left_->cuda_allocate(left_cuda, helper);
    }

    expression* right_cuda = nullptr;
    if(this->right_ != nullptr)
    {
        cudaMalloc(&right_cuda, sizeof(expression));
        helper->free_list->push_back(right_cuda);
        this->right_->cuda_allocate(right_cuda, helper);
    }

    expression* condition_cuda = nullptr;
    if(this->condition_ != nullptr)
    {
        cudaMalloc(&condition_cuda, sizeof(expression));
        helper->free_list->push_back(condition_cuda);
        this->condition_->cuda_allocate(condition_cuda, helper);
    }

    const expression copy = expression(this->type_,
        this->value_, this->variable_id_, left_cuda, right_cuda, condition_cuda);
    cudaMemcpy(cuda_p, &copy, sizeof(expression), cudaMemcpyHostToDevice);
}


//FACTORY CONSTRUCTORS
expression* expression::literal_expression(const double value)
{
    return new expression(literal_e,  value, NO_V_ID, nullptr, nullptr, nullptr);
}

expression* expression::clock_expression(const unsigned clock_id)
{
    return new expression(clock_variable_e,  NO_VALUE, clock_id, nullptr, nullptr, nullptr);
}

expression* expression::variable_expression(const unsigned variable_id)
{
    return new expression(system_variable_e, NO_VALUE, variable_id, NO_VALUE, nullptr, nullptr);
}

expression* expression::plus_expression(expression* left, expression* right)
{
    return new expression(plus_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::minus_expression(expression* left, expression* right)
{
    return new expression(minus_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::multiply_expression(expression* left, expression* right)
{
    return new expression(multiply_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::division_expression(expression* left, expression* right)
{
    return new expression(division_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::power_expression(expression* left, expression* right)
{
    return new expression(power_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::negate_expression(expression* expr)
{
    return new expression(negation_e, NO_VALUE, NO_V_ID, expr, nullptr, nullptr);
}

expression* expression::sqrt_expression(expression* expr)
{
    return new expression(sqrt_e, NO_VALUE, NO_V_ID, expr, nullptr, nullptr);
}

expression* expression::less_equal_expression(expression* left, expression* right)
{
    return new expression(less_equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::less_expression(expression* left, expression* right)
{
    return new expression(less_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::greater_equal_expression(expression* left, expression* right)
{
    return new expression(greater_equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::greater_expression(expression* left, expression* right)
{
    return new expression(greater_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::equal_expression(expression* left, expression* right)
{
    return new expression(equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::not_equal_expression(expression* left, expression* right)
{
    return new expression(equal_e, NO_VALUE, NO_V_ID, left, right, nullptr);
}

expression* expression::not_expression(expression* expr)
{
    return new expression(equal_e, NO_VALUE, NO_V_ID, expr, nullptr, nullptr);
}

expression* expression::conditional_expression(expression* condition, expression* left,
    expression* right)
{
    return new expression(conditional_e, NO_VALUE, NO_V_ID, left, right, condition);
}
