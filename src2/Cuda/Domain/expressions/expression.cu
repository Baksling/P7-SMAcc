#include "expression.h"
#include "../simulator_state.h"

expression::expression(const expression_type type, const double value, const unsigned variable_id,
                                             expression* left, expression* right, expression* condition)
{
    this->type_        = type;
    this->left_        = left;
    this->right_       = right;
    
    if(this->type_ == literal_e)
    {
        this->value = value;
    }
    else if(this->type_ == clock_variable_e || this->type_ == system_variable_e)
    {
        this->variable_id = variable_id;
    }
    else if(this->type_ == conditional_e)
    {
        this->condition = condition;
    }
    else
    {
        //done to instantiate memory to known value.
        this->value = NO_VALUE;
    }
}

double expression::evaluate_current(simulator_state* state) const
{
    //! The stack has values in reverse order. 
    double v1, v2;
    switch(this->type_)
    {
    case literal_e:
        return this->value;
    case clock_variable_e:
        return state->get_timers().at(static_cast<int>(this->variable_id))->get_temp_time();
    case system_variable_e:
        return state->get_variables().at(static_cast<int>(this->variable_id))->get_temp_time();
    case random_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate random expression\n");
        v1 = state->value_stack.pop();
        return curand_uniform_double(state->random) * v1; //v1 repressents max value 
    case plus_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate plus expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 + v2;
    case minus_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate minus expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 - v2;
    case multiply_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate multiply expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 * v2;
    case division_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate division expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        if(v2 == 0.0) // NOLINT(clang-diagnostic-float-equal)
        {
            printf("Division by zero"); return v2;
        }
        return v1 / v2;
    case power_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate power expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return pow(v1, v2);
    case sqrt_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate sqrt expression\n");
        v1 = state->value_stack.pop();
        if(0 > v1)
        { printf("sqrt of negative numbers"); return 0.0; }
        return sqrt(v1);
    case negation_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate negation expression\n");
        v1 = state->value_stack.pop();
        return -v1;
    case conditional_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate conditional expression\n");
        v2 = state->value_stack.pop();
        state->value_stack.pop();
        return v2;
    case less_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate less_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 <= v2;
    case greater_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate greater_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 >= v2;
    case less_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate less expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 < v2;
    case greater_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate greater expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 > v2;
    case equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 == v2; // NOLINT(clang-diagnostic-float-equal)
    case not_equal_e:
        if(state->value_stack.count() < 2) printf("stack not big enough to evaluate not_equal expression\n");
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 != v2; // NOLINT(clang-diagnostic-float-equal)
    case not_e:
        if(state->value_stack.count() < 1) printf("stack not big enough to evaluate not expression\n");
        v1 = state->value_stack.pop();
        return v1 == 0 ? 1.0 : 0.0;
    }

    printf("evaluation of expression not matching any known type\n");
    return 0.0;
}

double expression::evaluate(simulator_state* state)
{
    state->value_stack.clear();
    if(this->is_leaf())
    {
        return this->evaluate_current(state);
    }
    state->expression_stack.clear();

    expression* current = this;
    while (true)
    {
        while(current != nullptr)
        {
            state->expression_stack.push(current);
            
            if(!current->is_leaf()) //only push twice if it has children
                state->expression_stack.push(current);
            
            current = current->get_left();
        }
        if(state->expression_stack.is_empty())
        {
            break;
        }
        current = state->expression_stack.pop();
        
        if(!state->expression_stack.is_empty() && state->expression_stack.peak() == current)
        {
            current = current->get_right(&state->value_stack);
        }
        else
        {
            const double val = current->evaluate_current(state);
            state->value_stack.push(val);
            current = nullptr;
        }
    }

    if(state->value_stack.is_empty())
    {
        printf("Expression evaluation ended in no values! PANIC!\n");
        return 0.0;
    }
    
    return state->value_stack.pop();
    
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
        result = "clock variable id: " + std::to_string(this->variable_id);
        break;
    case system_variable_e:
        result = "system variable id: " + std::to_string(this->variable_id);
        break;
    case random_e:
        result = "random(" + this->left_->type_to_string() + ")";
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

void expression::pretty_print(std::ostream& os) const
{
    os << this->to_string();
}

std::string expression::to_string() const
{
    std::string left, right;
    std::string temp = " ";
    if (this->left_ != nullptr) left = this->left_->to_string();
    else temp = "";
    if (this->right_ != nullptr) right = this->right_->to_string();
    else temp = "";
    
    return (this->type_ == expression_type::literal_e
        ? "(" + std::to_string(this->value)
        : "(" + left + temp + this->type_to_string() + temp + right) + ")";
}

GPU CPU bool expression::is_leaf() const
{
    return this->left_ == nullptr && this->right_ == nullptr && this->type_ != conditional_e;
}

GPU CPU expression* expression::get_left() const
{
    //The left node is dependent on the type. The condition is the switch
    return this->type_ == conditional_e ? this->condition : this->left_;
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
    if (this->type_ == conditional_e) v->visit(this->condition);
    if (this->left_ != nullptr) v->visit(this->left_);
    if (this->right_ != nullptr) v->visit(this->right_);
}

unsigned expression::get_depth() const
{
    const unsigned conditional = this->type_ == conditional_e ? this->condition->get_depth() : 0;
    const unsigned left = this->left_ != nullptr ? this->left_->get_depth() : 0;
    const unsigned right = this->right_ != nullptr ? this->right_->get_depth() : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}

bool expression::is_constant() const
{
    if(this->type_ == literal_e) return true;
    if(this->type_ == clock_variable_e || this->type_ == system_variable_e) return false;

    const bool left = this->left_ != nullptr ? this->left_->is_constant() : true;
    const bool right = this->right_ != nullptr ? this->right_->is_constant() : true;
    const bool cond = this->type_ == conditional_e ? this->condition->is_constant() : true;

    return left && right && cond;
}

bool expression::contains_clock_expression() const
{
    if(this->type_ == clock_variable_e) return true;
    
    const bool con = this->type_ == conditional_e && this->condition->contains_clock_expression();
    const bool left = this->left_ != nullptr && this->left_->contains_clock_expression();
    const bool right = this->right_ != nullptr && this->right_->contains_clock_expression();

    return con || left || right;
}

void expression::cuda_allocate(expression* cuda_p, allocation_helper* helper) const
{
    expression* left_cuda = nullptr;
    if(this->left_ != nullptr)
    {
        helper->allocate(&left_cuda, sizeof(expression));
        this->left_->cuda_allocate(left_cuda, helper);
    }

    expression* right_cuda = nullptr;
    if(this->right_ != nullptr)
    {
        helper->allocate(&right_cuda, sizeof(expression));
        this->right_->cuda_allocate(right_cuda, helper);
    }

    expression* condition_cuda = nullptr;
    if(this->type_ == conditional_e)
    {
        helper->allocate(&condition_cuda, sizeof(expression));
        this->condition->cuda_allocate(condition_cuda, helper);
    }

    const expression copy = expression(
        this->type_,
        this->value,
        this->variable_id,
        left_cuda,
        right_cuda,
        condition_cuda);
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

expression* expression::random_expression(expression* expr)
{
    return new expression(random_e, NO_VALUE, NO_V_ID, expr, nullptr, nullptr);
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
