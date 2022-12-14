#include "variable_expression_evaluator.h"

#include "helper_methods.h"

// Parser constructor.
variable_expression_evaluator::variable_expression_evaluator(unordered_map<string,int>* local_vars, unordered_map<string,int>* global_vars)
{
    exp_ptr_ = nullptr;
    local_vars_ = local_vars;
    global_vars_ = global_vars;
}

// Parser entry point.
expression* variable_expression_evaluator::eval_exp(char *exp)
{
    
    exp_ptr_ = exp;
    get_token();
    
    if (!*token_) 
    {
        THROW_LINE("No Expression Present")
    }
    
    expression* result = eval_exp1();
    if (*token_) // last token must be null
    {
        THROW_LINE("Syntax Error")
    }
    return result;
}
// Process an assignment.
expression* variable_expression_evaluator::eval_exp1()
{
    int slot;
    expression* result;
    if (tok_type_ == UPDATEVARIABLE) 
    {
        char temp_token[80];
        // save old token
        char *t_ptr = exp_ptr_;
        strcpy(temp_token, token_);
        // compute the index of the variable
        slot = *token_ - 'A';
        get_token();
        if (*token_ != '=') 
        {
            exp_ptr_ = t_ptr; // return current token
            strcpy(token_, temp_token); // restore old token
            tok_type_ = UPDATEVARIABLE;
        }
        else {
            get_token(); // get next part of exp
            
            result = eval_exp2();
            //vars[slot] = result;
            return result;
        }
    }
    result = eval_exp2();
    return result;
}
// Add or subtract two terms.
 expression* variable_expression_evaluator::eval_exp2()
{
    char op;
    expression* result = eval_exp3();
    
    while ((op = *token_) == '+' || op == '-')
    {
        get_token();
        expression* temp = eval_exp3();
        switch (op) 
        {
        case '-':
            result = expression::minus_expression(result, temp);
            break;
        case '+':
            result = expression::plus_expression(result, temp);
        }
    }

    return  result;
}
// Multiply or divide two factors.
expression* variable_expression_evaluator::eval_exp3()
{
    char op;
    double temp;
    expression* result = eval_exp4();
    while ((op = *token_) == '*' || op == '/')
    {
        get_token();
        expression* temp = eval_exp4();
        switch (op) 
        {
        case '*':
            result = expression::multiply_expression(result, temp);
            break;
        case '/':  // NOLINT(clang-diagnostic-implicit-fallthrough)
            result = expression::division_expression(result, temp);
        }
    }

    return result;
}
// Process an exponent.
expression* variable_expression_evaluator::eval_exp4()
{
    expression* result = eval_exp5();
    while (*token_ == '^')
    {
        get_token();
        expression* temp = eval_exp5();
        result = expression::power_expression(result, temp);
    }
    return result;
}
// Evaluate a unary + or -.
expression* variable_expression_evaluator::eval_exp5()
{
    char op = 0;
    if ((tok_type_ == UPDATEDELIMITER) && *token_ == '+' || *token_ == '-')
    {
        op = *token_;
        get_token();
    }
    expression* result = eval_exp6();
    
    if (op == '-')
    {
        result = expression::negate_expression(result);
    }
    
    return result;
}
// Process a function, a parenthesized expression, a value or a variable
expression* variable_expression_evaluator::eval_exp6()
{
    const bool is_func = (tok_type_ == UPDATEFUNCTION);
    char temp_token[80];
    if (is_func)
    {
        strcpy(temp_token, token_);
        get_token();
    }
    if (*token_ == '(') 
    {
        get_token();
        auto result = eval_exp2();
        if (*token_ != ')')
        {
            THROW_LINE("Unbalanced Parentheses");
        }
        if (is_func)
        {
            if (!strcmp(temp_token, "random"))
                result = expression::random_expression(result);
            else if (!strcmp(temp_token, "sqrt"))
                result = expression::sqrt_expression(result);
            else
            {
                THROW_LINE("Unknown Function");
            }
        }
        get_token();
        return result;
    }
    else
        switch (tok_type_)
        {
            case UPDATEVARIABLE:
                {
                    const string var = token_;
                    expression* result;

                    if (local_vars_->count(var))
                        result = expression::variable_expression(local_vars_->at(var));
                    else if (global_vars_->count(var))
                        result = expression::variable_expression(global_vars_->at(var));
                    else
                    {
                        THROW_LINE("VAR NOT DECLARED")
                    }
                    
                    get_token();
                    return result;
                }
                //result = vars[*token - 'A'];
                
            case UPDATENUMBER:
                {
                    expression* result =  expression::literal_expression(stod(token_));
                    get_token();
                    return result;
                }
            default:
                {
                    THROW_LINE("Syntax Error")
                }
        }
}
// Obtain the next token.
void variable_expression_evaluator::get_token()
{
    tok_type_ = 0;
    char* temp = token_;
    *temp = '\0';
    if (!*exp_ptr_)  // at end of expression
        return;
    while (isspace(*exp_ptr_))  // skip over white space
        ++exp_ptr_; 
    if (strchr("+-*/%^=()", *exp_ptr_)) 
    {
        tok_type_ = UPDATEDELIMITER;
        *temp++ = *exp_ptr_++;  // advance to next char
    }
    else if (isalpha(*exp_ptr_)) 
    {
        while (!strchr(" +-/*%^=()\t\r", *exp_ptr_) && (*exp_ptr_))
            *temp++ = *exp_ptr_++;
        while (isspace(*exp_ptr_))  // skip over white space
            ++exp_ptr_;
        tok_type_ = (*exp_ptr_ == '(') ? UPDATEFUNCTION : UPDATEVARIABLE;
    }
    else if (isdigit(*exp_ptr_) || *exp_ptr_ == '.')
    {
        while (!strchr(" +-/*%^=()\t\r", *exp_ptr_) && (*exp_ptr_))
            *temp++ = *exp_ptr_++;
        tok_type_ = UPDATENUMBER;
    }
    *temp = '\0';
}

expression* variable_expression_evaluator::evaluate_variable_expression(const string& input, unordered_map<string, int>* local_vars, unordered_map<string, int>* global_vars)
{
    variable_expression_evaluator ob(local_vars, global_vars);
    expression* ans = ob.eval_exp(const_cast<char*>(input.substr(0, input.length()).c_str()));
    return ans;
}