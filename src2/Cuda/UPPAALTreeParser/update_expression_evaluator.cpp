#include "update_expression_evaluator.h"

#include "helper_methods.h"

// Parser constructor.
update_expression_evaluator::update_expression_evaluator(map<string,int>* local_vars, map<string,int>* global_vars)
{
    exp_ptr = NULL;
    local_vars_ = local_vars;
    global_vars_ = global_vars;
}

// Parser entry point.
expression* update_expression_evaluator::eval_exp(char *exp)
{
    cout << "\n ::::3::::";
    cout.flush();
    
    exp_ptr = exp;
    cout << "\n ::::4::::";
    cout.flush();
    get_token();
    cout << "\n ::::5::::";
    cout.flush();
    
    if (!*token) 
    {
        THROW_LINE("No Expression Present")
    }
    
    expression* result = eval_exp1();
    if (*token) // last token must be null
    {
        THROW_LINE("Syntax Error")
    }
    cout << "\n ::::12::::";
    cout.flush();
    cout << "\n ::::77::::"<<result->get_value();
    cout.flush();
    return result;
}
// Process an assignment.
expression* update_expression_evaluator::eval_exp1()
{
    int slot;
    char temp_token[80];
    expression* result;
    if (tok_type == UPDATEVARIABLE) 
    {
        // save old token
        char *t_ptr = exp_ptr;
        strcpy(temp_token, token);
        // compute the index of the variable
        slot = *token - 'A';
        get_token();
        if (*token != '=') 
        {
            exp_ptr = t_ptr; // return current token
            strcpy(token, temp_token); // restore old token
            tok_type = UPDATEVARIABLE;
        }
        else {
            get_token(); // get next part of exp
            
            result = eval_exp2();
            //vars[slot] = result;
            return result;
        }
    }
    result = eval_exp2();
    cout << "\n ::::66::::"<<result->get_value();
    cout.flush();
    return result;
}
// Add or subtract two terms.
 expression* update_expression_evaluator::eval_exp2()
{
    char op;
    expression* result = eval_exp3();
    cout << "\n ::::55::::"<<result->get_value();
    cout.flush();
    
    while ((op = *token) == '+' || op == '-')
    {
        get_token();
        expression* temp = eval_exp3();
        switch (op) 
        {
        case '-':
            result = expression::minus_expression(result, temp);
        case '+':
            result = expression::plus_expression(result, temp);
        }
    }

    return  result;
}
// Multiply or divide two factors.
expression* update_expression_evaluator::eval_exp3()
{
    char op;
    double temp;
    expression* result = eval_exp4();
    cout << "\n ::::44::::"<<result->get_value();
    cout.flush();
    while ((op = *token) == '*' || op == '/')
    {
        get_token();
        expression* temp = eval_exp4();
        switch (op) 
        {
        case '*':
            result = expression::multiply_expression(result, temp);
        case '/':
            result = expression::division_expression(result, temp);
        }
    }

    return result;
}
// Process an exponent.
expression* update_expression_evaluator::eval_exp4()
{
    double temp;
    expression* result = eval_exp5();
    cout << "\n ::::33::::"<<result->get_value();
    cout.flush();
    while (*token == '^')
    {
        // get_token();
        eval_exp5();
        // result = pow(result, temp); TODO Parse ^
    }
    return result;
}
// Evaluate a unary + or -.
expression* update_expression_evaluator::eval_exp5()
{
    char op;
    op = 0;
    if ((tok_type == UPDATEDELIMITER) && *token == '+' || *token == '-')
    {
        op = *token;
        get_token();
    }
    expression* result = eval_exp6();
    cout << "\n ::::22::::"<<result->get_value();
    cout.flush();
    return result;
    if (op == '-')
    {
        //TODO Parse NEgate
        
    }
}
// Process a function, a parenthesized expression, a value or a variable
expression* update_expression_evaluator::eval_exp6()
{
    bool isfunc = (tok_type == UPDATEFUNCTION);
    char temp_token[80];
    if (isfunc)
    {
        strcpy(temp_token, token);
        get_token();
    } 
    if (*token == '(') 
    {
        get_token();
        return eval_exp2();
        if (*token != ')')
            strcpy(errormsg, "Unbalanced Parentheses");
        if (isfunc)
        {
            // TODO Parse Funcs?
            // if (!strcmp(temp_token, "SIN"))
            //     result = sin(PI / 180 * result);
            // else if (!strcmp(temp_token, "COS"))
            //     result = cos(PI / 180 * result);
            // else if (!strcmp(temp_token, "TAN"))
            //     result = tan(PI / 180 * result);
            // else if (!strcmp(temp_token, "ASIN"))
            //     result = 180 / PI*asin(result);
            // else if (!strcmp(temp_token, "ACOS"))
            //     result = 180 / PI*acos(result);
            // else if (!strcmp(temp_token, "ATAN"))
            //     result = 180 / PI*atan(result);
            // else if (!strcmp(temp_token, "SINH"))
            //     result = sinh(result);
            // else if (!strcmp(temp_token, "COSH"))
            //     result = cosh(result);
            // else if (!strcmp(temp_token, "TANH"))
            //     result = tanh(result);
            // else if (!strcmp(temp_token, "ASINH"))
            //     result = asinh(result);
            // else if (!strcmp(temp_token, "ACOSH"))
            //     result = acosh(result);
            // else if (!strcmp(temp_token, "ATANH"))
            //     result = atanh(result);
            // else if (!strcmp(temp_token, "LN"))
            //     result = log(result);
            // else if (!strcmp(temp_token, "LOG"))
            //     result = log10(result);
            // else if (!strcmp(temp_token, "EXP"))
            //     result = exp(result);
            // else if (!strcmp(temp_token, "SQRT"))
            //     result = sqrt(result);
            // else if (!strcmp(temp_token, "SQR"))
            //     result = result*result;
            // else if (!strcmp(temp_token, "ROUND"))
            //     result = round(result);
            // else if (!strcmp(temp_token, "INT"))
            //     result = floor(result);
            // else
            //     strcpy(errormsg, "Unknown Function");
        }
        get_token();
    }
    else
        switch (tok_type)
        {
            case UPDATEVARIABLE:
                {
                    cout << "\nTEST" << token - 'A';
                    cout << "\n ::::101::::" << token;
                    cout.flush();
                    string var = token;
                    expression* result;
                    
                    if (local_vars_->count(var))
                        result = expression::literal_expression(local_vars_->at(var));
                    else if (global_vars_->count(var))
                        result = expression::literal_expression(global_vars_->at(var));
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
                    cout << "\n ::::10::::";
                    cout.flush();
                    expression* result =  expression::literal_expression(atof(token));
                    //result = atof(token);
                    get_token();
                    cout << "\n ::::11::::"<<result->get_value();
                    cout.flush();
                    return result;
                }
            default:
                {
                    THROW_LINE("Syntax Error")
                }
        }
}
// Obtain the next token.
void update_expression_evaluator::get_token()
{
    char *temp;
    tok_type = 0;
    temp = token;
    *temp = '\0';
    if (!*exp_ptr)  // at end of expression
        return;
    while (isspace(*exp_ptr))  // skip over white space
        ++exp_ptr; 
    if (strchr("+-*/%^=()", *exp_ptr)) 
    {
        tok_type = UPDATEDELIMITER;
        *temp++ = *exp_ptr++;  // advance to next char
    }
    else if (isalpha(*exp_ptr)) 
    {
        while (!strchr(" +-/*%^=()\t\r", *exp_ptr) && (*exp_ptr))
            *temp++ = *exp_ptr++;
        while (isspace(*exp_ptr))  // skip over white space
            ++exp_ptr;
        tok_type = (*exp_ptr == '(') ? UPDATEFUNCTION : UPDATEVARIABLE;
    }
    else if (isdigit(*exp_ptr) || *exp_ptr == '.')
    {
        while (!strchr(" +-/*%^=()\t\r", *exp_ptr) && (*exp_ptr))
            *temp++ = toupper(*exp_ptr++);
        tok_type = UPDATENUMBER;
    }
    *temp = '\0';
    if ((tok_type == UPDATEVARIABLE) && (token[1]))
        strcpy(errormsg, "Only first letter of variables is considered, NAAAAAT");
}

expression* update_expression_evaluator::parse_update_expr(string input, map<string, int>* local_vars, map<string, int>* global_vars)
{
    cout << "\n ::::1::::" << input;
    update_expression_evaluator ob(local_vars, global_vars);
    cout << "\n ::::3::::" << input;
    cout << "\n ::::2::::" << input;
    cout.flush();
    expression* ans = ob.eval_exp((char*)input.substr(0,input.length()).c_str());
    cout << "\n13: " <<"\n";
    cout.flush();
    cout << "\n ::::88::::"<<ans->get_value();
    cout.flush();
    return ans;
}