#include "update_expression_evaluator.h"

// Parser constructor.
update_expression_evaluator::update_expression_evaluator(map<string, int> vars)
{
    this->vars_ = vars;
    int i;
    exp_ptr = NULL;
    for (i = 0; i < UPDATE_NUMVARS; i++)
        this->vars[i] = 0.0;
    errormsg[0] = '\0';
}

// Parser entry point.
update_expression* update_expression_evaluator::eval_exp(char *exp)
{
    errormsg[0] = '\0';
    exp_ptr = exp;
    get_token();
    
    if (!*token) 
    {
        THROW_LINE("No Expression Present")
    }
    
    update_expression* result = eval_exp1();
    if (*token) // last token must be null
    {
        THROW_LINE("Syntax Error")
    }
    
    return result;
}
// Process an assignment.
update_expression* update_expression_evaluator::eval_exp1()
{
    int slot;
    char temp_token[80];
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
            update_expression* result = eval_exp2();
            //vars[slot] = result;
            return result;
        }
    }
    return eval_exp2();
}
// Add or subtract two terms.
update_expression* update_expression_evaluator::eval_exp2()
{
    char op;
    double temp;
    update_expression* result = eval_exp3();
    while ((op = *token) == '+' || op == '-')
    {
        get_token();
        update_expression* temp = eval_exp3();
        switch (op) 
        {
        case '-':
            return update_expression::minus_expression(result, temp);
        case '+':
            return update_expression::plus_expression(result, temp);
        }
    }
}
// Multiply or divide two factors.
update_expression* update_expression_evaluator::eval_exp3()
{
    char op;
    double temp;
    update_expression* result = eval_exp4();
    while ((op = *token) == '*' || op == '/') 
    {
        get_token();
        update_expression* temp = eval_exp4();
        switch (op) 
        {
        case '*':
            return update_expression::multiply_expression(result, temp);
            break;
        case '/':
            return update_expression::division_expression(result, temp);
            break;
        }
    }
}
// Process an exponent.
update_expression* update_expression_evaluator::eval_exp4()
{
    double temp;
    update_expression* result = eval_exp5();
    while (*token == '^')
    {
        // get_token();
        eval_exp5();
        // result = pow(result, temp); TODO Parse ^
    }
    return result;
}
// Evaluate a unary + or -.
update_expression* update_expression_evaluator::eval_exp5()
{
    char op;
    op = 0;
    if ((tok_type == UPDATEDELIMITER) && *token == '+' || *token == '-')
    {
        op = *token;
        get_token();
    }
    update_expression* result = eval_exp6();
    return result;
    if (op == '-')
    {
        //TODO Parse NEgate
        
    }
}
// Process a function, a parenthesized expression, a value or a variable
update_expression* update_expression_evaluator::eval_exp6()
{
    bool isfunc = (tok_type == UPDATEFUNCTION);
    char temp_token[80];
    if (isfunc)
    {
        strcpy(temp_token, token);
        get_token();
    } 
    if ((*token == '(')) 
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
                update_expression* result = update_expression::variable_expression(vars_.at(token));
                get_token();
                return result;
            }
            //result = vars[*token - 'A'];
            
        case UPDATENUMBER:
            {
                update_expression* result =  update_expression::literal_expression(atof(token));
                //result = atof(token);
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
            *temp++ = toupper(*exp_ptr++);
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
        strcpy(errormsg, "Only first letter of variables is considered");
}

update_expression* update_expression_evaluator::parse_update_expr(string input, map<string, int> vars)
{
    update_expression_evaluator ob(vars);
    update_expression* ans = ob.eval_exp((char*)input.substr(0,input.length()).c_str());
    return ans;
}