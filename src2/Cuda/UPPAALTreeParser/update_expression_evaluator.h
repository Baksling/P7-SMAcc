
#ifndef UPDATE_EXPRESSION_EVALUATOR_H
#define UPDATE_EXPRESSION_EVALUATOR_H

#include <iostream>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <math.h> 
#include <cctype>
#include <cstring>
#include <iostream>
#include <map>

#include "parser_exception.h"
#include "../Domain/UpdateExpressions/update_expression.h"



#define PI 3.14159265358979323846 
 
using namespace std;
 
enum update_types { UPDATEDELIMITER = 1, UPDATEVARIABLE, UPDATENUMBER, UPDATEFUNCTION };
const int UPDATE_NUMVARS = 26;
class update_expression_evaluator {
    char *exp_ptr; // points to the expression
    char token[256]; // holds current token
    char tok_type; // holds token's type
    double vars[UPDATE_NUMVARS]; // holds variable's values
    map<string, int> vars_;
    update_expression* eval_exp1();
    update_expression* eval_exp2();
    update_expression* eval_exp3();
    update_expression* eval_exp4();
    update_expression* eval_exp5();
    update_expression* eval_exp6();
    update_expression* eval_exp(char *exp);
    void get_token();
public:
    update_expression_evaluator(map<string, int> vars);
    static update_expression* parse_update_expr(string input, map<string, int> vars);
    
    char errormsg[64];
};


#endif