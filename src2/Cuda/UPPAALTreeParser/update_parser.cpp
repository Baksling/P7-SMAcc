#include "update_parser.h"

using namespace std;
using namespace helper;

update_expression* update_parser::parse(string decl, map<string, int> vars)
{
    string line_wo_ws = replace_all(decl, " ", "");
    string nums = take_after(line_wo_ws, '=');
    
    return update_expression_evaluator::parse_update_expr(nums,vars);
}
