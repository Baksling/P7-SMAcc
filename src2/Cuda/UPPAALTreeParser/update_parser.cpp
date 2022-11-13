#include "update_parser.h"

using namespace std;
using namespace helper;

expression* update_parser::parse(string decl, unordered_map<string, int>* local_vars, unordered_map<string, int>* global_vars)
{
    string line_wo_ws = replace_all(decl, " ", "");
    string nums = take_after(line_wo_ws, '=');
    expression* res = update_expression_evaluator::parse_update_expr(nums,local_vars, global_vars);
    return res;
}
