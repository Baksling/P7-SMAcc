#include "update_parser.h"

using namespace std;
using namespace helper;

expression* update_parser::parse(string decl, map<string, int>* local_vars, map<string, int>* global_vars)
{
    string line_wo_ws = replace_all(decl, " ", "");
    string nums = take_after(line_wo_ws, '=');
    expression* res = update_expression_evaluator::parse_update_expr(nums,local_vars, global_vars);
    cout << "\n::::::::::" << (res == nullptr) << "hehe";
    cout.flush();
    cout << "\n::::::::::" << (res->get_value()) << "hehe";
    cout.flush();
    return res;
}
