#pragma once
#include "update_expression_evaluator.h"
#include <list>
#include <string>
#include <list>
#include <string>

#include "declaration.h"
#include "helper_methods.h"
class update_parser
{
private:
public:
    static update_expression* parse(string decl, map<string, int> vars);
};
