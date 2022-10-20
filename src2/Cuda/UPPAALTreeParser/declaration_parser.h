#pragma once
#include <list>
#include <string>
#include <list>
#include <string>

#include "helper_methods.h"

#include "declaration.h"

class declaration_parser
{
private:
    int global_clock_id_counter_ = 0;
    list<declaration> parse_clocks(const string& line);
    // string parse_equation(string eq_string);
    // declaration number_parser(const string& line);
    string val_;
public:
    std::list<declaration> parse(const std::string& decl);
};
