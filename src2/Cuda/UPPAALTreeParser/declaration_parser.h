﻿#pragma once
#include <list>
#include <string>
#include <list>
#include <string>

#include "helper_methods.h"

#include "declaration.h"

using namespace std;

class declaration_parser
{
private:
    int global_clock_id_counter_ = 0;
    int global_chan_id_counter_ = 0;
    list<declaration> parse_keyword(const string& line, declaration_types type);
    string val_;
public:
    std::list<declaration> parse(const std::string& decl);
};
