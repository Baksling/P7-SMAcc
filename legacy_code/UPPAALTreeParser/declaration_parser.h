#pragma once
#include <list>
#include <string>
#include <list>
#include <map>
#include <string>

#ifndef DECLARATION_PARSER
#define DECLARATION_PARSER
#include "helper_methods.h"

#include "declaration.h"

using namespace std;



class declaration_parser
{
private:
    int global_clock_id_counter_ = 0;
    int global_chan_id_counter_ = 0;
    list<declaration> parse_keyword(const string& lines, declaration_types type);
    void number_parser(const string& input_string, list<declaration>* result);
    string val_;
    const map<declaration_types, string> decl_type_map_ {{bool_type, "bool"}, {clock_type,"clock"}, {double_type,"double"}, {int_type, "int"}, {chan_type, "broadcastchan"}};
public:
    std::list<declaration> parse(const std::string& decl);
};

#endif