#pragma once
#include <string>
#include "helper_methods.h"
using namespace std;
using namespace helper;


struct extract_input
{
    const string input;
    
public:
    explicit extract_input(const string& input) : input(remove_whitespace(input))
    {
    }

    
};

struct extract_if_statement : extract_input
{
    const string condition;
    const string if_true;
    const string if_false;
    
    explicit extract_if_statement(const string& ex)
        : extract_input(ex), condition(), if_true(), if_false()
    {
    }

    explicit extract_if_statement(const string& condition, const string& if_true, const string& if_false, const string& trimmed_input) : extract_input(trimmed_input),
        condition(condition), if_true(if_true), if_false(if_false)
    {
    }
};

struct extract_condition : extract_input
{
    const string op;
    const string left;
    const string right;
    
    explicit extract_condition(const string& ex)
        : extract_input(ex), op(), left(), right()
    {
    }

    explicit extract_condition(const string& op,const string& left,const string& right, const string& trimmed_input)
        : extract_input(trimmed_input), op(op), left(left), right(right)
    {
    }
};

struct extract_assignment : extract_input
{
    const string left;
    const string right;
    
    explicit extract_assignment(const string& ex)
        : extract_input(ex), left(), right()
    {
    }

    explicit extract_assignment(const string& left, const string& right, const string& ex)
        : extract_input(ex), left(left), right(right)
    {
    }
};

struct extract_sync : extract_input
{
    const bool is_listener;
    const string keyword;
    
    explicit extract_sync(const string& ex)
            : extract_input(ex), is_listener(), keyword()
        {
    }
    
     explicit extract_sync(const bool is_listener, const string& keyword,const string& ex)
        : extract_input(ex),
          is_listener(is_listener),
          keyword(keyword)
    {
    }

};

struct extract_probability : extract_input
{
    const string value;
    explicit extract_probability(const string& ex)
       : extract_input(ex),
         value()
    {
    }
    
    explicit extract_probability(const string& value, const string& ex)
        : extract_input(ex),
          value(value)
    {
    }
};

struct extract_node_id : extract_input
{
    const int id;

    extract_node_id(const string& ex)
        : extract_input(ex),
          id()
    {
    }
    
    extract_node_id(const int id, const string& ex)
        : extract_input(ex),
          id(id)
    {
    }
};

class string_extractor
{
public:
    static extract_if_statement extract(const extract_if_statement& extract);
    static extract_condition extract(const extract_condition& extract);
    static extract_assignment extract(const extract_assignment& extract);
    static extract_sync extract(const extract_sync& extract);
    static extract_probability extract(const extract_probability& extract);
    static int extract(const extract_node_id& extract);
};

