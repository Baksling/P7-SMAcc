#ifndef ABSTRACT_PARSER_H
#define ABSTRACT_PARSER_H

#pragma once
#include <unordered_map>

#include "helper_methods.h"
#include "../engine/Domain.h"

class abstract_parser
{
public:
    virtual ~abstract_parser() = default;
    virtual network parse(const std::string& file) = 0;
    virtual std::unordered_map<int, std::string>* get_nodes_with_name() = 0;
    virtual std::unordered_map<int, std::string>* get_clock_names() = 0;
    virtual std::unordered_map<int, int>* get_subsystems() = 0;
    virtual std::unordered_map<int, string>* get_template_names() = 0;

    static bool try_parse_block_threads(const std::string& str, unsigned* out_blocks, unsigned* out_threads)
    {
        const std::list<std::string> split = helper::split_all(str, ",");
        if(split.size() != 2) return false;
        const string& blocks = split.front();
        const string& threads = split.back();

        try
        {
            *out_blocks  = static_cast<unsigned>(stoi(blocks));
            *out_threads = static_cast<unsigned>(stoi(threads));
        }
        catch(invalid_argument&)
        {
            return false;
        }
        return true;
    }

    static bool try_parse_units(const std::string& str, bool* is_time, double* value)
    {
        const char unit = str.back();
        if(unit != 's' && unit != 't') return false;
        *is_time = unit == 't';

        const string val = str.substr(0, str.size() - 1);

        try
        {
            *value = stod(val);
        }
        catch(invalid_argument&)
        {
            return false;
        }
        return true;
    }
};

#endif