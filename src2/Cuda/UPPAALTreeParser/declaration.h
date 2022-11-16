#pragma once
#include <iostream>
#include <map>
#include <string>

#include "helper_methods.h"

using namespace std;

#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

enum declaration_types {clock_type, chan_type, double_type, int_type};

class declaration
{
private:
    declaration_types type_;
    string var_name_;
    double value_;
    int id_;

    std::map<declaration_types, std::string> print_map_;
    void populate_map()
    {
        if (print_map_.count(clock_type)) print_map_[clock_type] = "clock";
        else print_map_.insert(std::pair<declaration_types, std::string>(clock_type, "clock"));

        if (print_map_.count(double_type)) print_map_[double_type] = "double";
        else print_map_.insert(std::pair<declaration_types, std::string>(double_type, "double"));

        if (print_map_.count(int_type)) print_map_[int_type] = "int";
        else print_map_.insert(std::pair<declaration_types, std::string>(int_type, "int"));
        // print_map_.insert_or_assign(clock_type, "clock");
        // print_map_.insert_or_assign(double_type, "double");
        // print_map_.insert_or_assign(int_type, "int");
    }
public:
    declaration(const declaration_types type, const string& var_name, const string& val, const int id)
    {
        this->type_ = type;
        this->var_name_ = var_name;
        this->value_ = stod(val);
        this->id_ = id;
    }
    
    declaration_types get_type() const
    {
        return this->type_;
    }
    
    string get_name() const
    {
        return this->var_name_;
    }

    double get_value() const
    {
        return this->value_;
    }

    int get_id() const
    {
        return this->id_;
    }

    void to_string()
    {
        populate_map();
        cout << "TYPE: " + print_map_[type_] << " " << var_name_ << " = " << value_ << "\n";
    }
};
