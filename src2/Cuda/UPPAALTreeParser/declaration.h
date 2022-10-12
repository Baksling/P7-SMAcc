#pragma once
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <type_traits>
#include "parser_exception.h"
#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

enum declaration_types {clock_type, double_type, int_type};

using namespace std;


class declaration
{
private:
    declaration_types type_;
    string var_name_;
    int i_value_;
    float f_value_;
    std::map<declaration_types, std::string> print_map_;
    void populate_map()
    {
        print_map_.insert_or_assign(clock_type, "clock");
        print_map_.insert_or_assign(double_type, "double");
        print_map_.insert_or_assign(int_type, "int");
    }
public:
    declaration(const declaration_types type, const string& var_name, int val) : f_value_(-1)
    {
        this->type_ = type;
        this->var_name_ = var_name;
        this-> i_value_ = val;
    }
    
    declaration(const declaration_types type, const string& var_name, float val) : i_value_(-1)
    {
        this->type_ = type;
        this->var_name_ = var_name;
        this-> f_value_ = val;
    }
    
    
    declaration_types get_type() const
    {
        return this->type_;
    }
    
    string get_name() const
    {
        return this->var_name_;
    }
    
    void get_value(int& out) const
    {
        out = this->i_value_;
    }

    void get_value(float& out) const
    {
        out = this->f_value_;
    }

    void to_string()
    {
        populate_map();
        cout << "TYPE: " + print_map_[type_] << " " << var_name_ << " = " << (type_ == int_type ? i_value_ : f_value_) << "\n";
    }
    
};
