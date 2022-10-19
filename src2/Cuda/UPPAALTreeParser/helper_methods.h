#pragma once
#include <list>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "parser_exception.h"



using namespace std;

namespace helper
{

    inline string take_after(string s, char while_char)
    {
        return s.substr(s.find(while_char)+1);
    }

    inline string take_while(string s, char while_char)
    {
        return s.substr(0,s.find(while_char));
    }

    inline string remove_while(string s, char while_char)
    {
        for (int i = 0; i<s.length(); i++)
        {
            if (s[i] != while_char)
                return s.substr(i);
        }

        THROW_LINE("how")
    }
    
    inline string replace_all(std::string str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
        return str;
    }

    inline bool in_array(const char &value, const vector<char> &array)
    {
        return std::find(array.begin(), array.end(), value) != array.end();
    } 

    inline int get_expr_value(const string& expr)
    {
        const string expr_wo_ws = replace_all(expr, " ", "");
        unsigned long long index = expr_wo_ws.length();
        while (true)
        {
            if (index == 0)
            {
                return stoi(expr_wo_ws);
            }
            
            if (!in_array(expr_wo_ws[--index], {'1','2','3','4','5','6','7','8','9','0'}))
            {
                return stoi(expr_wo_ws.substr(index+1));
            }
        }
    }

    inline float get_expr_value_float(const string& expr)
    {
        const string expr_wo_ws = replace_all(expr, " ", "");
        unsigned long long index = expr_wo_ws.length();
        while (true)
        {
            if (index == 0)
            {
                return stof(expr_wo_ws);
            }
            
            if (in_array(expr_wo_ws[--index], {'=',' ','<','>'}))
            {
                return stof(expr_wo_ws.substr(index+1));
            }
        }
    }

    inline string get_expr_value_string(const string& expr)
    {
        const string expr_wo_ws = replace_all(expr, " ", "");
        unsigned long long index = expr_wo_ws.length();
        while (true)
        {
            if (index == 0)
            {
                return expr_wo_ws;
            }
            
            if (in_array(expr_wo_ws[--index], {'=',' ','<','>'}))
            {
                return expr_wo_ws.substr(index+1);
            }
        }
    }



// template <typename T>
// static T* list_to_arr(list<T> l)
// {
//     T* arr = static_cast<T*>(malloc(sizeof(T) * l.size()));
//     int k = 0;
//     for (T const &i: l) {
//         arr[k++] = i;
//     }
//     
//     return arr;
// }

    inline int xml_id_to_int(string id_string)
    {
        return stoi(id_string.replace(0,2,""));
    }

// int uppaal_tree_parser::get_timer_id(const string& expr) const
// {
//     const string expr_wout_spaces = replace_all(expr, string(" "), string(""));
//     int index = 0;
//
//     while (true)
//     {
//         if (static_cast<int>(expr.size()) == index)
//         {
//             THROW_LINE("sum tin wong")
//         }
//         
//         if (in_array(expr_wout_spaces[++index], {'<','>','='}))
//         {
//             break;
//         }
//     }
//
//     const string sub = expr_wout_spaces.substr(0, index);
//
//     if ( timers_map_.count(sub) == 0)
//     {
//         THROW_LINE("sum tin wong")
//     }
//     
//     return timers_map_.at(sub);
// }

    template <typename T>
    void insert_into_list(list<list<T>>* t_list, int index, T item)
    {
        auto l_front = t_list->begin();
        std::advance(l_front, index);
        l_front->emplace_back(item);
    }

    inline list<std::string> split_expr(const string& expr)
    {
        list<string> result;
        std::stringstream test(expr);
        std::string segment;
        while(std::getline(test, segment, '&'))
        {
            string s = replace_all(segment, "&", "");
            result.push_back(s);
        }
        return result;
    }

    inline list<std::string> split_expr(const string& expr, const char split_on)
    {
        list<string> result;
        std::stringstream test(expr);
        std::string segment;
        while(std::getline(test, segment, split_on))
        {

            result.push_back(segment);
        }
        return result;
    }
}

