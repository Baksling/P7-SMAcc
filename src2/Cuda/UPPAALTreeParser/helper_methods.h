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
    inline string take_after(const string& s, const char while_char)
    {
        return s.substr(s.find(while_char)+1);
    }

        inline string take_while(const string& s, const char while_char)
    {
        return s.substr(0,s.find(while_char));
    }

    inline string remove_while(const string& s, const char while_char)
    {
        for (size_t i = 0; i < s.length(); i++)
        {
            if (s[i] != while_char)
                return s.substr(i);
        }

        THROW_LINE("how: " + s + " \n WITH THE WHILE_CHAR: " + while_char)
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

    inline bool does_not_contain(const string& input, const string& does_not_contain )
    {
        // const bool not_contained = input.find_first_not_of(does_not_contain) != std::string::npos;

        //I BRUTE FORCE IT!
        
        for (int i = 0; i < static_cast<int>(input.length()); ++i)
        {
            for (int j = 0; j < static_cast<int>(does_not_contain.length()); ++j)
            {
                if (i + j >= static_cast<int>(input.length())) continue;
                
                if (input.c_str()[i + j] != does_not_contain.c_str()[j]) break;

                if (j == static_cast<int>(does_not_contain.length()) - 1) return false;
            }
        }

        return true;
        
        // return not_contained;
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
        string expr_wo_ws = replace_all(expr, " ", "");
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

    inline int xml_id_to_int(string id_string)
    {
        return stoi(id_string.replace(0,2,""));
    }

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

