#include "declaration_parser.h"

#include "arithmetic-expression-evaluator.h"
using namespace std;
using namespace helper;
using namespace evalguy;

declaration number_parser(const string& line, declaration_types type)
{
    const string line_wo_ws = replace_all(line, " ", "");
    const string nums = take_after(line_wo_ws, '=');
    return declaration{type, take_while(line_wo_ws, '='), to_string(eval_expr(nums)), 1};
}

list<declaration> declaration_parser::parse_keyword(const string& line, declaration_types type)
{
    string keyword;
    if (type == clock_type)
    {
        keyword = "clock";
    }
    else if (type == chan_type)
    {
        keyword = "broadcastchan";
    }
    
    list<declaration> result;
    const size_t keyword_start = line.find(keyword);
    const size_t keyword_end = line.find(';');
        
    string keyword_vals = line.substr(keyword_start+keyword.length(),keyword_end-1);
    keyword_vals = replace_all(keyword_vals, string(" "), string(""));

    std::stringstream keyword_vals_stream(keyword_vals);
    std::string keyword_val;
    
    while(std::getline(keyword_vals_stream, keyword_val, ','))
    {
        string keyword_val_without = keyword_val;
        if (keyword_val.find(';') != std::string::npos)
            keyword_val_without = replace_all(keyword_val, ";", "");

        if (keyword_val_without.find('=') != std::string::npos)
        {
            const string nums = take_after(keyword_val_without, '=');
            result.emplace_back(type, take_while(keyword_val_without, '='), to_string(eval_expr(nums)), type == clock_type ? global_clock_id_counter_++ : global_chan_id_counter_++);
        }
        else
            result.emplace_back(type, keyword_val_without, "0", type == clock_type ? global_clock_id_counter_++ : global_chan_id_counter_++);
    }

    return result;
}

bool is_this_keyword(string expr, string keyword)
{
    return expr.substr(0, keyword.length()) == keyword;
}

list<declaration> declaration_parser::parse(const string& decl)
{
    const list<string> lines = split_expr(decl, '\n');
    list<declaration> result;

    for (const auto& line : lines)
    {
        if (line.empty())
            continue;
        
        string line_trimmed = remove_while(line, ' ');
        //Remove tabs
        line_trimmed = remove_while(line, '\v');
        
        if (line_trimmed.substr(0,2) == "//")
            continue;

        //cout << "\n!! LETS GOOO1:" << line_trimmed<<":";
        //cout.flush();
        
        if (is_this_keyword(line_trimmed,"clock"))
        {
            list<declaration> cloc_decls = parse_keyword(line_trimmed, clock_type);
            result.insert(result.end(), cloc_decls.begin(), cloc_decls.end());
        }
        
        else if (is_this_keyword(line_trimmed,"const"))
        {   
            string const_string = remove_while(line_trimmed.substr(5), ' ');
            if (is_this_keyword(const_string,"double"))
                result.emplace_back(number_parser(const_string.substr(6), double_type));
            if (is_this_keyword(const_string,"int"))
                result.emplace_back(number_parser(const_string.substr(3), int_type));
        }

        else if (is_this_keyword(line_trimmed, "broadcastchan"))
        {
            //cout << "\n!! LETS GOOO:" << line_trimmed<<":";
            //cout.flush();
            list<declaration> chan_decls = parse_keyword(line_trimmed, chan_type);
            //cout << "\n CHANSIZE:"<< chan_decls.size();
            //cout << "\n FRONT - BACK:" << chan_decls.front().get_name() << ":" << chan_decls.back().get_name();
            result.insert(result.end(), chan_decls.begin(), chan_decls.end());
        }

        else if (is_this_keyword(line_trimmed, "double"))
            result.emplace_back(number_parser(line_trimmed.substr(6), double_type));
        
        else if (is_this_keyword(line_trimmed, "int"))
            result.emplace_back(number_parser(line_trimmed.substr(3), int_type));
        //result.emplace_back(clock_type,line_trimmed,"3.0", 0);
    }
    
    return result;
}
