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

list<declaration> declaration_parser::parse_clocks(const string& line)
{
    list<declaration> result;
    const size_t clock_start = line.find("clock");
    const size_t clock_end = line.find(';');
        
    string clocks = line.substr(clock_start+5,clock_end-1);
    clocks = replace_all(clocks, string(" "), string(""));

    std::stringstream clocks_stream(clocks);
    std::string clock;
    
    while(std::getline(clocks_stream, clock, ','))
    {
        string clock_without = clock;
        if (clock.find(';') != std::string::npos)
            clock_without = replace_all(clock, ";", "");

        if (clock_without.find('=') != std::string::npos)
        {
            const string nums = take_after(clock_without, '=');
            result.emplace_back(clock_type, take_while(clock_without, '='), to_string(eval_expr(nums)),global_clock_id_counter_++);
        }
        else
            result.emplace_back(clock_type, clock_without, "0",global_clock_id_counter_++);
    }

    return result;
}

list<declaration> declaration_parser::parse(const string& decl)
{
    const list<string> lines = split_expr(decl, '\n');
    list<declaration> result;

    for (const auto& line : lines)
    {
        string line_trimmed = remove_while(line, ' ');
        if (line_trimmed.substr(0,2) == "//")
            continue;
        
        if (line_trimmed.substr(0, 5) == "clock")
        {
            list<declaration> cloc_decls = parse_clocks(line_trimmed);
            result.insert(result.end(), cloc_decls.begin(), cloc_decls.end());
        }
        
        else if (line_trimmed.substr(0, 5) == "const")
        {
            string const_string = remove_while(line_trimmed.substr(5), ' ');
            if (const_string.substr(0, 6) == "double")
                result.emplace_back(number_parser(const_string.substr(6), double_type));
            if (const_string.substr(0, 3) == "int")
                result.emplace_back(number_parser(const_string.substr(3), int_type));
        }

        else if (line_trimmed.substr(0, 6) == "double")
            result.emplace_back(number_parser(line_trimmed.substr(6), double_type));
        
        else if (line_trimmed.substr(0, 3) == "int")
            result.emplace_back(number_parser(line_trimmed.substr(3), int_type));
        //result.emplace_back(clock_type,line_trimmed,"3.0", 0);
    }
    
    return result;
}
