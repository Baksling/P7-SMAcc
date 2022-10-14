#include "declaration_parser.h"

#include "arithmetic-expression-evaluator.h"
using namespace std;
using namespace helper;
using namespace evalguy;
// float calculate(float num1, float num2, string op)
// {
//     //cout << num1 << " " << num2 << " " << op << " \n";
//     if (op == "/")
//         return num1/num2;
//     if (op == "*")
//         return num1*num2;
//     if (op == "+")
//         return num1+num2;
//     if (op == "-")
//         return num1-num2;
//
//     THROW_LINE("OP NOT SEEN BEFORE: " + to_string(num1) + op + to_string(num2))
// }

// // float parse_plus_minus(string eq)
// // {
// //     string eq_string_ = replace_all(eq, " ", "");
// //     eq_string_ = replace_all(eq_string_, ";", "");
// //
// //     bool first_hit = false;
// //     float current_num = 0;
// //     string op;
// //     int last_index = 0;
// //     for (int i = 0; i < eq_string_.length(); i++)
// //     {
// //         if(in_array(eq_string_[i],{'/', '*','+','-'}))
// //         {
// //             cout << "\n"<< eq_string_ << "\n";
// //             float num = stof(eq_string_.substr(last_index, i));
// //             if (!first_hit)
// //             {
// //                 current_num = num;
// //                 first_hit = true;
// //             }
// //             else
// //             {
// //                current_num = calculate(current_num, num, op);
// //             }
// //             
// //             last_index = i+1;
// //             op = eq_string_[i];
// //         }
// //
// //         if (i == eq_string_.length()-1)
// //         {
// //             float num = stof(eq_string_.substr(last_index, i));
// //
// //             current_num = calculate(current_num, num, op);
// //
// //         }
// //     }
// //     return current_num;
// // }
// string parse_equation2(string eq_string)
// {
//     string eq_string_ = replace_all(eq_string, " ", "");
//     eq_string_ = replace_all(eq_string_, ";", "");
//
//     bool first_hit = false;
//     string plus_minus_str;
//     float current_num = 0;
//     string op;
//     int last_index = 0;
//     int count_index = 0;
//     for (int i = eq_string_.length()-1; i >= 0; i--, count_index++)
//     {
//         if(in_array(eq_string_[i],{'/', '*','+','-'}))
//         {
//             float num = stof(eq_string_.substr(last_index, count_index));
//             if (eq_string_[i] == '+')
//             {
//                 cout << "\nBEFORE P " <<current_num;
//                 if (!op.empty())
//                     current_num = calculate(current_num, num, op);
//                 else
//                     current_num = num;
//
//                 cout << eq_string_.substr(0,count_index+1);
//                 cout << "\nBEFORE P " <<current_num;
//                 current_num = stof(parse_equation2(eq_string_.substr(0,count_index+1))) + current_num;
//                 cout << "\nAFTER P " <<current_num;
//                 break;
//             }
//
//             if (eq_string_[i] == '-')
//             {
//                 if (!op.empty())
//                     current_num = calculate(current_num, num, op);
//                 else
//                     current_num = num;
//
//                 cout << "\nBEFORE M " <<current_num;
//                 current_num = stof(parse_equation2(eq_string_.substr(0,count_index+1))) - current_num;
//                 cout << "\nAFTER M " <<current_num;
//                 break;
//             }
//             
//             if (!first_hit)
//             {
//                 float num = stof(eq_string_.substr(i+1, count_index));
//                 last_index = i+1;
//                 current_num = num;
//                 cout << "\nFIRST HIT " <<current_num;
//                 first_hit = true;
//             }
//             else
//             {
//                 cout << "\nCALC " <<current_num << " " << op << num;
//                 current_num = calculate(current_num, num, op);
//                 cout << "\nAFTER CALC " <<current_num;
//             }
//             
//             
//             last_index = i+1;
//             op = eq_string[i];
//         }
//
//         if (i == 0)
//         {
//             float num = stof(eq_string_.substr(last_index, count_index));
//             if (!op.empty())
//                 current_num = calculate(current_num, num, op);
//             else
//                 current_num = num;
//
//         }
//     }
//     return to_string(current_num);
// }
//
// string reverse_equation(string eq_string)
// {
//     //cout<<eq_string;
//     string eq_string_ = replace_all(eq_string, " ", "");
//     eq_string_ = replace_all(eq_string_, ";", "");
//     int last_index = 0;
//     string result_string;
//     for (int i = 0; i < eq_string_.length(); i++)
//     {
//         if(in_array(eq_string_[i],{'+'}))
//         {
//             //cout << "!";
//             
//             result_string = "+" + eq_string_.substr(last_index, i) +  result_string;
//             cout << "\nHERE: "<<eq_string_.substr(last_index, i) + "\n";
//             cout << "\nHERE1.5: "<< result_string  + "\n";
//             last_index = i+1;
//         }
//         cout << "\n" << result_string.length() << " "<< eq_string.length();
//         if(result_string.length() == eq_string.length())
//         {
//             cout << "\nFFOIEWFOIEWFO: " << result_string.substr(1);
//
//             return result_string.substr(1);
//
//         }
//
//         if(i == eq_string_.length()-1)
//         {
//             result_string = eq_string_.substr(last_index, i) + result_string;
//             cout << "\nHERE2: "<<result_string + "\n";
//         }
//     }
//     //cout << "\nEND"<< result_string + "\n";
//     return result_string;
// }
//
// string parse_equation(string eq_string)
// {
//     string eq_string_ = replace_all(eq_string, " ", "");
//     eq_string_ = replace_all(eq_string_, ";", "");
//
//     bool first_hit = false;
//     string plus_minus_str;
//     float current_num = 0;
//     string op;
//     int last_index = 0;
//     for (int i = 0; i < eq_string_.length(); i++)
//     {
//         if(in_array(eq_string_[i],{'/', '*','+','-'}))
//         {
//             float num = stof(eq_string_.substr(last_index, i));
//             if (eq_string_[i] == '+')
//             {
//                 if (!op.empty())
//                     current_num = calculate(current_num, num, op);
//                 else
//                     current_num = num;
//                 
//                 current_num = current_num + stof(parse_equation(eq_string_.substr(i+1)));
//                 break;
//             }
//
//             if (eq_string_[i] == '-')
//             {
//                 if (!op.empty())
//                     current_num = calculate(current_num, num, op);
//                 else
//                     current_num = num;
//                 
//                 current_num = current_num - stof(parse_equation(eq_string_.substr(i+1)));
//                 break;
//             }
//             
//             if (!first_hit)
//             {
//                 current_num = num;
//                 first_hit = true;
//             }
//             else
//             {
//                 current_num = calculate(current_num, num, op);
//             }
//             
//             
//             last_index = i+1;
//             op = eq_string[i];
//         }
//
//         if (i == eq_string_.length()-1)
//         {
//             float num = stof(eq_string_.substr(last_index, i));
//             current_num = calculate(current_num, num, op);
//
//         }
//     }
//     return to_string(current_num);
// }

declaration number_parser(const string& line, declaration_types type)
{
    string line_wo_ws = replace_all(line, " ", "");
    string nums = take_after(line_wo_ws, '=');
    return declaration(type, take_while(line_wo_ws, '='), to_string(eval_expr(nums)), 1);
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
        
    int var_amount = 0;
    while(std::getline(clocks_stream, clock, ','))
    {
        string clock_without = clock;
        if (clock.find(';') != std::string::npos)
            clock_without = replace_all(clock, ";", "");

        if (clock_without.find('=') != std::string::npos)
        {
            string nums = take_after(clock_without, '=');
            result.emplace_back(clock_type, take_while(clock_without, '='), to_string(eval_expr(nums)),global_clock_id_counter_++);
        }
        else
            result.emplace_back(clock_type, clock_without, "0",global_clock_id_counter_++);
    }

    return result;
}

list<declaration> declaration_parser::parse(string decl)
{
    const list<string> lines = split_expr(decl, '\n');
    list<declaration> result;

    for (const auto& line : lines)
    {
        string line_trimmed = remove_while(line, ' ');
        //cout << "\n\n\n HERE:|" << line_trimmed<<"|" << "\n\n"; 
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

        if (line_trimmed.substr(0, 6) == "double")
            result.emplace_back(number_parser(line_trimmed.substr(6), double_type));
        
        if (line_trimmed.substr(0, 3) == "int")
            result.emplace_back(number_parser(line_trimmed.substr(3), int_type));
        //result.emplace_back(clock_type,line_trimmed,"3.0", 0);
    }
    
    return result;
}
