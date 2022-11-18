#include "string_extractor.h"

string get_constraint_op(const string& expr)
{
    if(expr.find("<=") != std::string::npos)
        return "<=";
    if(expr.find(">=") != std::string::npos)
        return ">=";
    if(expr.find("==") != std::string::npos)
        return "==";
    if(expr.find("!=") != std::string::npos)
        return "!=";
    if(expr.find('<') != std::string::npos)
        return "<";
    if(expr.find('>') != std::string::npos)
        return ">";
    THROW_LINE("Operand in " + expr + " not found, sad..");
}

inline extract_if_statement string_extractor::extract(const extract_if_statement& extract)
{
    // condition?if_true:if_false;
    const string input = extract.input;
    const string condition = replace_all(take_while(input, "?"), " ", "");
    const string if_true = replace_all(take_while(take_after(input, "?"), ":"), " ", "");
    const string if_false = replace_all(take_after(input, ":"), ";", "");
    return extract_if_statement(condition, if_true, if_false,input);
}

inline extract_condition string_extractor::extract(const extract_condition& extract)
{
    // left op right
    const string condition = extract.input;
    const string op = get_constraint_op(condition);
    const string left = take_while(condition, op);
    const string right = take_after(condition, op);
    return extract_condition(op, left, right,condition);
}

inline extract_assignment string_extractor::extract(const extract_assignment& extract)
{
    // left_side=right_side
    const string left_side = take_while(extract.input, "=");
    const string right_side_of_equal = take_after(extract.input, "=");
    return extract_assignment(left_side,right_side_of_equal,extract.input);
}

inline extract_sync string_extractor::extract(const extract_sync& extract)
{
    const bool is_listener = extract.input.find('?')!=std::string::npos;
    const string sync_keyword = replace_all(extract.input, is_listener ? "?" : "!", "");
    return extract_sync(is_listener, sync_keyword, extract.input);
}

inline extract_probability string_extractor::extract(const extract_probability& extract)
{
    return extract_probability(extract.input, extract.input);
}

inline int string_extractor::extract(const extract_node_id& extract)
{
    string result = extract.input;
    return stoi(result.replace(0,2,""));
}