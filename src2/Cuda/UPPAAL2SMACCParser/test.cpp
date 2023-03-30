#include "../engine/Domain.h"
#include "include/ctpg/ctpg.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace ctpg;
using namespace ctpg::buffers;
using namespace ctpg::ftors;

struct binary_op
{
    constexpr int operator()(int x1, char op, int x2) const
    {
        switch (op)
        {
        case '+':
            return x1 + x2;
        case '-':
            return x1 - x2;
        case '*':
            return x1 * x2;
        case '/':
            return x1 / x2;
        case '%':
            return x1 % x2;
        default:
            return 0;
        }
    }
};

constexpr int get_int(std::string_view sv)
{
    int sum = 0;
    for (size_t i = 0; i < sv.size(); ++i)
    {
        sum *= 10;
        int digit = sv[i] - '0';
        sum += digit;
    }
    return sum;
}

constexpr nterm<expr*> expression("expr");
constexpr nterm<string> decl("decl");
constexpr nterm<string> vardecl("vardecl");
constexpr nterm<string> vardecl2("vardecl2");
constexpr nterm<string> type("type");
constexpr nterm<string> typeprefix("typeprefix");
constexpr nterm<string> type_id("typeid");
constexpr nterm<string> var_id("varid");

constexpr char_term o_plus('+', 1, associativity::ltor);
constexpr char_term o_minus('-', 1, associativity::ltor);
constexpr char_term o_mul('*', 2, associativity::ltor);
constexpr char_term o_mol('%', 2, associativity::ltor);
constexpr char_term o_div('/', 2, associativity::ltor);

constexpr char number_pattern[] = "[1-9][0-9]*";
constexpr regex_term<number_pattern> number("number");

constexpr char id_pattern[] = "[a-zA-Z_][a-zA-Z0-9_]*";
constexpr regex_term<id_pattern> id("id");


constexpr parser p(
    decl,
    terms(number, o_plus, o_minus, o_mul, o_mol, o_div, ',', ';', '(', ')', '=', ":=", "const", "broadcast", "int", "clock", "chan", "double", id),
    nterms(expression, decl, vardecl,vardecl2, type, typeprefix, type_id, var_id),
    rules(
        decl() >= []() { return ""; },
        decl(vardecl, decl) >= [](auto a, auto rec) { return a + rec; },
        vardecl(type, var_id, vardecl2, ';') >= [](auto x, auto var_id, auto vardecls, auto semi) { return x + var_id + vardecls + semi.get_value(); },
        vardecl2() >= []() { return ""; },
        vardecl2(',', id, vardecl2) >= [](auto, auto v, auto vardecls) { return string(v.get_value()) + vardecls; },
        var_id(id, '=', expression) >= [](auto id, auto, auto expr) {return string(id.get_value()) + to_string(expr);},
        var_id(id, ":=", expression) >= [](auto id, auto, auto expr) {return string(id.get_value()) + to_string(expr);},
        var_id(id) >= [](auto id) { return id.get_value(); },
        expression(expression, '+', expression) >= binary_op{},
        expression(expression, '-', expression) >= binary_op{},
        expression(expression, '*', expression) >= binary_op{},
        expression(expression, '%', expression) >= binary_op{},
        expression(expression, '/', expression) >= binary_op{},
        expression('-', expression)[3] >= [](char, int x) { return -x; },
        expression('(', expression, ')') >= _e2,
        expression(number) >= [](const auto& sv){ return get_int(sv); },
        type(typeprefix, type_id) >= [](string prefix, string type_id) { return prefix + ":" + type_id; },
        typeprefix("const") >= [](auto v) { return v.get_value(); },
        typeprefix("broadcast") >= [](auto v) { return v.get_value(); },
        type_id("int") >= [](auto v) { return v.get_value(); },
        type_id("clock") >= [](auto v) { return v.get_value(); },
        type_id("chan") >= [](auto v) { return v.get_value(); },
        type_id("double") >= [](auto v) { return v.get_value(); }
    )
);

// constexpr auto res_ok = p.parse(cstring_buffer("-120 * 2 / 10"));
// constexpr int v = res_ok.value();

// constexpr auto res_fail = p.parse(cstring_buffer("--"));
// constexpr bool b = res_fail.has_value();

int main(int argc, char* argv[])
{
    // if (argc != 2)
    // {
    //     p.write_diag_str(std::cout);

    //     std::cout << std::endl << "constexpr parse: " << v << std::endl;
    //     static_assert(b == false);
    //     return 0;
    // }

    auto res = p.parse(parse_options{}.set_verbose(), string_buffer(argv[1]), std::cerr);
    if (res.has_value())
    {
        string rv = res.value();
        std::cout << "runtime parse: " << rv << std::endl;
    }
    return 0;
}
