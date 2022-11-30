#include "pretty_print_visitor.h"

pretty_print_visitor::pretty_print_visitor(std::ostream* stream)
{
    this->stream_ = stream;
}

void pretty_print_visitor::visit(automata* a)
{
    *this->stream_ << "MODEL START\n";
    
    visitor::accept(a, this);
    *this->stream_ << "MODEL END\n";
    *this->stream_ << "pully porky\n";
}

void pretty_print_visitor::visit(node* n)
{
    this->scope_ = 1;
    *this->stream_ << "NODE = id: " << n->id
                   << " | is branch: " << (n->is_branch_point ? "True" : "False")
                   << " | is goal: " << (n->is_goal ? "True" : "False");

    this->scope_++;
    print_indent();
    
    *this->stream_ << "lambda: " << pretty_expr(n->lamda)
                   << '\n';
    
    accept(n, this);
    if(this->scope_ > 0) this->scope_--;
}

void pretty_print_visitor::visit(edge* e)
{
    print_indent();
    
    *this->stream_ << "EDGE = dest: " << e->dest->id
                   << " | channel id: " << e->channel
                   << " | weight: " << pretty_expr(e->weight)
                   << '\n';

    this->scope_++;
    accept(e, this);
    this->scope_--;
}

void pretty_print_visitor::visit(constraint* c)
{
    print_indent();

    *this->stream_ << "CONSTRAINT = " << (c->uses_variable ? "var" + std::to_string(c->variable_id) : pretty_expr(c->value))
                   << ' ' << constraint_type_to_string(c) << ' ' << pretty_expr(c->expression) << '\n';

    this->scope_++;
    accept(c, this);
    this->scope_--;
}

void pretty_print_visitor::visit(clock_var* cv)
{
    print_indent();
    *this->stream_ << "COCK = id: " << cv->id
                   << " | value: " << cv->value
                   << " | rate: " << cv->rate
                   << " | track: " << (cv->should_track ? "True" : "False")
                   << "\n";

    this->scope_++;
    accept(cv, this);
    this->scope_--;
}

void pretty_print_visitor::visit(update* u)
{
    print_indent();
    *this->stream_ << "UPDATE = var " << u->variable_id << ": " << pretty_expr(u->expression) << '\n';

    this->scope_++;
    accept(u, this);
    this->scope_--;
}

void pretty_print_visitor::visit(expr* u)
{
    //Handled by each individual method, to make it prettier.
    // *this->stream_ << pretty_expr(u) << '\n';
    
    //no accept here, cuz we dont want to print expr tree :)
}

std::string pretty_print_visitor::constraint_type_to_string(const constraint* c)
{
    switch (c->operand)
    {
    case constraint::less_equal_c: return "<=";
    case constraint::less_c: return "<";
    case constraint::greater_equal_c: return ">=";
    case constraint::greater_c: return ">";
    case constraint::equal_c: return "==";
    case constraint::not_equal_c: return "!=";
    }

    return "unknown";
}

std::string pretty_print_visitor::expr_type_to_string(const expr* ex)
{
    switch (ex->operand) {
    case expr::literal_ee: return std::to_string(ex->value); 
    case expr::clock_variable_ee: return "v" + std::to_string(ex->variable_id); 
    case expr::random_ee: return "*random()"; 
    case expr::plus_ee: return "+"; 
    case expr::minus_ee: return "-"; 
    case expr::multiply_ee: return "*"; 
    case expr::division_ee: return "/"; 
    case expr::power_ee: return "^"; 
    case expr::negation_ee: return "~"; 
    case expr::sqrt_ee: return "sqrt"; 
    case expr::less_equal_ee: return "<="; 
    case expr::greater_equal_ee: return ">="; 
    case expr::less_ee: return "<"; 
    case expr::greater_ee: return ">"; 
    case expr::equal_ee: return "=="; 
    case expr::not_equal_ee: return "!="; 
    case expr::not_ee: return "!"; 
    case expr::conditional_ee: return "if"; 
    default: return "unknown";
    }
}

std::string pretty_print_visitor::pretty_expr(const expr* ex)
{
    if(IS_LEAF(ex->operand)) return expr_type_to_string(ex);
    std::string left;
    std::string right;
    if (ex->left != nullptr) left = pretty_expr(ex->left);
    else left = "";
    if (ex->right != nullptr) right = pretty_expr(ex->right);
    else right = "";

    return "(" + left + expr_type_to_string(ex) + right + ")";
}
