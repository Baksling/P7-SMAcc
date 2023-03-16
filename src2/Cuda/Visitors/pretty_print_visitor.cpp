#include "pretty_print_visitor.h"

pretty_print_visitor::pretty_print_visitor(std::ostream* stream,
    std::unordered_map<int, std::string>* node_names, std::unordered_map<int, std::string>* variable_names)
{
    this->stream_ = stream;
    this->var_names_ = variable_names;
    this->node_names_ = node_names;
    visit_set_.clear();
}


void pretty_print_visitor::visit(network* a)
{
    *this->stream_ << "MODEL START\n";
    
    visitor::accept(a, this);
    *this->stream_ << "MODEL END\n";
}

void pretty_print_visitor::visit(node* n)
{
    if(this->has_visited(n)) return;
    this->scope_ = 1;

    *this->stream_ << "NODE = id: " << n->id << " (" << print_node_name(n->id)
                   << ") | type: " << node_type_to_string(n)
                   << " | lambda: " << pretty_expr(n->lamda) << '\n';

    this->scope_++;
    accept(n, this);
    if(this->scope_ > 0) this->scope_--;
}

void pretty_print_visitor::visit(edge* e)
{
    if(this->has_visited(e)) return;

    print_indent();
    
    *this->stream_ << "EDGE = dest: " << e->dest->id << " ("; print_node_name(e->dest->id);
    *this->stream_ << ") | channel id: " << e->channel
                   << " | weight: " << pretty_expr(e->weight)
                   << "\n";

    this->scope_++;
    accept(e, this);
    this->scope_--;
}

void pretty_print_visitor::visit(constraint* c)
{
    if(this->has_visited(c)) return;
    print_indent();

    *this->stream_ << "CONSTRAINT = "
        << (c->uses_variable
            ? "var" + std::to_string(c->variable_id) + "(" + print_var_name(c->variable_id) +")"
            : pretty_expr(c->value))
        << ' ' << constraint_type_to_string(c)
        << ' ' << pretty_expr(c->expression) << '\n';

    this->scope_++;
    accept(c, this);
    this->scope_--;
}

void pretty_print_visitor::visit(clock_var* cv)
{
    if(this->has_visited(cv)) return;
    print_indent();
    *this->stream_ << "Var " <<  cv->id << " ("
                   << print_var_name(cv->id)
                   << "): | value: " << cv->value
                   << " | rate: " << cv->rate
                   << " | track: " << (cv->should_track ? "True" : "False")
                   << "\n";

    this->scope_++;
    accept(cv, this);
    this->scope_--;
}

void pretty_print_visitor::visit(update* u)
{
    if(this->has_visited(u)) return;
    print_indent();
    *this->stream_
        << "UPDATE = var " << u->variable_id << " ("
        << print_var_name(u->variable_id) << "): " << pretty_expr(u->expression) << '\n';

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

std::string pretty_print_visitor::print_node_name(const int id) const
{
    return (this->node_names_->count(id) ? this->node_names_->at(id) : std::string("_"));
}

std::string pretty_print_visitor::print_var_name(const int id) const
{
    return (this->var_names_->count(id) ? this->var_names_->at(id) : std::string("_"));
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
    case constraint::compiled_c: return "COMPILED";
    default: return "unknown";
    }
}

std::string pretty_print_visitor::node_type_to_string(const node* n)
{
    switch(n->type)
    {
    case node::location: return "Location";
    case node::goal: return "Goal";
    case node::branch: return "Branch point";
    case node::urgent: return "urgent";
    case node::committed: return "committed";
    default: return "unknown";
    }
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
    case expr::modulo_ee: return "%";
    case expr::less_equal_ee: return "<="; 
    case expr::greater_equal_ee: return ">="; 
    case expr::less_ee: return "<"; 
    case expr::greater_ee: return ">"; 
    case expr::equal_ee: return "=="; 
    case expr::not_equal_ee: return "!="; 
    case expr::not_ee: return "!"; 
    case expr::conditional_ee: return "if";
    case expr::compiled_ee: return "expr_id_" + std::to_string(ex->compile_id) ;
    default: return "unknown";
    }
}

std::string pretty_print_visitor::pretty_expr(const expr* ex)
{
    if(IS_LEAF(ex->operand)) return expr_type_to_string(ex);

    if(ex->operand == expr::conditional_ee)
    {
        const std::string left = pretty_expr(ex->left);
        const std::string right = pretty_expr(ex->right);
        const std::string cond_else = pretty_expr(ex->conditional_else);

        return "(if " + left + " then " + right + " else " + cond_else + ")" ;
    }
    
    std::string left;
    std::string right;
    if (ex->left != nullptr) left = pretty_expr(ex->left);
    else left = "";
    if (ex->right != nullptr) right = pretty_expr(ex->right);
    else right = "";

    return "(" + left + expr_type_to_string(ex) + right + ")";
}
