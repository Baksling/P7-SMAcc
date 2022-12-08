#include "expr_compiler_visitor.h"

expr_compiler_visitor::expr_compiler_visitor()
{
    this->init_store();
}

void expr_compiler_visitor::init_store()
{
    this->expr_store_ << "switch(ex->compile_id){\n";
}


void expr_compiler_visitor::visit(network* a)
{
    if(has_visited(a)) return;
    
    accept(a, this);
}

void expr_compiler_visitor::visit(node* n)
{
    if(has_visited(n)) return;
    
    accept(n, this);
}

void expr_compiler_visitor::visit(edge* e)
{
    if(has_visited(e)) return;
    
    accept(e, this);
}

void expr_compiler_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;
    
    accept(c, this);
}

void expr_compiler_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;
    
    accept(cv, this);
}

void expr_compiler_visitor::visit(update* u)
{
    if(has_visited(u)) return;
    
    accept(u, this);
}

void expr_compiler_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;
    if(this->finalized_)
        throw std::runtime_error("Cannot add expression to compiled expression collection, as it is already marked as finalied.");

    if(!IS_LEAF(ex->operand))
    {
        const int id = this->compile_id_enumerator_++;

        this->expr_store_ << "case " << id << ": return";
        compile_expr(this->expr_store_, ex, true);
        this->expr_store_ << ";\n";
        
        ex->operand = expr::compiled_ee;
        ex->left = nullptr;
        ex->right = nullptr;
        ex->compile_id = id;   
    }
    
    accept(ex, this);
}

std::stringstream& expr_compiler_visitor::get_compiled_expressions()
{
    if(!this->finalized_)
    {
        this->expr_store_ << "\n}\n";
        this->finalized_ = true;
    }
    
    return this->expr_store_;
}

void expr_compiler_visitor::clear()
{
    this->finalized_ = false;
    this->expr_store_.clear();
    this->compile_id_enumerator_ = 0;
}


void expr_compiler_visitor::compile_expr(std::stringstream& ss, const expr* e, const bool is_root = false)
{
    if(is_root && IS_LEAF(e->operand)) return;
    ss << '(';
    switch (e->operand)
    {
    case expr::literal_ee: ss << e->value; break;
    case expr::clock_variable_ee: ss << "state->variables.store[" << e->variable_id << "].value"; break;
    case expr::random_ee: ss << "(1.0 - curand_uniform_double())*"; compile_expr(ss, e->left);  break;
    case expr::plus_ee:  compile_expr(ss, e->left); ss << '+'; compile_expr(ss, e->right); break;
    case expr::minus_ee: compile_expr(ss, e->left); ss << '-'; compile_expr(ss, e->right); break;
    case expr::multiply_ee: compile_expr(ss, e->left); ss << '*'; compile_expr(ss, e->right); break;
    case expr::division_ee: compile_expr(ss, e->left); ss << '/'; compile_expr(ss, e->right); break;
    case expr::power_ee: ss << "pow("; compile_expr(ss, e->left); ss << ','; compile_expr(ss, e->right); ss << ')'; break;
    case expr::negation_ee: ss << '-'; compile_expr(ss, e->left); break;
    case expr::sqrt_ee: ss << "sqrt("; compile_expr(ss, e->left); ss << ')'; break;
    case expr::less_equal_ee: compile_expr(ss, e->left); ss << "<="; compile_expr(ss, e->right); break;
    case expr::greater_equal_ee: compile_expr(ss, e->left); ss << ">="; compile_expr(ss, e->right); break;
    case expr::less_ee: compile_expr(ss, e->left); ss << '<'; compile_expr(ss, e->right); break;
    case expr::greater_ee: compile_expr(ss, e->left); ss << '>'; compile_expr(ss, e->right); break;
    case expr::equal_ee: compile_expr(ss, e->left); ss << "=="; compile_expr(ss, e->right); break;
    case expr::not_equal_ee: compile_expr(ss, e->left); ss << "!="; compile_expr(ss, e->right); break;
    case expr::not_ee:  ss<<'!'; compile_expr(ss, e->left); break;
    case expr::conditional_ee: compile_expr(ss, e->left);
        ss << '?'; compile_expr(ss, e->right);
        ss << ':'; compile_expr(ss, e->conditional_else);
        break;
    case expr::compiled_ee: throw std::runtime_error("Cannot compile already compiled expression");
    }
    ss << ')';
}

