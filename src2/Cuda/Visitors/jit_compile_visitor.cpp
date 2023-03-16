#include "jit_compile_visitor.h"

#include <iostream>

jit_compile_visitor::jit_compile_visitor() : clocks_(arr<clock_var>::empty())
{
    this->init_store();
}

void jit_compile_visitor::init_store()
{
    this->expr_store_ << "switch(ex->compile_id){\n";
    this->con_store_ << "switch(con->compile_id){\n";
    this->invariant_store_ << "switch(con->compile_id){\n";
}


void jit_compile_visitor::visit(network* a)
{
    if(has_visited(a)) return;

    this->clocks_ = a->variables;

    for (int i = 0; i < a->automatas.size; ++i)
    {
        visit(a->automatas.store[i]);
    }
}

void jit_compile_visitor::visit(node* n)
{
    if(has_visited(n)) return;

    accept(n, this);
}

void jit_compile_visitor::visit(edge* e)
{
    if(has_visited(e)) return;
    
    accept(e, this);
}

void jit_compile_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;

    if(this->finalized_)
        throw std::runtime_error("Cannot add constraint to compiled constraints collection, as it is already marked as finalied.");
    
    const int id = this->con_compile_id_enumerator_++;

    this->con_store_ << "case " << id << ": return ";
    compile_con(this->con_store_, this->expr_map_cache_, c);
    this->con_store_ << "; break;\n";
    
    if(c->uses_variable && IS_INVARIANT(c->operand))
    {
        this->invariant_store_ << "case " << id << ": v0 = ";
        const bool is_invariant = compile_invariant(this->invariant_store_, this->clocks_, c, this->expr_map_cache_);
        this->invariant_store_ << "; break;\n";
        if(!is_invariant) throw std::runtime_error("invariant cannot be compiled as invariant.");
    }

    c->operand = constraint::compiled_c;
    c->compile_id = id;
    // c->expression is compiled too, for safety ;) 

    accept(c, this);
}

void jit_compile_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;
    
    accept(cv, this);
}

void jit_compile_visitor::visit(update* u)
{
    if(has_visited(u)) return;
    
    accept(u, this);
}

void jit_compile_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;
    if(this->finalized_)
        throw std::runtime_error("Cannot add expression to compiled expression collection, as it is already marked as finalized.");

    if(ex->operand == expr::compiled_ee) return;
    if(!IS_LEAF(ex->operand))
    {
        const int id = this->expr_compile_id_enumerator_++;
        std::stringstream ss;
        
        this->expr_store_ << "case " << id << ": return";
        compile_expr(ss, ex, true);
        
        this->expr_map_cache_.insert(std::pair<expr*, std::string>( ex, ss.str() ));
        this->expr_store_ << ss.rdbuf();
        
        this->expr_store_ << ";\n";
        
        ex->operand = expr::compiled_ee;
        ex->left = nullptr;
        ex->right = nullptr;
        ex->compile_id = id;   
    }
    
    accept(ex, this);
}


void jit_compile_visitor::finalize()
{
    if(!this->finalized_)
    {
        this->expr_store_ << "\n}\n";
        this->con_store_ << "\n}\n";
        this->invariant_store_ << "\n}\n";
        this->finalized_ = true;
    }
}

std::stringstream& jit_compile_visitor::get_expr_compilation()
{
    if(!this->finalized_) throw std::runtime_error("compilers finalize method not called before results are requested.");

    return this->expr_store_;
}

std::stringstream& jit_compile_visitor::get_constraint_compilation()
{
    if(!this->finalized_) throw std::runtime_error("compilers finalize method not called before results are requested.");

    return this->con_store_;
}

std::stringstream& jit_compile_visitor::get_invariant_compilation()
{
    if(!this->finalized_) throw std::runtime_error("compilers finalize method not called before results are requested.");

    return this->invariant_store_;
}

void jit_compile_visitor::clear()
{
    visitor::clear();

    this->finalized_ = false;
    this->expr_store_.clear();
    this->con_store_.clear();
    this->invariant_store_.clear();
    this->expr_map_cache_.clear();
    this->expr_compile_id_enumerator_ = 0;
    this->con_compile_id_enumerator_ = 0;
    init_store();
}


void jit_compile_visitor::compile_expr(std::stringstream& ss, const expr* e, const bool is_root = false)
{
    if(is_root && IS_LEAF(e->operand)) return;
    ss << '(';
    switch (e->operand)
    {
    case expr::literal_ee: ss << e->value; break;
    case expr::clock_variable_ee: ss << "state->variables.store[" << e->variable_id << "].value"; break;
    case expr::random_ee: ss << "(1.0 - curand_uniform_double(state->random))*"; compile_expr(ss, e->left);  break;
    case expr::plus_ee:  compile_expr(ss, e->left); ss << '+'; compile_expr(ss, e->right); break;
    case expr::minus_ee: compile_expr(ss, e->left); ss << '-'; compile_expr(ss, e->right); break;
    case expr::multiply_ee: compile_expr(ss, e->left); ss << '*'; compile_expr(ss, e->right); break;
    case expr::division_ee: compile_expr(ss, e->left); ss << '/'; compile_expr(ss, e->right); break;
    case expr::power_ee: ss << "pow("; compile_expr(ss, e->left); ss << ','; compile_expr(ss, e->right); ss << ')'; break;
    case expr::negation_ee: ss << '-'; compile_expr(ss, e->left); break;
    case expr::sqrt_ee: ss << "sqrt("; compile_expr(ss, e->left); ss << ')'; break;
    case expr::modulo_ee: compile_expr(ss, e->left); ss << '%'; compile_expr(ss, e->right); break;
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
    default: ;
    }
    ss << ')';
}

inline void jit_compile_visitor::left_constraint_expr(const constraint* con, std::stringstream& ss, std::unordered_map<const expr*, std::string>& compile_map)
{
    if(con->uses_variable)
    {
        ss << "(state->variables.store[" << con->variable_id << "].value)";
    }
    else if(con->value->operand == expr::compiled_ee)
    {
        ss << compile_map[con->value];
    }
    else
    {
        compile_expr(ss, con->value, false);
    }
}

inline void jit_compile_visitor::right_constraint_expr(const constraint* con, std::stringstream& ss, std::unordered_map<const expr*, std::string>& compile_map)
{
    if(con->expression->operand == expr::compiled_ee)
    {
        ss << compile_map[con->expression];
    }
    else
    {
        compile_expr(ss, con->expression);
    }
}

void jit_compile_visitor::compile_con(std::stringstream& ss,
    std::unordered_map<const expr*, std::string>& compile_map,
    const constraint* con)
{
    switch (con->operand)
    {
    case constraint::less_equal_c: left_constraint_expr(con, ss, compile_map); ss << "<="; right_constraint_expr(con, ss, compile_map); break;
    case constraint::less_c: left_constraint_expr(con, ss, compile_map); ss << "<"; right_constraint_expr(con, ss, compile_map); break;
    case constraint::greater_equal_c: left_constraint_expr(con, ss, compile_map); ss << ">="; right_constraint_expr(con, ss, compile_map); break;
    case constraint::greater_c: left_constraint_expr(con, ss, compile_map); ss << ">"; right_constraint_expr(con, ss, compile_map); break;
    case constraint::equal_c: ss << "abs("; left_constraint_expr(con, ss, compile_map); ss << "-"; right_constraint_expr(con, ss, compile_map); ss << ") <= DBL_EPSILON"; break;
    case constraint::not_equal_c: ss << "abs("; left_constraint_expr(con, ss, compile_map); ss << "-"; right_constraint_expr(con, ss, compile_map); ss << ") > DBL_EPSILON"; break;
    case constraint::compiled_c: throw std::runtime_error("cannot compile already compiled constraint.");
    default: throw std::out_of_range("constraint operand not recognized");
    }
    
    // if(con->uses_variable)
    // {
    //     ss << "(state->variables.store[" << con->variable_id << "].value)";
    // }
    // else if(con->value->operand == expr::compiled_ee)
    // {
    //     ss << compile_map[con->value];
    // }
    // else
    // {
    //     compile_expr(ss, con->value, false);
    // }
    //
    // switch(con->operand)
    // {
    // case constraint::less_equal_c: ss << "<="; break;
    // case constraint::less_c: ss << "<"; break;
    // case constraint::greater_equal_c: ss << ">="; break;
    // case constraint::greater_c: ss << ">"; break;
    // case constraint::equal_c: ss << "=="; break;
    // case constraint::not_equal_c: ss << "!="; break;
    // case constraint::compiled_c: throw std::runtime_error("cannot compile already compiled constraint.");
    // // default: throw std::out_of_range("constraint operand not recognized");
    // }
    //
    // if(con->expression->operand == expr::compiled_ee)
    // {
    //     ss << compile_map[con->expression];
    // }
    // else
    // {
    //     compile_expr(ss, con->expression);
    // }
}

bool jit_compile_visitor::compile_invariant(std::stringstream& ss,
                                              const arr<clock_var>& clocks,
                                              const constraint* con,
                                              std::unordered_map<const expr*, std::string>& expr_cache)
{
    // if(IS_INVARIANT(con->operand) && con->uses_variable)
    if(!IS_INVARIANT(con->operand) || !con->uses_variable) return false;
    
    if(con->variable_id < 0 || clocks.size <= con->variable_id)
    {
        std::cout << "variable ID: " << con->variable_id << " | clock size: " << clocks.size << std::endl; 
        throw std::runtime_error("Could not find constraint variable while compiling constraint.");
    }

    const clock_var* var = &clocks.store[con->variable_id];
    if(var->rate == 0) return false;
    
    ss << '(';
    if(con->expression->operand == expr::compiled_ee)
    {
        ss << expr_cache[con->expression];
    }
    else
    {
        compile_expr(ss, con->expression);
    }


    switch(con->operand)
    {
    case constraint::less_equal_c:
    case constraint::less_c: ss << '-'; break;
    case constraint::greater_equal_c: 
    case constraint::greater_c:
    case constraint::equal_c: 
    case constraint::not_equal_c: throw std::runtime_error("constraint operand is not invariant.");
    case constraint::compiled_c: throw std::runtime_error("cannot compile already compiled constraint.");
    // default: throw std::out_of_range("constraint operand not recognized");
    }
    ss << "(state->variables.store[" << con->variable_id << "].value)";

    ss << ')';
    ss << "/ (" << var->rate << ')';

    return true;
}

