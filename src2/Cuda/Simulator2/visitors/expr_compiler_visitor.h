#pragma once
#include <sstream>

#include "visitor.h"

class expr_compiler_visitor : public visitor
{
private:
    std::stringstream expr_store_;
    // std::stringstream con_store_;
    bool finalized_ = false;
    int compile_id_enumerator_ = 0;

    static void compile_expr(std::stringstream& ss, const expr* e, bool is_root);
    void init_store();
public:
    explicit expr_compiler_visitor();
    void visit(network* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override;
    void visit(update* u) override;
    void visit(expr* ex) override;

    std::stringstream& get_compiled_expressions();
    void clear();
    
};
