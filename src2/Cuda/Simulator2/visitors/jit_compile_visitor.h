#pragma once
#include <sstream>
#include <unordered_map>
#include "visitor.h"

class jit_compile_visitor : public visitor
{
private:
    arr<clock_var> clocks_;
    std::stringstream expr_store_;
    std::stringstream con_store_;
    std::stringstream invariant_store_;
    bool finalized_ = false;
    int expr_compile_id_enumerator_ = 0;
    int con_compile_id_enumerator_ = 0;

    static void compile_expr(std::stringstream& ss, const expr* e, bool is_root);
    static void compile_con(std::stringstream& ss, const constraint* con);
    static bool compile_invariant(std::stringstream& ss, const arr<clock_var>& clocks,
                                  const constraint* con);
    void init_store();
public:
    explicit jit_compile_visitor();
    void visit(network* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override;
    void visit(update* u) override;
    void visit(expr* ex) override;
    void finalize();
    
    std::stringstream& get_expr_compilation();
    std::stringstream& get_constraint_compilation();
    std::stringstream& get_invariant_compilation();

    void clear() override;
};
