#pragma once

#include "visitor.h"
#include "ostream"

class pretty_print_visitor : public visitor
{
private:
     std::ostream* stream_;
     unsigned scope_ = 0;
     void print_indent() const
     {
          for (unsigned j = 0; j < this->scope_; ++j)
               *this->stream_ << "  ";
     }
     static std::string constraint_type_to_string(const constraint* c);
     static std::string expr_type_to_string(const expr* ex);
     static std::string pretty_expr(const expr* ex);
public:
     explicit pretty_print_visitor(std::ostream* stream);

     void visit(automata* a) override;
     void visit(node* n) override;
     void visit(edge* e) override;
     void visit(constraint* c) override;
     void visit(clock_var* cv) override ;
     void visit(update* u) override;
     void visit(expr* u) override;
};
