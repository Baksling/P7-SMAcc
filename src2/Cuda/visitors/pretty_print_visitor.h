#pragma once

#include "visitor.h"
#include <ostream>
#include <string>
#include <unordered_map>

class pretty_print_visitor : public visitor
{
private:
     std::ostream* stream_;
     unsigned scope_ = 0;
     std::unordered_map<int, std::string>* var_names_;
     std::unordered_map<int, std::string>* node_names_;
     void print_indent() const
     {
          for (unsigned j = 0; j < this->scope_; ++j)
               *this->stream_ << "  ";
     }
     static std::string constraint_type_to_string(const constraint* c);
     static std::string node_type_to_string(const node* n);
     static std::string expr_type_to_string(const expr* ex);
     static std::string pretty_pn_expr(const expr* ex);
     static std::string pretty_expr(const expr* ex);
public:
     explicit pretty_print_visitor(std::ostream* stream,
          std::unordered_map<int, std::string>* node_names,
          std::unordered_map<int, std::string>* variable_names);

     void visit(network* a) override;
     void visit(node* n) override;
     void visit(edge* e) override;
     void visit(constraint* c) override;
     void visit(clock_var* cv) override ;
     void visit(update* u) override;
     void visit(expr* u) override;
     std::string print_node_name(int id) const;
     std::string print_var_name(int id) const;
};
