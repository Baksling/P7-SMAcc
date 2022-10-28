#pragma once

#ifndef PRETTY_VISITOR_H
#define PRETTY_VISITOR_H

#include <unordered_set>
#include "visitor.h"


class pretty_visitor final : public visitor
{
private:
    std::unordered_set<int> checker_ = {};
    int* check_ = nullptr;
    int scope_ = 0;
    void indentation() const;
public:
    explicit pretty_visitor();
    
    virtual void visit(constraint_t* constraint) override;

    virtual void visit(edge_t* edge) override;

    virtual void visit(node_t* node) override;

    virtual void visit(stochastic_model_t* model) override;

    virtual void visit(clock_variable* timer) override;

    virtual void visit(update_t* update) override;

    virtual void visit(expression* expression) override;
    
    void pretty_helper();
};

#endif
