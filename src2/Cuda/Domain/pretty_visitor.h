#pragma once

#ifndef PRETTY_VISITOR_H
#define PRETTY_VISITOR_H

#include <set>
#include "common.h"



class pretty_visitor : public visitor
{
private:
    std::set<node_t*> checker = {};
public:
    virtual void visit(constraint_t* constraint) override;

    virtual void visit(edge_t* edge) override;

    virtual void visit(node_t* node) override;

    virtual void visit(stochastic_model_t* model) override;

    virtual void visit(clock_timer_t* timer) override;

    virtual void visit(update_t* update) override;

    void pretty_helper();
};

#endif
