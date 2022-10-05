#pragma once

#ifndef CUDA_VISITOR_H
#define CUDA_VISITOR_H

#include <set>
#include "visitor.h"
#include <map>


class cuda_visitor : public visitor
{
private:
    std::list<void*> pointer_lst_;
    std::set<node_t*> checker_ = {};
    std::map<void*, void* > visitor_map_;
    void* field;
public:
    void visit(constraint_t* constraint) override;
    void visit(edge_t* edge) override;
    void visit(node_t* node) override;
    void visit(stochastic_model_t* model) override;
    void visit(clock_timer_t* timer) override;
    void visit(update_t* update) override;
};

#endif
