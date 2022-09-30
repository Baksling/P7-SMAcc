#pragma once

#ifndef VISITOR_H
#define VISITOR_H

#include "../constraint_t.h"
#include "../edge_t.h"
#include "../stochastic_model_t.h"
#include "../../Projekt/cuda_map.h"

class visistor
{
public:
    virtual void visit(constraint_t* constraint) = 0;
    virtual void visit(edge_t* edge) = 0;
    virtual void visit(node_t* node) = 0;
    virtual void visit(stochastic_model_t* model) = 0;
    virtual void visit(timer_t* timer) = 0;
    virtual void visit(update_t* update) = 0;
};

#endif
