#pragma once

#ifndef VISITOR_H
#define VISITOR_H


class constraint_t;
class edge_t;
class node_t;
class stochastic_model_t;
class clock_variable;
class update_t;
class expression;

class visitor
{
public:
    virtual ~visitor() = default;
    virtual void visit(constraint_t* constraint) = 0;
    virtual void visit(edge_t* edge) = 0;
    virtual void visit(node_t* node) = 0;
    virtual void visit(stochastic_model_t* model) = 0;
    virtual void visit(clock_variable* timer) = 0;
    virtual void visit(update_t* update) = 0;
    virtual void visit(expression* expression) = 0;
};

#endif
