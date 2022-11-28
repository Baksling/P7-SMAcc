﻿#ifndef DOMAIN_H
#define DOMAIN_H

struct state;
struct edge;
struct node;

#include "macro.cu"
#include "my_stack.cu"

template<typename T>
struct arr
{
    T* store;
    int size;

    static arr<T> empty(){ return arr<T>{nullptr, 0}; }
};

#define IS_LEAF(x) ((x) <= 2)
struct expr
{
    enum operators
    {
        //value types
        literal_ee = 0,
        clock_variable_ee = 1,

        //random
        random_ee,

        //arithmatic types
        plus_ee,
        minus_ee,
        multiply_ee,
        division_ee,
        power_ee,
        negation_ee,
        sqrt_ee,

        //boolean types
        less_equal_ee,
        greater_equal_ee,
        less_ee,
        greater_ee,
        equal_ee,
        not_equal_ee,
        not_ee,

        //conditional types
        conditional_ee
    } operand;

    expr* left;
    expr* right;

    union
    {
        expr* conditional_else;
        double value;
        int variable_id;
    };

    CPU GPU double evaluate_expression(state* state);
};


/**
 * \brief Takes in constraint::operators and returns bool whether the operand is a constraint
 * \param a constraint::operators
 */
#define IS_INVARIANT(a) ((a) < 2)
struct constraint
{
    enum operators
    {
        less_equal_c = 0,
        less_c = 1,
        greater_equal_c = 2,
        greater_c = 3,
        equal_c = 4,
        not_equal_c = 5
    } operand;

    bool uses_variable;
    union //left hand side
    {
        expr* value;
        int variable_id;
    };
    expr* expression; //right hand side
    CPU GPU bool evaluate_constraint(state* state) const;
    CPU GPU static bool evaluate_constraint_set(const arr<constraint>& con_arr, state* state);
};

struct clock_var
{
    int id;
    bool should_track;
    unsigned rate;
    double value;
    double max_value;
    double temp_value;

    CPU GPU void add_time(const double time)
    {
        this->value += time*this->rate;
        this->temp_value = this->value;
        this->max_value = fmax(this->max_value, this->value);
    }
    CPU GPU void set_value(const double val)
    {
        this->value = val;
        this->temp_value = val;
        this->max_value = fmax(this->max_value, this->value);
    }
    CPU GPU void reset_temp()
    {
        this->temp_value = this->value;
    }
};


struct node
{
    int id;
    expr* lamda;
    arr<edge> edges;
    arr<constraint> invariants;
    bool is_branch_point;
    bool is_goal;
    CPU GPU double max_progression(state* state, bool* is_finite) const;
};

struct update
{
    int variable_id;
    expr* expression;
    CPU GPU void apply_temp_update(state* state) const;
    CPU GPU void apply_update(state* state) const;
};


#define TAU_CHANNEL 0
#define IS_TAU(x) ((x) == 0)
#define IS_LISTENER(x) ((x) < 0)
#define CAN_SYNC(brod, list) ((brod) == (-(list)))
#define IS_BROADCASTER(x) ((x) > 0)


struct edge
{
    int channel;
    expr* weight;
    node* dest;
    arr<constraint> guards;
    arr<update> updates;
    CPU GPU void apply_updates(state* state) const;
    CPU GPU bool edge_enabled(state* state) const;
};

struct automata
{
    arr<node*> network;
    arr<clock_var> variables;
};

struct state
{
    unsigned simulation_id;
    unsigned steps;
    double global_time;

    arr<node*> models;
    arr<clock_var> variables;
    
    curandState* random;
    my_stack<expr*> expr_stack;
    my_stack<double> value_stack;

    CPU GPU void broadcast_channel(int channel, const node* source);

    CPU GPU static state init(void* cache, curandState* random,  const automata* model, const unsigned expr_depth)
    {
        node** nodes = static_cast<node**>(cache);
        cache = static_cast<void*>(&nodes[model->network.size]);
        
        clock_var* vars = static_cast<clock_var*>(cache);
        cache = static_cast<void*>(&vars[model->variables.size]);
        
        expr** exp = static_cast<expr**>(cache);
        cache = static_cast<void*>(&exp[expr_depth*2+1]);
        
        double* val_store = static_cast<double*>(cache);
        cache = static_cast<void*>(&val_store[expr_depth]);
        
        return state{
            0,
            0,
            0.0,
            arr<node*>{ nodes, model->network.size },
            arr<clock_var>{ vars, model->variables.size },
            random,
            my_stack<expr*>(exp, static_cast<int>(expr_depth*2+1)),
            my_stack<double>(val_store, static_cast<int>(expr_depth))
        };
    }

    CPU GPU void reset(const unsigned sim_id, const automata* model)
    {
        this->simulation_id = sim_id;
        this->steps = 0;
        this->global_time = 0.0;
        
        for (int i = 0; i < model->network.size; ++i)
        {
            this->models.store[i] = model->network.store[i];
        }

        for (int i = 0; i < model->variables.size; ++i)
        {
            this->variables.store[i] = model->variables.store[i];
        }
    }
};
#endif
