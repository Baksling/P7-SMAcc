#ifndef DOMAIN_H
#define DOMAIN_H

struct state;
struct edge;
struct node;

#include "macro.h"
#include "my_stack.h"

#define HAS_HIT_MAX_STEPS(x) ((x) < 0)

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
    } operand = literal_ee;

    expr* left = nullptr;
    expr* right = nullptr;

    union
    {
        double value = 1.0;
        int variable_id;
        expr* conditional_else;
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

    CPU GPU static state init(void* cache, curandState* random,  const automata* model, const unsigned expr_depth);

    CPU GPU void reset(const unsigned sim_id, const automata* model);
};
#endif
