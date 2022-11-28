#include "my_stack.cu"
#include "macro.cu"

#define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0

struct state;
struct edge;
struct node;

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

    expr* left = nullptr;
    expr* right = nullptr;

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


CPU GPU double evaluate_expression_node(const expr* expr, state* state)
{
    double v1, v2;
    switch (expr->operand) {
    case expr::literal_ee:
        return expr->value;
    case expr::clock_variable_ee:
        return state->variables.store[expr->variable_id].temp_value;
    case expr::random_ee:
        v1 = state->value_stack.pop();
        return (1.0 - curand_uniform_double(state->random)) * v1;
    case expr::plus_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 + v2;
    case expr::minus_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 - v2;
    case expr::multiply_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 * v2;
    case expr::division_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 / v2;
    case expr::power_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return pow(v1, v2);
    case expr::negation_ee: 
        v1 = state->value_stack.pop();
        return -v1;
    case expr::sqrt_ee: 
        v1 = state->value_stack.pop();
        return sqrt(v1);
    case expr::less_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 <= v2;
    case expr::greater_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 >= v2;
    case expr::less_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 < v2;
    case expr::greater_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return v1 > v2;
    case expr::equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return abs(v1 - v2) <= DBL_EPSILON;
    case expr::not_equal_ee: 
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return abs(v1 - v2) > DBL_EPSILON;
    case expr::not_ee:
        v1 = state->value_stack.pop();
        return abs(v1) < DBL_EPSILON * 1.0;
    case expr::conditional_ee:
        v1 = state->value_stack.pop();
        state->value_stack.pop();
        return v1;
    default: return 0.0;
    }
}

CPU GPU double expr::evaluate_expression(state* state)
{
    state->expr_stack.clear();
    state->value_stack.clear();
    expr* current = this;
    while (true)
    {
        while(current != nullptr)
        {
            state->expr_stack.push(current);
            
            if(!IS_LEAF(current->operand)) //only push twice if it has children
                state->expr_stack.push(current);
            
            current = current->left;
        }
        if(state->expr_stack.count() == 0)
        {
            break;
        }
        current = state->expr_stack.pop();
        
        if(state->expr_stack.count() > 0 && state->expr_stack.peak() == current)
        {
            current = (current->operand == conditional_ee && abs(state->value_stack.peak()) < DBL_EPSILON)
                ? current->conditional_else
                : current->right;
        }
        else
        {
            double val = evaluate_expression_node(current, state);
            state->value_stack.push(val);
            current = nullptr;
        }
    }

    if(state->value_stack.count() == 0)
    {
        printf("Expression evaluation ended in no values! PANIC!\n");
    }
    
    return state->value_stack.pop();
}


CPU GPU double node::max_progression(state* state, bool* is_finite) const
{
    double max_bound = INFINITY;

    for (int i = 0; i < this->invariants.size; ++i)
    {
        const constraint* con = &this->invariants.store[i];
        if(!IS_INVARIANT(con->operand)) continue;
        if(!con->uses_variable) continue;
        const clock_var var = state->variables.store[con->variable_id];
        if(var.rate == 0) continue;
        const double time = state->variables.store[con->variable_id].value;
        const double expr_value = con->expression->evaluate_expression(state);
        
        max_bound = fmin(max_bound,  (expr_value - time) / var.rate); //rate is >0.
    }
    *is_finite = isfinite(max_bound);
    return max_bound;
}

CPU GPU bool constraint::evaluate_constraint(state* state) const
{
    const double left = state->variables.store[this->variable_id].value;
    const double right = this->expression->evaluate_expression(state);

    switch (this->operand)
    {
    case constraint::less_equal_c: return left <= right;
    case constraint::less_c: return left < right;
    case constraint::greater_equal_c: return left >= right;
    case constraint::greater_c: return left < right;
    case constraint::equal_c: return left == right;  // NOLINT(clang-diagnostic-float-equal)
    case constraint::not_equal_c: return left != right;  // NOLINT(clang-diagnostic-float-equal)
    default: return false;
    }
}


CPU GPU bool constraint::evaluate_constraint_set(const arr<constraint>& con_arr, state* state)
{
    for (int i = 0; i < con_arr.size; ++i)
    {
        if(!con_arr.store[i].evaluate_constraint(state))
            return false;
    }
    return true;
}

CPU GPU inline void update::apply_temp_update(state* state) const
{
    const double value = this->expression->evaluate_expression(state);
    state->variables.store[this->variable_id].temp_value = value;
}

CPU GPU inline void update::apply_update(state* state) const
{
    const double value = this->expression->evaluate_expression(state);
    state->variables.store[this->variable_id].set_value(value);
}

CPU GPU inline void edge::apply_updates(state* state) const
{
    for (int i = 0; i < this->updates.size; ++i)
    {
        this->updates.store[i].apply_update(state);
    }
}

CPU GPU inline bool edge::edge_enabled(state* state) const
{
    for (int i = 0; i < this->guards.size; ++i)
    {
        if(!this->guards.store[i].evaluate_constraint(state))
            return false;
    }

    for (int i = 0; i < this->updates.size; ++i)
    {
        this->updates.store[i].apply_temp_update(state);
    }

    bool is_valid = true;
        
    for (int i = 0; i < this->dest->invariants.size; ++i)
    {
        if(this->dest->invariants.store[i].evaluate_constraint(state))
        {
            is_valid = false;
            break;
        }
    }

    //this is always <= state.variables.size
    for (int i = 0; i < this->updates.size; ++i)
    {
        const int id = this->updates.store[i].variable_id;
        state->variables.store[id].reset_temp();
    }

    return is_valid;
}

CPU GPU void inline state::broadcast_channel(const int channel, const node* source)
{
    if(!IS_BROADCASTER(channel)) return;

    for (int i = 0; i < this->models.size; ++i)
    {
        const node* current = this->models.store[i];
        if(current->id == source->id) continue;
        if(current->is_goal) continue;
        if(!constraint::evaluate_constraint_set(current->invariants, this)) continue;
        
        const unsigned offset = curand(this->random) % current->edges.size;
        for (int j = 0; j < current->edges.size; ++j)
        {
            const edge* current_e = &current->edges.store[(i + offset) % current->edges.size];
            if(!IS_LISTENER(current_e->channel)) continue;
            if(!CAN_SYNC(channel, current_e->channel)) continue;

            node* dest = current_e->dest;
            
            this->models.store[i] = dest;

            current_e->apply_updates(this);
            break;
        }
    }
}
