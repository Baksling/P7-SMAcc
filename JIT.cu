kernal.cu
#include <cmath>
#include <limits.h>
#include <mutex>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#ifndef MACRO_H
#define MACRO_H



//HACK TO MAKE CPU WORK!
//HACK SLUT

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

#define GPU __device__ 
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__

#define DBL_MAX 1.7976931348623158e+308 //max 64 bit double value
// #define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define DBL_EPSILON (0.00001)

//While loop done to enfore ; after macro call. See: 
//https://stackoverflow.com/a/61363791/17430854
#define CUDA_CHECK(x)             \
do{                          \
if ((x) != cudaSuccess) {    \
    throw std::runtime_error(std::string("cuda error ") + std::to_string(x) + " in file '" + __FILE__ + "' on line "+  std::to_string(__LINE__)); \
}                             \
}while(0)


__host__ __device__ __forceinline__ void cuda_syncthreads_()
{
    #ifdef __CUDACC__
    #define cuda_SYNCTHREADS() __syncthreads()
    #else
    #define cuda_SYNCTHREADS()
    #endif
} 

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

#endif
#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#define SHARED_MEMORY_PR_THREAD 32

struct io_paths;
struct output_properties;

struct sim_config
{
    //simulation setup
    unsigned int blocks = 1;
    unsigned int threads = 1;
    unsigned int cpu_threads = 1;
    unsigned int simulation_amount = 1;
    unsigned int simulation_repetitions = 1;
    unsigned long long seed = 1;
    int write_mode = 0;
    bool use_max_steps = true;
    unsigned int max_steps_pr_sim = 1;
    double max_global_progression = 1;
    bool verbose = false;
    enum pretty_print
    {
        no_print = 0,
        print_model = 1,
        print_reduction = 2
    } model_print_mode = no_print;
    
    enum device_opt
    {
        device,
        host,
        both
    } sim_location = device;
    
    //model parameters (setup using function)
    bool use_shared_memory = false;
    bool use_jit = false;
    unsigned max_expression_depth = 1;
    unsigned max_edge_fanout = 0;
    unsigned tracked_variable_count = 1;
    unsigned variable_count = 1;
    unsigned network_size = 1;
    unsigned node_count = 0;
    unsigned initial_urgent = 0;
    unsigned initial_committed = 0;
    
    //paths
    io_paths* paths;

    output_properties* properties;
    double alpha = 0.005;
    double epsilon = 0.005;
    
    //pointers
    void* cache = nullptr;
    curandState* random_state_arr = nullptr;
    
    size_t total_simulations() const
    {
        return static_cast<size_t>(blocks) * threads * simulation_amount;
    }

    bool can_use_cuda_shared_memory(const size_t model_size) const
    {
        return (static_cast<size_t>(this->threads) * SHARED_MEMORY_PR_THREAD) > (model_size);
    }
};

#endif
template<typename T>
class my_stack
{
private:
    T* store_;
    int size_;
    int count_;
public:
    CPU GPU explicit my_stack(T* store, int size)
    {
        this->store_ = store;
        this->size_ = size;
        this->count_ = 0;
    }

    CPU GPU void push(T& t)
    {
        store_[count_++] = t;
    }

    CPU GPU T pop()
    {
        // if(this->count_ <= 0) printf("stack is empty, cannot pop! PANIC!");
        return this->store_[--this->count_];
    }

    CPU GPU T peak()
    {
        // if(this->count_ <= 0) printf("stack is empty, cannot peak! PANIC!");
        return this->store_[this->count_ - 1];
    }

    CPU GPU int count() const
    {
        return this->count_;
    }

    CPU GPU void clear()
    {
        this->count_ = 0;
    }
};
    #ifndef DOMAIN_H
#define DOMAIN_H

struct state;
struct edge;
struct node;


#define HAS_HIT_MAX_STEPS(x) ((x) < 0)

template<typename T>
struct arr
{
    T* store;
    int size;

    static arr<T> empty(){ return arr<T>{nullptr, 0}; }
};

#define IS_LEAF(x) ((x) < 2)
struct expr  // NOLINT(cppcoreguidelines-pro-type-member-init)
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
        modulo_ee,

        //boolean types
        less_equal_ee,
        greater_equal_ee,
        less_ee,
        greater_ee,
        equal_ee,
        not_equal_ee,
        not_ee,

        //conditional types
        conditional_ee,
        compiled_ee
        
    } operand = literal_ee;
    
    expr* left = nullptr;
    expr* right = nullptr;

    union
    {
        double value = 1.0;
        int variable_id;
        expr* conditional_else;
        int compile_id;
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
        not_equal_c = 5,
        compiled_c
    } operand;

    bool uses_variable;
    union //left hand side
    {
        expr* value;
        int variable_id;
        int compile_id;
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

    CPU GPU void add_time(const double time);
    CPU GPU void set_value(const double val);
};


#define IS_URGENT(x) ((x) > 2)
struct node
{
    int id{};
    enum node_types
    {
        location = 0,
        goal = 1,
        branch = 2,
        urgent = 3,
        committed = 4,
    } type = location;
    expr* lamda{};
    arr<edge> edges = arr<edge>::empty();
    arr<constraint> invariants = arr<constraint>::empty();
    CPU GPU double max_progression(state* state, bool* is_finite) const;
};

struct update
{
    int variable_id;
    expr* expression;
    CPU GPU void apply_update(state* state) const;
};


#define TAU_CHANNEL 0
#define IS_TAU(x) ((x) == 0)
#define IS_LISTENER(x) ((x) < 0)
#define CAN_SYNC(brod, list) ((brod) == (-(list)))
#define IS_BROADCASTER(x) ((x) > 0)


struct edge
{
    int channel{};
    expr* weight{};
    node* dest{};
    arr<constraint> guards = arr<constraint>::empty();
    arr<update> updates = arr<update>::empty();
    CPU GPU void apply_updates(state* state) const;
    CPU GPU bool edge_enabled(state* state) const;
};

struct network
{
    arr<node*> automatas;
    arr<clock_var> variables;
};



struct state
{
    unsigned urgent_count;
    unsigned committed_count;
    unsigned simulation_id;
    unsigned steps;
    double global_time;

    arr<node*> models;
    arr<clock_var> variables;

    struct w_edge
    {
        edge* e;
        double w;
    };
    
    curandState* random;
    my_stack<expr*> expr_stack;
    my_stack<double> value_stack;
    my_stack<w_edge> edge_stack;

    CPU GPU void traverse_edge(int process_id, node* dest);
    CPU GPU void broadcast_channel(const int channel, const int process);
    CPU GPU static state init(void* cache, curandState* random, const network* model, const unsigned expr_depth, const unsigned fanout);
    CPU GPU void reset(const unsigned sim_id, const network* model, int initial_urgent_count, int initial_committed_count);
};
#endif


//Please do not change the argument names, they are required for JIT compilation 
CPU GPU double evaluate_compiled_expression(const expr* ex, state* state)
{
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    switch(ex->compile_id){

}

    
    return 0.0;
}

//Please do not change the argument names, they are required for JIT compilation 
CPU GPU bool evaluate_compiled_constraint(const constraint* con, state* state)
{
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    switch(con->compile_id){
case 0: return abs((state->variables.store[13].value)-(0)) <= DBL_EPSILON; break;
case 1: return abs((state->variables.store[13].value)-(1)) <= DBL_EPSILON; break;
case 2: return (state->variables.store[1].value)<=(167); break;
case 3: return (state->variables.store[1].value)<=(85); break;
case 4: return abs((state->variables.store[13].value)-(5)) <= DBL_EPSILON; break;
case 5: return (state->variables.store[1].value)<=(159); break;
case 6: return (state->variables.store[1].value)>=(159); break;
case 7: return abs((state->variables.store[13].value)-(6)) <= DBL_EPSILON; break;
case 8: return abs((state->variables.store[13].value)-(8)) <= DBL_EPSILON; break;
case 9: return (state->variables.store[1].value)<(159); break;
case 10: return abs((state->variables.store[13].value)-(3)) <= DBL_EPSILON; break;
case 11: return abs((state->variables.store[13].value)-(7)) <= DBL_EPSILON; break;
case 12: return abs((state->variables.store[13].value)-(4)) <= DBL_EPSILON; break;
case 13: return (state->variables.store[1].value)>=(76); break;
case 14: return abs((state->variables.store[13].value)-(2)) <= DBL_EPSILON; break;
case 15: return abs((state->variables.store[14].value)-(0)) <= DBL_EPSILON; break;
case 16: return (state->variables.store[5].value)<=(30); break;
case 17: return (state->variables.store[3].value)<=(30); break;
case 18: return abs((state->variables.store[14].value)-(5)) <= DBL_EPSILON; break;
case 19: return abs((state->variables.store[7].value)-(1)) <= DBL_EPSILON; break;
case 20: return (state->variables.store[5].value)>=(5); break;
case 21: return abs((state->variables.store[14].value)-(9)) <= DBL_EPSILON; break;
case 22: return (state->variables.store[3].value)>=(5); break;
case 23: return abs((state->variables.store[14].value)-(3)) <= DBL_EPSILON; break;
case 24: return abs((state->variables.store[11].value)-(1)) <= DBL_EPSILON; break;
case 25: return abs((state->variables.store[14].value)-(7)) <= DBL_EPSILON; break;
case 26: return abs((state->variables.store[14].value)-(1)) <= DBL_EPSILON; break;
case 27: return abs((state->variables.store[9].value)-(1)) <= DBL_EPSILON; break;
case 28: return abs((state->variables.store[14].value)-(8)) <= DBL_EPSILON; break;
case 29: return abs((state->variables.store[14].value)-(2)) <= DBL_EPSILON; break;
case 30: return abs((state->variables.store[14].value)-(4)) <= DBL_EPSILON; break;
case 31: return abs((state->variables.store[14].value)-(6)) <= DBL_EPSILON; break;
case 32: return abs((state->variables.store[15].value)-(0)) <= DBL_EPSILON; break;
case 33: return abs((state->variables.store[15].value)-(1)) <= DBL_EPSILON; break;
case 34: return (state->variables.store[2].value)<=(167); break;
case 35: return (state->variables.store[2].value)<=(85); break;
case 36: return abs((state->variables.store[15].value)-(5)) <= DBL_EPSILON; break;
case 37: return (state->variables.store[2].value)<=(159); break;
case 38: return (state->variables.store[2].value)>=(159); break;
case 39: return abs((state->variables.store[15].value)-(6)) <= DBL_EPSILON; break;
case 40: return abs((state->variables.store[15].value)-(8)) <= DBL_EPSILON; break;
case 41: return (state->variables.store[2].value)<(159); break;
case 42: return abs((state->variables.store[15].value)-(3)) <= DBL_EPSILON; break;
case 43: return abs((state->variables.store[15].value)-(7)) <= DBL_EPSILON; break;
case 44: return abs((state->variables.store[15].value)-(4)) <= DBL_EPSILON; break;
case 45: return (state->variables.store[2].value)>=(76); break;
case 46: return abs((state->variables.store[15].value)-(2)) <= DBL_EPSILON; break;
case 47: return abs((state->variables.store[16].value)-(0)) <= DBL_EPSILON; break;
case 48: return (state->variables.store[6].value)<=(30); break;
case 49: return (state->variables.store[4].value)<=(30); break;
case 50: return abs((state->variables.store[16].value)-(5)) <= DBL_EPSILON; break;
case 51: return abs((state->variables.store[8].value)-(1)) <= DBL_EPSILON; break;
case 52: return (state->variables.store[6].value)>=(5); break;
case 53: return abs((state->variables.store[16].value)-(9)) <= DBL_EPSILON; break;
case 54: return (state->variables.store[4].value)>=(5); break;
case 55: return abs((state->variables.store[16].value)-(3)) <= DBL_EPSILON; break;
case 56: return abs((state->variables.store[12].value)-(1)) <= DBL_EPSILON; break;
case 57: return abs((state->variables.store[16].value)-(7)) <= DBL_EPSILON; break;
case 58: return abs((state->variables.store[16].value)-(1)) <= DBL_EPSILON; break;
case 59: return abs((state->variables.store[10].value)-(1)) <= DBL_EPSILON; break;
case 60: return abs((state->variables.store[16].value)-(8)) <= DBL_EPSILON; break;
case 61: return abs((state->variables.store[16].value)-(2)) <= DBL_EPSILON; break;
case 62: return abs((state->variables.store[16].value)-(4)) <= DBL_EPSILON; break;
case 63: return abs((state->variables.store[16].value)-(6)) <= DBL_EPSILON; break;

}

    
    return false;
}

//Please do not change the argument names, they are required for JIT compilation 
CPU GPU double evaluate_compiled_constraint_upper_bound(const constraint* con, state* state, bool* is_finite)
{
    //If this variable is marked const, then JIT compilation will not work.
    // ReSharper disable once CppLocalVariableMayBeConst
    double v0 = DBL_MAX;
    
    //DO NOT REMOVE FOLLOWING COMMENT! IT IS USED AS SEARCH TARGET FOR JIT COMPILATION!!!
    switch(con->compile_id){
case 5: v0 = ((159)-(state->variables.store[1].value))/ (1); break;
case 9: v0 = ((159)-(state->variables.store[1].value))/ (1); break;
case 37: v0 = ((159)-(state->variables.store[2].value))/ (1); break;
case 41: v0 = ((159)-(state->variables.store[2].value))/ (1); break;

}


    *is_finite = v0 < DBL_MAX; 
    return v0;
}

CPU GPU double evaluate_expression_node(const expr* expr, state* state)
{
    double v1, v2;
    switch (expr->operand) {
    case expr::literal_ee:
        return expr->value;
    case expr::clock_variable_ee:
        return state->variables.store[expr->variable_id].value;
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
    case expr::modulo_ee:
        v2 = state->value_stack.pop();
        v1 = state->value_stack.pop();
        return static_cast<double>(static_cast<int>(v1) % static_cast<int>(v2));
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
        return (abs(v1) < DBL_EPSILON);
    case expr::conditional_ee:
        v1 = state->value_stack.pop();
        state->value_stack.pop();
        return v1;
    case expr::compiled_ee: return 0.0; break;
    }
    return 0.0;
}

CPU GPU double expr::evaluate_expression(state* state)
{
    if(this->operand == literal_ee)
        return this->value;
    if(this->operand == clock_variable_ee)
        return state->variables.store[this->variable_id].value;
    if(this->operand == compiled_ee)
        return evaluate_compiled_expression(this, state);

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
        // printf("Expression evaluation ended in no values! PANIC!\n");
        return 0.0;
    }
    
    return state->value_stack.pop();
}


CPU GPU double node::max_progression(state* state, bool* is_finite) const
{
    double max_bound = DBL_MAX;

    for (int i = 0; i < this->invariants.size; ++i)
    {
        const constraint con = this->invariants.store[i];
        double limit;
        
        if(IS_INVARIANT(con.operand))
        {
            if(!con.uses_variable) continue;
            const clock_var var = state->variables.store[con.variable_id];
            if(var.rate == 0) continue;
            limit = (con.expression->evaluate_expression(state) - var.value) / var.rate;
        }
        else if(con.operand == constraint::compiled_c)
        {
            bool finite = false;
            limit = evaluate_compiled_constraint_upper_bound(&con, state, &finite);

            if(!finite) continue;
        }
        else continue;
        max_bound = fmin(max_bound,  limit); //rate is >0.
    }
    *is_finite = max_bound < DBL_MAX;
    return max_bound;
}

CPU GPU bool constraint::evaluate_constraint(state* state) const
{
    if(this->operand == compiled_c)
        return evaluate_compiled_constraint(this, state);
    const double left = this->uses_variable
        ? state->variables.store[this->variable_id].value
        : this->value->evaluate_expression(state);
    const double right = this->expression->evaluate_expression(state);

    switch (this->operand)
    {
    case less_equal_c: return left <= right;
    case less_c: return left < right;
    case greater_equal_c: return left >= right;
    case greater_c: return left > right;
    case equal_c: return abs(left - right) <= DBL_EPSILON;
    case not_equal_c: return abs(left - right) > DBL_EPSILON;
    case compiled_c: return false;
    }
    return false;
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

void clock_var::add_time(const double time)
{
    this->value += time*this->rate;
    this->max_value = fmax(this->max_value, this->value);
}

void clock_var::set_value(const double val)
{
    this->value = val;
    this->max_value = fmax(this->max_value, this->value);
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
    return true;
}

CPU GPU void state::traverse_edge(const int process_id, node* dest)
{
    const node* current = this->models.store[process_id];
    
    this->urgent_count = this->urgent_count + IS_URGENT(dest->type) - IS_URGENT(current->type);
    this->committed_count = this->committed_count + (dest->type == node::committed) - (current->type == node::committed);
    
    this->models.store[process_id] = dest;
}

void inline state::broadcast_channel(const int channel, const int process)
{
    if(!IS_BROADCASTER(channel)) return;
    
    for (int p = 0; p < this->models.size; ++p)
    {
        const node* current = this->models.store[p];
        
        if(p == process) continue;
        if(current->type == node::goal) continue;
        if(!constraint::evaluate_constraint_set(current->invariants, this)) continue;
        
        const unsigned offset = curand(this->random) % current->edges.size;
        
        for (int e = 0; e < current->edges.size; ++e)
        {
            const edge current_e = current->edges.store[(e + offset) % current->edges.size];
            // if(!IS_LISTENER(current_e.channel)) continue; <-- redundant. Already assured by CAN_SYNC
            if(!CAN_SYNC(channel, current_e.channel)) continue;

            this->traverse_edge(p, current_e.dest);

            current_e.apply_updates(this);
            break;
        }
    }
}

state state::init(void* cache, curandState* random, const network* model, const unsigned expr_depth, const unsigned fanout)
{
    node** nodes = static_cast<node**>(cache);
    cache = static_cast<void*>(&nodes[model->automatas.size]);
        
    clock_var* vars = static_cast<clock_var*>(cache);
    cache = static_cast<void*>(&vars[model->variables.size]);
        
    expr** exp = static_cast<expr**>(cache);
    cache = static_cast<void*>(&exp[expr_depth*2+1]);
        
    double* val_store = static_cast<double*>(cache);
    cache = static_cast<void*>(&val_store[expr_depth]);

    state::w_edge* fanout_store = static_cast<state::w_edge*>(cache);
    // cache = static_cast<void*>(&cache[fanout]);
    
    
    return state{
        0,
        0,
        0,
        0,
        0.0,
        arr<node*>{ nodes, model->automatas.size },
        arr<clock_var>{ vars, model->variables.size },
        random,
        my_stack<expr*>(exp, static_cast<int>(expr_depth*2+1)),
        my_stack<double>(val_store, static_cast<int>(expr_depth)),
        my_stack<state::w_edge>(fanout_store, static_cast<int>(fanout))
    };
}

void state::reset(const unsigned sim_id, const network* model, int initial_urgent_count, int initial_committed_count)
{
    this->simulation_id = sim_id;
    this->steps = 0;
    this->global_time = 0.0;
    this->urgent_count = initial_urgent_count;
    this->committed_count = initial_committed_count;
    for (int i = 0; i < model->automatas.size; ++i)
    {
        this->models.store[i] = model->automatas.store[i];
    }

    for (int i = 0; i < model->variables.size; ++i)
    {
        this->variables.store[i] = model->variables.store[i];
    }
}
#pragma once

struct model_size
{
    unsigned network_size = 0;
    unsigned nodes = 0; 
    unsigned edges = 0; 
    unsigned constraints = 0;
    unsigned updates = 0;
    unsigned variables = 0;
    unsigned expressions = 0;

    CPU GPU size_t total_memory_size() const;

    bool operator==(const model_size& rhs) const
    {
        return network_size == rhs.network_size && nodes == rhs.nodes
        && edges == rhs.edges
        && constraints == rhs.constraints
        && updates == rhs.updates
        && variables == rhs.variables
        && expressions == rhs.expressions;
    }
};

class model_oracle
{
public:
    model_oracle(void* point, const model_size& model_count)
    {
        this->initial_point = point;
        this->point = point;
        this->model_counter = model_count;
    }
    
    void* initial_point;
    void* point;
    model_size model_counter;

    template<typename T>
    CPU GPU T* get_diff(void* p1, T* p2, char* source) const;

    CPU GPU network* network_point() const;
    CPU GPU node** network_nodes_point() const;
    CPU GPU node* node_point() const;
    CPU GPU edge* edge_point() const;
    CPU GPU constraint* constraint_point() const;
    CPU GPU update* update_point() const;
    CPU GPU expr* expression_point() const;
    CPU GPU clock_var* variable_point() const;

    GPU network* move_to_shared_memory(char* shared_mem, const int threads) const;
};

template <typename T>
CPU GPU T* model_oracle::get_diff(void* p1, T* p2, char* source) const
{
    const char* c1 = static_cast<char*>(p1);
    const char* c2 = static_cast<char*>(static_cast<void*>(p2));

    return static_cast<T*>(static_cast<void*>(&source[(c2 - c1)]));
}

CPU GPU size_t model_size::total_memory_size() const
{
    return  sizeof(network)
        +   sizeof(void*) * this->network_size
        +   sizeof(node) * this->nodes
        +   sizeof(edge) * this->edges
        +   sizeof(constraint) * this->constraints
        +   sizeof(update) * this->updates
        +   sizeof(expr) * this->expressions
        +   sizeof(clock_var) * this->variables;
}

CPU GPU network* model_oracle::network_point() const
{
    return static_cast<network*>(point);
}

CPU GPU node** model_oracle::network_nodes_point() const
{
    void* p = &network_point()[1];
    return static_cast<node**>(p);
}

CPU GPU node* model_oracle::node_point() const
{
    void* p = &network_nodes_point()[model_counter.network_size];
    return static_cast<node*>(p);
}

CPU GPU edge* model_oracle::edge_point() const
{
    void* p = &node_point()[model_counter.nodes];
    return static_cast<edge*>(p);
}

CPU GPU constraint* model_oracle::constraint_point() const
{
    void* p = &edge_point()[model_counter.edges];
    return static_cast<constraint*>(p);
}

CPU GPU update* model_oracle::update_point() const
{
    void* p = &constraint_point()[model_counter.constraints];
    return static_cast<update*>(p);
}

CPU GPU expr* model_oracle::expression_point() const
{
    void* p = &update_point()[model_counter.updates];
    return static_cast<expr*>(p);
}

CPU GPU clock_var* model_oracle::variable_point() const
{
    void* p = &expression_point()[model_counter.expressions];
    return static_cast<clock_var*>(p);
}


GPU network* model_oracle::move_to_shared_memory(char* shared_mem, const int threads) const
{
    size_t* wide_shared_memory = static_cast<size_t*>(static_cast<void*>(shared_mem));
    const size_t size = this->model_counter.total_memory_size() / sizeof(size_t);

    for (size_t i = 0; i < size; i += threads)
    {
        const size_t idx = i + threadIdx.x;
        if(!(idx < size)) continue;
        wide_shared_memory[idx] = static_cast<size_t*>(this->point)[idx];
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.nodes; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(idx >= static_cast<int>(this->model_counter.nodes)) continue;

        node* n = get_diff<node>(this->point, &this->node_point()[idx], shared_mem);

        n->edges.store = get_diff<edge>(this->initial_point, n->edges.store, shared_mem);
        n->invariants.store = get_diff<constraint>(this->initial_point, n->invariants.store, shared_mem);
        
        n->lamda = get_diff(this->initial_point, n->lamda, shared_mem);
    }
    cuda_SYNCTHREADS();


    for (unsigned i = 0; i < this->model_counter.edges; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.edges))) continue;
        
        edge* e = get_diff<edge>(this->point, &this->edge_point()[idx], shared_mem);

        e->dest = get_diff<node>(this->initial_point, e->dest, shared_mem);
        e->guards.store  = get_diff<constraint>(this->initial_point, e->guards.store, shared_mem);
        e->updates.store = get_diff<update>(this->initial_point, e->updates.store, shared_mem);
        e->weight = get_diff<expr>(this->initial_point, e->weight, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.constraints; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.constraints))) continue;

        constraint* con = get_diff<constraint>(this->point, &this->constraint_point()[idx], shared_mem);

        con->expression = get_diff<expr>(this->initial_point, con->expression, shared_mem);
        if(!con->uses_variable && con->operand != constraint::compiled_c)
            con->value = get_diff<expr>(this->initial_point, con->value, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.updates; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.updates))) continue;

        update* u = get_diff<update>(this->point, &this->update_point()[idx], shared_mem);
        
        u->expression = get_diff(this->initial_point, u->expression, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.expressions; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.expressions))) continue;

        expr* ex = get_diff<expr>(this->point, &this->expression_point()[idx], shared_mem);

        if(ex->left != nullptr)
            ex->left = get_diff<expr>(this->initial_point, ex->left, shared_mem);
        if(ex->right != nullptr)
            ex->right = get_diff<expr>(this->initial_point, ex->right, shared_mem);
        if(ex->operand == ex->conditional_ee && ex->conditional_else != nullptr)
            ex->conditional_else = get_diff<expr>(this->initial_point, ex->conditional_else, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.network_size; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(idx >= static_cast<int>(this->model_counter.network_size)) continue;

        node** nn = get_diff<node*>(this->point, this->network_nodes_point(), shared_mem);
        nn[idx] = get_diff<node>(this->initial_point, nn[idx], shared_mem);
    }
    cuda_SYNCTHREADS();
    
    if(threadIdx.x == 0)
    {
        network* n = get_diff(this->point, this->network_point(), shared_mem);

        n->automatas.store = get_diff(this->point, this->network_nodes_point(), shared_mem);
        n->variables.store = get_diff(this->point, this->variable_point(), shared_mem);
    }
    cuda_SYNCTHREADS();

    return static_cast<network*>(static_cast<void*>(shared_mem));
}
#pragma once



struct node_results
{
    unsigned reached;
    unsigned total_steps;
    double total_time;

    double avg_steps() const
    {
        if(reached == 0) return 0.0;
        return static_cast<double>(total_steps) / static_cast<double>(reached);
    }

    double avg_time() const
    {
        if(reached == 0) return 0.0;
        return total_time / static_cast<double>(reached);

    }
};

struct variable_result
{
    double total_values;
    double max_value;

    double avg_max_value(const unsigned total_simulations) const
    {
        if(total_simulations == 0) return 0.0;
        return total_values / static_cast<double>(total_simulations);
    }    
};

struct result_pointers
{
private:
    const bool owns_pointers_;
    void* source_p_;
public:
    explicit result_pointers(const bool owns_pointers,
        void* free_p,
        node_results* nodes,
        variable_result* variables,
        int threads,
        unsigned total_simulations);

    node_results* nodes = nullptr;
    variable_result* variables = nullptr;
    int simulations_per_thread = 0;
    int threads = 0;
    unsigned total_simulations = 0;

    unsigned sim_per_thread() const
    {
        return static_cast<unsigned>(ceilf(static_cast<float>(this->total_simulations) / static_cast<float>(threads)));
    }
    
    void free_internals() const;
};

class memory_allocator;


class result_store
{
    friend struct state;
private:
    bool is_cuda_;
    
    unsigned simulations_;
    unsigned node_count_;
    unsigned variables_count_;
    int n_parallelism;
    
    
    node_results* node_p_ = nullptr;
    variable_result* variable_p_ = nullptr;
    
    
    size_t total_data_size() const;
    
public:
    explicit result_store(
        unsigned total_sim,
        unsigned variables,
        unsigned node_count,
        int n_parallelism,
        memory_allocator* helper);

    result_pointers load_results() const;
    void clear() const;

    //This must be in .h for RDC=false to be used.
    CPU GPU void write_output(const unsigned idx,  const state* sim) const
    {
        const int offset = static_cast<int>(this->node_count_ * idx);
        for (int i = 0; i < sim->models.size; ++i)
        {
            const int index = offset + sim->models.store[i]->id - 1;
            this->node_p_[index].reached++;
            this->node_p_[index].total_steps += sim->steps;
            this->node_p_[index].total_time += sim->global_time;
        }

        const int var_offset = static_cast<int>(this->variables_count_ * idx);
        for (int i = 0, j = 0; i < sim->variables.size; ++i)
        {
            if(!sim->variables.store[i].should_track) continue;
            const int index = var_offset + j++;
            this->variable_p_[index].total_values += sim->variables.store[i].max_value;
            this->variable_p_[index].max_value = fmax(
                this->variable_p_[index].max_value,
                sim->variables.store[i].max_value);
        }
    }
};


CPU GPU size_t thread_heap_size(const sim_config* config)
{
    const size_t size =
          static_cast<size_t>(config->max_expression_depth*2+1) * sizeof(void*) + //this is a expression*, but it doesnt like sizeof(expression*)
          config->max_expression_depth * sizeof(double) +
          config->network_size * sizeof(node) +
          config->variable_count * sizeof(clock_var) +
          config->max_edge_fanout * sizeof(state::w_edge);

    const unsigned long long int padding = (8 - (size % 8));

    return padding < 8 ? size + padding : size;
}  

CPU GPU double determine_progress(const node* node, state* state)
{
    bool is_finite = true;
    const double random_val = curand_uniform_double(state->random);
    const double max = node->max_progression(state, &is_finite);
    const double lambda = node->lamda->evaluate_expression(state);

    if(is_finite)
    {
        return (1.0 - random_val) * max;
    }
    else
    {
        // return lambda > 0.0 ? -log(random_val) / (lambda) : lambda;
        return (-log(random_val)) / (lambda - (lambda == 0.0));
    }
}

CPU GPU inline bool can_progress(const node* n)
{
    //#No brackets gang!
    for (int i = 0; i < n->edges.size; ++i)
        if(!IS_LISTENER(n->edges.store[i].channel))
            return true;
    return false;
} 

#define NO_PROCESS (-1)
#define IS_NO_PROCESS(x) ((x) < 0)
CPU GPU int progress_sim(state* sim_state, const sim_config* config)
{
    //determine if sim is done

    // if(config->use_max_steps * sim_state->steps  >= config->max_steps_pr_sim
    //     + !config->use_max_steps * sim_state->global_time >= config->max_global_progression)
    
    if((config->use_max_steps && sim_state->steps  >= config->max_steps_pr_sim)
        || (!config->use_max_steps && sim_state->global_time >= config->max_global_progression) )
            return NO_PROCESS;

    //progress number of steps
    sim_state->steps++;

    // const double max_progression_time = config->use_max_steps
    //                                         ? DBL_MAX
    //                                         : config->max_global_progression - sim_state->global_time;

    const double max_progression_time = ((config->use_max_steps) * DBL_MAX)
                + ((!config->use_max_steps) * (config->max_global_progression - sim_state->global_time));

    double min_progression_time = max_progression_time;
    int winning_process = NO_PROCESS;
    // node** winning_model = nullptr;
    for (int i = 0; i < sim_state->models.size; ++i)
    {
        const node* current = sim_state->models.store[i];
        
        //if goal is reached, dont bother
        if(current->type == node::goal) continue;
        
        //If all channels that are left is listeners, then dont bother
        //This also ensures that current_node has edges
        if(!can_progress(current)) continue;
        
        //if it is not in a valid state, then it is disabled 
        if(!constraint::evaluate_constraint_set(current->invariants, sim_state)) continue;

        
        //determine current models progress
        const double local_progress = determine_progress(current, sim_state);

        // printf("progress %lf\n", local_progress);
        //If negative progression, skip. Represents NO_PROGRESS
        //Set current as winner, if it is the earliest active model.
        if(
            local_progress >= 0.0
            && local_progress < min_progression_time
            && (sim_state->committed_count == 0
                || (sim_state->committed_count > 0
                    && current->type == node::committed)))
        {
            min_progression_time = local_progress;
            winning_process = i;
            // winning_model = &sim_state->models.store[i];
        }
    }
    // printf(" I WON! Node: %d \n", winning_model->current_node->get_id());
    if(min_progression_time < max_progression_time && sim_state->urgent_count == 0)
    {
        for (int i = 0; i < sim_state->variables.size; ++i)
        {
            sim_state->variables.store[i].add_time(min_progression_time);
        }
        sim_state->global_time += min_progression_time;
    }

    return winning_process;
}

CPU GPU edge* pick_next_edge_stack(const arr<edge>& edges, state* state)
{
    state->edge_stack.clear();
    int valid_count = 0;
    state::w_edge valid_edge = {nullptr, 0.0};
    double weight_sum = 0.0;
    
    for (int i = 0; i < edges.size; ++i)
    {
        edge* e = &edges.store[i];
        if(IS_LISTENER(e->channel)) continue;
        if(!constraint::evaluate_constraint_set(e->guards, state)) continue;
        
        const double weight = e->weight->evaluate_expression(state);
        //only consider edge if it its weight is positive.
        //Negative edge value is semantically equivalent to disabled.
        if(weight <= 0.0) continue;
        valid_edge = state::w_edge{ e, weight };
        valid_count++;
        weight_sum += weight;
        state->edge_stack.push(valid_edge);
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1) return valid_edge.e;

    const double r_val = (1.0 - curand_uniform_double(state->random)) * weight_sum;
    double r_acc = 0.0;

    //pick the weighted random value.
    valid_edge = { nullptr, 0.0 }; //reset valid edge !IMPORTANT
    for (int i = 0; i < valid_count; ++i)
    {
        valid_edge = state->edge_stack.pop();
        r_acc += valid_edge.w;
        if(r_val < r_acc) break;
    }

    return valid_edge.e;
}

CPU GPU void simulate_automata(
    const unsigned idx,
    const network* model,
    const result_store* output,
    const sim_config* config)
{
    void* cache = static_cast<void*>(&static_cast<char*>(config->cache)[(idx*thread_heap_size(config)) / sizeof(char)]);
    curandState* r_state = &config->random_state_arr[idx];
    curand_init(config->seed, idx, idx, r_state);
    state sim_state = state::init(cache, r_state, model, config->max_expression_depth, config->max_edge_fanout);
    
    for (unsigned i = 0; i < config->simulation_amount; ++i)
    {
        const unsigned int sim_id = i + config->simulation_amount * static_cast<unsigned int>(idx);
        sim_state.reset(sim_id, model, config->initial_urgent, config->initial_committed);
        
        //run simulation
        while (true)
        {
            const int process = progress_sim(&sim_state, config);
            // printf("current process_id: %d %lf | urgent_count: %d\n", process, sim_state.global_time, sim_state.urgent_count);
            if(IS_NO_PROCESS(process)) break;
            
            do
            {
                const node* current = sim_state.models.store[process];
                const edge* e = pick_next_edge_stack(current->edges, &sim_state);
                // printf("current: %d, %d, %p\n", current->id, current->type, e);
                if(e == nullptr) break;
                
                sim_state.traverse_edge(process, e->dest);
                e->apply_updates(&sim_state);
                sim_state.broadcast_channel(e->channel, process);
                // printf("dest: %d, %d\n", sim_state.models.store[process]->id, sim_state.models.store[process]->type);
            } while (sim_state.models.store[process]->type == node::branch);
        }
        output->write_output(idx, &sim_state);
    }
}

__global__ void simulator_gpu_kernel(
    const model_oracle* oracle,
    const result_store* output,
    const sim_config* config)
{
    // ReSharper disable once CppTooWideScope
    extern __shared__ char shared_mem[];
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    network* model;
    if(config->use_shared_memory)
    {
        model = oracle->move_to_shared_memory(shared_mem, static_cast<int>(config->threads));
    }
    else
    {
        model = oracle->network_point();
    }
    cuda_SYNCTHREADS();

    simulate_automata(idx, model, output, config);
}