#ifndef MACRO_H
#define MACRO_H


#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <string>

//HACK TO MAKE CPU WORK!
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS
//HACK SLUT

#define GPU __device__ 
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__

#define DBL_MAX          1.7976931348623158e+308 //max 64 bit double value


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
    
    enum device_opt
    {
        device,
        host,
        both
    } sim_location;
    
    //model parameters (setup using function)
    bool use_shared_memory = false;
    unsigned int max_expression_depth = 1;
    unsigned tracked_variable_count = 1;
    unsigned variable_count = 1;
    unsigned network_size = 1;

    //paths
    std::string model_path{};
    std::string out_path{};
    
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

    CPU GPU void add_time(const double time)
    {
        this->value += time*this->rate;
        this->max_value = fmax(this->max_value, this->value);
    }
    CPU GPU void set_value(const double val)
    {
        this->value = val;
        this->max_value = fmax(this->max_value, this->value);
    }
};


struct node
{
    int id{};
    expr* lamda{};
    arr<edge> edges = arr<edge>::empty();
    arr<constraint> invariants = arr<constraint>::empty();
    bool is_branch_point{};
    bool is_goal{};
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
    arr<constraint> guards;
    arr<update> updates;
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
    unsigned simulation_id;
    unsigned steps;
    double global_time;

    arr<node*> models;
    arr<clock_var> variables;
    
    curandState* random;
    my_stack<expr*> expr_stack;
    my_stack<double> value_stack;

    CPU GPU void broadcast_channel(int channel, const node* source);

    CPU GPU static state init(void* cache, curandState* random,  const network* model, const unsigned expr_depth);

    CPU GPU void reset(const unsigned sim_id, const network* model);
};
#endif

#define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0

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
    }
    return 0.0;
}

CPU GPU double expr::evaluate_expression(state* state)
{
    if(this->operand == literal_ee)
    {
        return this->value;
    }
    if(this->operand == clock_variable_ee)
    {
        return state->variables.store[this->variable_id].value;
    }

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
    double max_bound = HUGE_VAL;

    for (int i = 0; i < this->invariants.size; ++i)
    {
        const constraint con = this->invariants.store[i];
        if(!IS_INVARIANT(con.operand)) continue;
        if(!con.uses_variable) continue;
        const clock_var var = state->variables.store[con.variable_id];
        if(var.rate == 0) continue;
        const double expr_value = con.expression->evaluate_expression(state);
        
        max_bound = fmin(max_bound,  (expr_value -  var.value) / var.rate); //rate is >0.
    }
    *is_finite = !isinf(max_bound);
    return max_bound;
}

CPU GPU bool constraint::evaluate_constraint(state* state) const
{
    const double left = this->uses_variable
        ? state->variables.store[this->variable_id].value
        : this->value->evaluate_expression(state);
    const double right = this->expression->evaluate_expression(state);

    switch (this->operand)
    {
    case constraint::less_equal_c: return left <= right;
    case constraint::less_c: return left < right;
    case constraint::greater_equal_c: return left >= right;
    case constraint::greater_c: return left > right;
    case constraint::equal_c: return left == right;  // NOLINT(clang-diagnostic-float-equal)
    case constraint::not_equal_c: return left != right;  // NOLINT(clang-diagnostic-float-equal)
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
            const edge current_e = current->edges.store[(j + offset) % current->edges.size];
            if(!IS_LISTENER(current_e.channel)) continue;
            if(!CAN_SYNC(channel, current_e.channel)) continue;
            
            node* dest = current_e.dest;

            this->models.store[i] = dest;

            current_e.apply_updates(this);
            break;
        }
    }
}

state state::init(void* cache, curandState* random, const network* model, const unsigned expr_depth)
{
    node** nodes = static_cast<node**>(cache);
    cache = static_cast<void*>(&nodes[model->automatas.size]);
        
    clock_var* vars = static_cast<clock_var*>(cache);
    cache = static_cast<void*>(&vars[model->variables.size]);
        
    expr** exp = static_cast<expr**>(cache);
    cache = static_cast<void*>(&exp[expr_depth*2+1]);
        
    double* val_store = static_cast<double*>(cache);
    // cache = static_cast<void*>(&val_store[expr_depth]);
        
    return state{
        0,
        0,
        0.0,
        arr<node*>{ nodes, model->automatas.size },
        arr<clock_var>{ vars, model->variables.size },
        random,
        my_stack<expr*>(exp, static_cast<int>(expr_depth*2+1)),
        my_stack<double>(val_store, static_cast<int>(expr_depth))
    };
}

void state::reset(const unsigned sim_id, const network* model)
{
    this->simulation_id = sim_id;
    this->steps = 0;
    this->global_time = 0.0;
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
    const size_t size = this->model_counter.total_memory_size() / sizeof(char);

    for (size_t i = 0; i < size; i += threads)
    {
        const size_t idx = i + threadIdx.x;
        if(!(idx < size)) continue;
        shared_mem[idx] = static_cast<char*>(this->point)[idx];
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
        if(!con->uses_variable)
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


struct sim_metadata
{
    unsigned int steps;
    double global_time;
};

struct result_pointers
{
private:
    const bool owns_pointers_;
    void* source_p_;
public:
    explicit result_pointers(const bool owns_pointers, void* free_p,
        sim_metadata* results, int* nodes, double* variables);

    sim_metadata* meta_results = nullptr;
    int* nodes = nullptr;
    double* variables = nullptr;

    void free_internals() const;
};

class result_store
{
    friend struct state;
private:
    bool is_cuda_;
    
    int simulations_;
    int models_count_;
    int variables_count_;
    
    sim_metadata* metadata_p_ = nullptr;
    double* variable_p_ = nullptr;
    int* node_p_ = nullptr;

    size_t total_data_size() const;
public:
    explicit result_store(const unsigned total_sim,
                          const unsigned variables,
                          const unsigned network_size,
                          memory_allocator* helper);

    result_pointers load_results() const;

    //This MUST be in here, in order not to cause RDC to be required.
    CPU GPU void write_output(const state* sim) const
    {
        const int sim_id = static_cast<int>(sim->simulation_id);

        this->metadata_p_[sim_id].steps = sim->steps; 
        this->metadata_p_[sim_id].global_time = sim->global_time;
        
        for (int i = 0, j = 0; i < sim->variables.size; ++i)
        {
            if(!sim->variables.store[i].should_track) continue;
            this->variable_p_[sim_id*this->variables_count_ + j++] = sim->variables.store[i].value;
        }

        for (int i = 0; i < sim->models.size; ++i)
        {
            // sim->models.store[i]->is_goal
            // ? sim->models.store[i]->id
            // : -sim->models.store[i]->id;

            // this->node_p_[sim_id*sim->models.size + i] = sim->models.store[i]->is_goal ? sim->models.store[i]->id : -sim->models.store[i]->id;
            this->node_p_[sim_id*sim->models.size + i] = sim->models.store[i]->id * ((sim->models.store[i]->is_goal*2) + (-1));
        }
    } 
};



CPU GPU size_t thread_heap_size(const sim_config* config)
{
    const size_t size =
          static_cast<size_t>(config->max_expression_depth*2+1) * sizeof(void*) + //this is a expression*, but it doesnt like sizeof(expression*)
          config->max_expression_depth * sizeof(double) +
          config->network_size * sizeof(node) +
          config->variable_count * sizeof(clock_var);

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
        // return lambda > 0 ? -log2(random_val) / (lambda) : lambda;
        return -log(random_val) / (lambda - (lambda == 0.0));
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

CPU GPU node** progress_sim(state* sim_state, const sim_config* config)
{
    //determine if sim is done

    // if(config->use_max_steps * sim_state->steps  >= config->max_steps_pr_sim
    //     + !config->use_max_steps * sim_state->global_time >= config->max_global_progression)

    if((config->use_max_steps && sim_state->steps  >= config->max_steps_pr_sim)
        || (!config->use_max_steps && sim_state->global_time >= config->max_global_progression) )
            return nullptr;

    //progress number of steps
    sim_state->steps++;

    // const double max_progression_time = config->use_max_steps
    //                                         ? DBL_MAX
    //                                         : config->max_global_progression - sim_state->global_time;

    const double max_progression_time = ((config->use_max_steps) * DBL_MAX)
                + ((!config->use_max_steps) * (config->max_global_progression - sim_state->global_time));

    double min_progression_time = max_progression_time;
    
    node** winning_model = nullptr;
    for (int i = 0; i < sim_state->models.size; ++i)
    {
        const node* current = sim_state->models.store[i];
        
        //if goal is reached, dont bother
        if(current->is_goal) continue;
        
        //If all channels that are left is listeners, then dont bother
        //This also ensures that current_node has edges
        if(!can_progress(current)) continue;
        
        //if it is not in a valid state, then it is disabled 
        if(!constraint::evaluate_constraint_set(current->invariants, sim_state)) continue;

        
        //determine current models progress
        const double local_progress = determine_progress(current, sim_state);

        //If negative progression, skip. Represents NO_PROGRESS
        //Set current as winner, if it is the earliest active model.
        if(local_progress >= 0.0 && local_progress < min_progression_time)
        {
            min_progression_time = local_progress;
            winning_model = &sim_state->models.store[i];
        }
    }
    // printf(" I WON! Node: %d \n", winning_model->current_node->get_id());
    if(min_progression_time < max_progression_time)
    {
        for (int i = 0; i < sim_state->variables.size; ++i)
        {
            sim_state->variables.store[i].add_time(min_progression_time);
        }
        sim_state->global_time += min_progression_time;
    }

    return winning_model;
}
#define BIT_IS_SET(n, i) ((n) & (1UL << (i)))
#define SET_BIT(n, i) (n) |= (1UL << (i)) 

CPU GPU edge* pick_next_edge(const arr<edge>& edges, state* state)
{
    //TODO set max nr. of outgoing edges to 64
    // const int edge_amount = umin(edges->size, sizeof(unsigned long long)*8);
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge* valid_edge = nullptr;
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
        SET_BIT(valid_edges_bitarray, i);
        valid_edge = e;
        valid_count++;
        weight_sum += weight; 
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(state->random)) * weight_sum;
    double r_acc = 0.0;

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges.size; ++i)
    {
        if(!BIT_IS_SET(valid_edges_bitarray, i)) continue;
        const double weight = edges.store[i].weight->evaluate_expression(state);
        if(weight <= 0.0) continue;
        
        valid_edge = &edges.store[i];
        r_acc += weight;
        if(r_val < r_acc) break;
    }
    return valid_edge;
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
    state sim_state = state::init(cache, r_state, model, config->max_expression_depth);
    
    for (unsigned i = 0; i < config->simulation_amount; ++i)
    {
        const unsigned int sim_id = i + config->simulation_amount * static_cast<unsigned int>(idx);
        sim_state.reset(sim_id, model);
        
        //run simulation
        while (true)
        {
            node** state = progress_sim(&sim_state, config);
            
            if(state == nullptr || (*state)->is_goal) break;
            
            do
            {
                const edge* e = pick_next_edge((*state)->edges, &sim_state);
                if(e == nullptr) break;

                *state = e->dest;
                e->apply_updates(&sim_state);
                sim_state.broadcast_channel(e->channel, *state);
            } while ((*state)->is_branch_point);
        }
        output->write_output(&sim_state);
    }
}

__global__ void simulator_gpu_kernel_oracle(
    const model_oracle* oracle,
    const result_store* output,
    const sim_config* config)
{
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

__global__ void simulator_gpu_kernel(
    const network* model,
    const result_store* output,
    const sim_config* config)
{
    const unsigned long idx = threadIdx.x + blockDim.x * blockIdx.x;
    simulate_automata(idx, model, output, config);
}