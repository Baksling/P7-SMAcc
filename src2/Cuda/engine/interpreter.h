#include <curand_kernel.h>
#include <curand_uniform.h>

#include "Domain.h"
#include "../common/my_stack.h"

struct state;


struct value
{
    value()
    {
        type = int_t;
        integer = 0;
    }
    value(const int i)
    {
        type = int_t;
        integer = i;
    }
    value(const int64_t i)
    {
        type = int_t;
        integer = i;
    }

    value(const double f)
    {
        type = fp_t;
        real = f;
    }
    
    enum primitive_types
    {
        int_t,
        fp_t
    } type = int_t;
    union
    {
        long long int integer{};
        double real{};
    };

    value operator+(const value& left, const value& right) const
    {
        switch(type)
        {
        case int_t: return left.integer + right.integer;
        case fp_t: return left.real + right.real;
        }
        return {};
    }

    value operator-(const value& left, const value& right) const
    {
        switch(type)
        {
        case int_t: return left.integer - right.integer;
        case fp_t: return left.real - right.real;
        }
        return {};
    }
};

struct sim_state
{
    my_stack<int> ins_stack;
    curandState* random;
    value* variables;
};

inline bool boolify(const double x)
{
    return abs(x) > DBL_EPSILON;
}

#define IS_LEAF(x) ((x) < 4)
struct instruction 
{
    enum operators
    {
        //value types
        literal_i = 0,
        global_var = 1,
        local_var = 2,
        random_i = 3,

        //arithmatic types
        plus_i,
        minus_i,
        multiply_i,
        division_i,
        ratio_i,
        power_i,
        negation_i,
        sqrt_i,
        modulo_i,
        ln_i,

        //boolean types
        and_i,
        or_i,
        less_equal_i,
        greater_equal_i,
        less_i,
        greater_i,
        equal_i,
        not_equal_i,
        not_i,

        //control types
        goto_i,
        return_i,
        init_i,
        compiled_i
        
    } operand = literal_i;
    
    union expr_data
    {
        value value;
        int variable_id{};
        int length;
        int compile_id;
    } data = {{1}};

    CPU GPU value evaluate_expression(int& header, sim_state* state) const
    {
        double v1,v2;
        switch (this->operand) {
        case literal_i: return value{this->data.value};
        case global_var: return state->variables[this->data.variable_id];
        case local_var: return state->variables[this->data.variable_id]; //Todo handle better
        case random_i: return curand_uniform_double(state->random);
        case plus_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 + v2;
        case minus_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 - v2;
        case multiply_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 * v2;
        case division_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 / v2;
        case ratio_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 / v2;
        case power_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return pow(v1,v2);
        case negation_i:
            v1 = state->ins_stack.pop();
            return -v1;
        case sqrt_i:
            v1 = state->ins_stack.pop();
            return sqrt(v1);
        case modulo_i:
            v1 = state->ins_stack.pop();
            return -v1;
        case ln_i:
            v1 = state->ins_stack.pop();
            return log(v1);
        case and_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return boolify(v1) && boolify(v2);
        case or_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return boolify(v1) || boolify(v2);
        case less_equal_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 <= v2;
        case greater_equal_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 >= v2;
        case less_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 < v2;
        case greater_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return v1 > v2;
        case equal_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return boolify(v1 - v2);
        case not_equal_i:
            v2 = state->ins_stack.pop();
            v1 = state->ins_stack.pop();
            return !boolify(v1 - v2);
        case not_i:
            v1 = state->ins_stack.pop();
            return boolify(v1);
        case goto_i:
            v1 = state->ins_stack.pop();
            header += boolify(v1) * data.length; //equivilent to: "boolify(v1) ? data.length : 0", but faster
        case return_i:
            return header += data.length;
        case init_i: //TODO should throw error
        case compiled_i: return 0.0; //TODO should throw error
        }
        return 0.0;
    }
};

struct function
{
    instruction* ins;
    int size;

    double call(sim_state* state) const
    {
        state->ins_stack.clear();
        for (int i = 1; i < size; ++i)
        {
            ins[i].evaluate_expression(i, state);
        }
        return state->ins_stack.pop();
    }
};

