#include <curand_kernel.h>
#include <curand_uniform.h>

#include "../common/my_stack.h"

union i_val
{
    i_val(const float fp)
    {
        this->fp = fp;
    }

    i_val(const double fp)
    {
        this->fp = static_cast<float>(fp);
    }

    i_val(const int i)
    {
        this->i = i;
    }

    int i{};
    float fp;
};

struct intp_v
{
    intp_v(const float fl)
    {
        this->type = types::f32;
        this->value.float32 = fl;
    }

    intp_v(const int i)
    {
        this->type = types::i32;
        this->value.int32 = i;
    }
    intp_v(const double d)
    {
        this->type = types::f64;
        this->value.float64 = d;
    }
    
    enum types
    {
        i32,
        f32,
        f64,
    } type;
    union
    {
        int int32;
        float float32;
        double float64;
    } value{};

    double as_double()
    {
        switch (this->type) {
            case i32: return value.int32; 
            case f32: return static_cast<double>(value.float32);
            case f64: return value.float64;
        } 
    }

    intp_v operator+(const intp_v& first, const intp_v& second)
    {
        switch (first.type) {
            case i32: return first.value.int32 + second.value.int32; 
            case f32: return first.value.float32 + second.value.float32;
            case f64: return first.value.float64 + second.value.float64;
        }
    }

    intp_v operator-(intp_v& first, intp_v& second)
    {
        switch (first.type) {
        case i32: return first.value.int32 - second.value.int32; 
        case f32: return first.value.float32 - second.value.float32;
        case f64: return first.value.float64 - second.value.float64;
        }
    }
};

struct sim_state
{
    my_stack<i_val> int_stack;
    curandState* random;
    double* variables;
};

struct pn_expr
{
    enum types
    {
        //axioms
        literal_s = 0, 
        variable_s = 1,
        random_s = 2,

        //control
        cgoto_s = 4,

        //arithmatic
        plus_s, 
        minus_s, 
        multiply_s, 
        pow_s,
        sqrt_s,
        modulo_s,
        ln_s,

        //boolean
        negation_s,
        less_s,
        less_equal_s,
        greater_s,
        greater_equal_s,
        equal_s,
        not_equal_s,
        and_s,
        or_s,

        //compiled,
        compiled_s
    } type;

    union expr_metadata
    {
        i_val value;
        int variable_id;
        int goto_dest;
    } data;

    i_val evaluate(sim_state* state)
    {
        constexpr int neg_flip = (1 << 31); 
        constexpr i_val default_ = 0.0;
        i_val v1 = default_;
        i_val v2 = default_;
        switch(this->type)
        {
        case literal_s: return this->data.value;
        case variable_s: return state->variables[this->data.variable_id];
        case random_s: return curand_uniform(state->random);
        case cgoto_s: return default_;
        case plus_s:
            v2 = state->int_stack.pop();
            v1 = state->int_stack.pop();
            return v1 + v2;
        case minus_s:
            v2 = state->int_stack.pop();
            v1 = state->int_stack.pop();
            return v1 - v2;
        case multiply_s:
            v2 = state->int_stack.pop();
            v1 = state->int_stack.pop();
            return v1 * v2;
        case pow_s:
            v2 = state->int_stack.pop();
            v1 = state->int_stack.pop();
            return pow(v1.fp, v2.fp);
        case sqrt_s:
            v1 = state->int_stack.pop();
            return sqrt(v1.fp);
        case modulo_s:
            v2 = state->int_stack.pop();
            v1 = state->int_stack.pop();
            return (v1.i % v2.i);
        case ln_s:
            v1 = state->int_stack.pop();
            return log(v1.fp);
        case negation_s:
            v1 = state->int_stack.pop();
            return neg_flip ^ v1.i; //works for both float and ints
        case less_s: break;
        case less_equal_s: break;
        case greater_s: break;
        case greater_equal_s: break;
        case equal_s: break;
        case not_equal_s: break;
        case and_s: break;
        case or_s: break;
        case compiled_s: break;
        default: ;
        }
        return default_;
    }
};

struct function
{
    pn_expr* statements;
    int size;

    intp_v call(sim_state* state) const
    {
        state->int_stack.clear();
        // int i = 0;
        // int* ip = &i;
        for (int i = 0; i < size; ++i)
        {
            pn_expr* stmt = &statements[i];
            // if(stmt->type == pn_expr::return_s)
            //     return state->int_stack.pop().as_double(); //TODO this can be converted to a goto
            
            if(stmt->type == pn_expr::cgoto_s) //TODO this can be moved to eval scope
                i = i + stmt->data.goto_dest * !state->int_stack.pop().value.int32;
            else
                state->int_stack.push_val(stmt->evaluate(state));
        }
        return state->int_stack.pop();
    }
};

