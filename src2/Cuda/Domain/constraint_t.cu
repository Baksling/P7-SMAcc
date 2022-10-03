#include "constraint_t.h"

#include <assert.h>

GPU CPU double cuda_abs(const double f)
{
    return f < 0 ? -f : f;
}

GPU CPU bool is_boolean_operator(const logical_operator op)
{
    switch (op)
    {
    case less_equal: return false;
    case greater_equal: return false;
    case less: return false;
    case greater: return false;
    case equal: return false;
    case not_equal: return false;
    case And: return true;
    case Or: return true;
    case Not: return true;
    }
    return false;
}


CPU GPU double constraint_t::get_logical_value(const int timer_id, const lend_array<clock_timer_t>* timer_arr) const
{
    if(is_boolean_operator(this->type_)) return BIG_DOUBLE;
    
    return timer_id == NO_ID
    ? static_cast<double>(this->value_)
    : timer_arr->at(timer_id)->get_time();
}

GPU CPU bool constraint_t::validate_type() const
{
    switch (this->type_)
    {
    case less_equal:
    case greater_equal:
    case less: 
    case greater:
    case equal:
    case not_equal:
        return this->timer_id1_ != NO_ID && (this->value_ >= 0 || timer_id2_ != NO_ID);
    case And:
    case Or:
        return this->con1_ != nullptr && this->con2_ != nullptr;
    case Not:
        return this->con1_ != nullptr;
    }
    return false;
}

GPU CPU bool constraint_t::evaluate_as_leaf(const lend_array<clock_timer_t>* timer_arr) const
{
    const double v1 = this->get_logical_value(this->timer_id1_, timer_arr);
    const double v2 = this->get_logical_value(this->timer_id2_, timer_arr);
    
    switch (this->type_) {
        case less_equal: return v1 <= v2;
        case greater_equal: return v1 >= v2;
        case less: return v1 < v2;
        case greater: return v1 > v2;
        case equal: return cuda_abs(v1 - v2) < 0.01; //v1 == v2;
        case not_equal: return cuda_abs(v1 - v2) >= 0.01; //v1 != v2;
        case And: 
        case Or: 
        case Not: return false; 
    }
    return false;
}

GPU bool evaluate_boolean(const bool b1, const bool b2, const logical_operator type)
{
    switch (type) {
        case less_equal: return false;
        case greater_equal: return false;
        case less: return false;
        case greater: return false;
        case equal: return false; //v1 == v2;
        case not_equal: return false; //v1 != v2;
        case And: return b1 && b2;
        case Or: return b1 || b2;
        case Not: return !b1; 
    }
    return false;
}

unsigned int constraint_t::find_child_count() const
{
    if(!is_boolean_operator(this->type_)) return 0;

    switch (this->type_) {
        case less_equal:
        case greater_equal:
        case less: 
        case greater: 
        case equal: 
        case not_equal: return 0;
        case And:
        case Or:
            return this->con1_->children_count_ + this->con2_->children_count_ +  2;
        case Not: return this->con1_->children_count_ + 1; 
    }
    return 0;
}

constraint_t::constraint_t(const logical_operator type, constraint_t* con1, constraint_t* con2, const int timer_id1,
                           const int timer_id2, const float value)
{
    this->type_ = type;
    this->con1_ = con1;
    this->con2_ = con2;
    this->value_ = value;
    this->timer_id1_ = timer_id1;
    this->timer_id2_ = timer_id2;
    this->children_count_ = find_child_count();
    if(!this->validate_type())
    {
        throw std::invalid_argument( "The constraint is invalid >:(" );
    }
}

template<typename T>
class cuda_stack
{
private:
    int size_;
    T* stack_;
    int stack_pointer_;
public:
    GPU explicit cuda_stack(int size)
    {
        this->size_ = size;
        this->stack_ = static_cast<T*>(malloc(sizeof(T)*size_));
        this->stack_pointer_ = -1;
    }
    GPU void push(T item)
    {
        if(this->stack_pointer_ >= this->size_) return;
        ++this->stack_pointer_;
        this->stack_[this->stack_pointer_] = item;
    }
    GPU T top()
    {
        if(this->is_empty()) return NULL;
        return this->stack_[this->stack_pointer_];
    }
    GPU T pop()//might be nullptr
    {
        if(this->is_empty()) return NULL;
        T result = this->stack_[this->stack_pointer_];
        --this->stack_pointer_;
        return result;
    }
    GPU bool try_pop(T* p)
    {
        if(this->is_empty()) return false;
        *p = this->stack_[this->stack_pointer_];
        --this->stack_pointer_;
        return true;
    }
    GPU bool is_empty()
    {
        return this->stack_pointer_ < 0;
    }
    GPU int get_count() const
    {
        return (this->stack_pointer_ + 1) ;
    }
    GPU void free_internal()
    {
        free(this->stack_);
        this->stack_ = nullptr;
    }
    
};

GPU bool constraint_t::evaluate(const lend_array<clock_timer_t>* timer_arr)
{
    if(is_boolean_operator(this->type_) || this->children_count_ == 0) return this->evaluate_as_leaf(timer_arr); 

    cuda_stack<constraint_t*> stack = cuda_stack<constraint_t*>(this->children_count_*2+1);
    cuda_stack<bool> b_stack = cuda_stack<bool>(this->children_count_+1);
    constraint_t* current = this;
    
    while(true)
    {
        while(current != nullptr)
        {
            stack.push(current);
            stack.push(current);
            current = current->con1_;
        }
        if(stack.is_empty())
        {
            break;
        }
        current = stack.pop();
        if(!stack.is_empty() && stack.top() == current)
        {
            current = current->con2_;
        }
        else
        {
            if(current->type_ == logical_operator::Not)
            {
                assert(b_stack.get_count() > 0);
                const bool b1 = b_stack.pop();
                b_stack.push( evaluate_boolean(b1,false, current->get_type()) );
            }
            else if(is_boolean_operator(current->get_type()))
            {
                assert(b_stack.get_count() > 1);
                const bool b1 = b_stack.pop();
                const bool b2 = b_stack.pop();
                b_stack.push( evaluate_boolean(b1,b2, current->get_type()) );
            }            
            else
            {
                b_stack.push(current->evaluate_as_leaf(timer_arr));
            }
            current = nullptr;
        }
    }
    
    const bool result = b_stack.pop();
    stack.free_internal();
    b_stack.free_internal();
    return result;
}

GPU CPU logical_operator constraint_t::get_type() const
{
    return this->type_;
}

GPU double constraint_t::max_time_progression(const lend_array<clock_timer_t>* timer_arr, double max_progression) const
{
    cuda_stack<constraint_t*> stack = cuda_stack<constraint_t*>(this->children_count_);
    constraint_t* current = nullptr;
    
    while(true)
    {
        while(current != nullptr)
        {
            stack.push(current);
            stack.push(current);
            current = current->con1_;
        }
        if(stack.is_empty())
        {
            break;
        }
        current = stack.pop();
        if(!stack.is_empty() && stack.top() == current)
        {
            current = current->con2_;
        }
        else
        {
            if(current->type_ == less_equal || current->type_ == less)
            {
                double time = timer_arr->at(current->get_timer1_id())->get_time();
                double value = current->timer_id2_ != NO_ID
                    ? timer_arr->at(current->get_timer2_id())->get_time()
                    : static_cast<double>(current->get_value());

                double diff = current->timer_id2_ != NO_ID ? value - time : max_progression;
                if(diff < 0) diff = 0.0;
                max_progression = diff < max_progression ? diff : max_progression;
            }
            
            current = nullptr;
        }
    }

    return max_progression;
}

void constraint_t::accept(visitor* v)
{
    switch (this->type_)
    {
    case And:
    case Or:
        v->visit(this->con1_);
        v->visit(this->con2_);
        break;
    case Not:
        v->visit(this->con1_);
        break;
    default:
        break;
    }
}

CPU GPU int constraint_t::get_timer1_id() const
{
    return this->timer_id1_;
}
CPU GPU int constraint_t::get_timer2_id() const
{
    return this->timer_id2_;
}

CPU GPU float constraint_t::get_value() const
{
    return this->value_;
}


//! LESS THAN OR EQUAL
constraint_t constraint_t::less_equal_v(const int timer_id, const float value)
{
    return constraint_t{less_equal, nullptr, nullptr, timer_id, NO_ID, value};
}

constraint_t constraint_t::less_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{less_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};

}

//! GREATER THAN OR EQUAL
constraint_t constraint_t::greater_equal_v(const int timer_id, const float value)
{
    return constraint_t{greater_equal, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::greater_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{greater_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! LESS THAN
constraint_t constraint_t::less_v(const int timer_id, const float value)
{
    return constraint_t{less, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::less_t(const int timer_id, const int timer_id2)
{
    return constraint_t{less, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! GREATER THAN
constraint_t constraint_t::greater_v(const int timer_id, const float value)
{
    return constraint_t{greater, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::greater_t(const int timer_id, const int timer_id2)
{
    return constraint_t{greater, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! equal
constraint_t constraint_t::equal_v(const int timer_id, const float value)
{
    return constraint_t{equal, nullptr, nullptr, timer_id, NO_ID, value};

}

constraint_t constraint_t::equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//! NOT EQUAL
constraint_t constraint_t::not_equal_v(const int timer_id, const float value)
{
    return constraint_t{not_equal, nullptr, nullptr, timer_id, NO_ID, value};
}

constraint_t constraint_t::not_equal_t(const int timer_id, const int timer_id2)
{
    return constraint_t{not_equal, nullptr, nullptr, timer_id, timer_id2, UNUSED_VALUE};
}

//NOT
constraint_t constraint_t::not_constraint(constraint_t* constraint)
{
    return constraint_t{Not, constraint, nullptr, NO_ID, NO_ID, UNUSED_VALUE};
}

//OR
constraint_t constraint_t::or_constraint(constraint_t* constraint1, constraint_t* constraint2)
{
    return constraint_t{Or, constraint1, constraint2, NO_ID, NO_ID, UNUSED_VALUE};
}

//AND
constraint_t constraint_t::and_constraint(constraint_t* constraint1, constraint_t* constraint2)
{
    return constraint_t{Not, constraint1, constraint2, NO_ID, NO_ID, UNUSED_VALUE};
}

