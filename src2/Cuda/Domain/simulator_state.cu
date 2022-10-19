#include "simulator_state.h"
#include "expressions/expression.h"

double simulator_state::evaluate_expression(expression* expr)
{
    this->value_stack.clear();
    this->expression_stack.clear();
    
    expression* current = expr;
    while (true)
    {
        while(current != nullptr)
        {
            
            this->expression_stack.push(current);
            this->expression_stack.push(current);

            // if(!current->is_leaf()) //only push twice if it has children
            //      this->expression_stack_->push(current);
            current = current->get_left();
        }
        if(this->expression_stack.is_empty())
        {
            break;
        }
        current = this->expression_stack.pop();
        
        if(!this->expression_stack.is_empty() && this->expression_stack.peak() == current)
        {
            current = current->get_right(&this->value_stack);
        }
        else
        {
            current->evaluate(this);
            current = nullptr;
        }
    }

    if(this->value_stack.count() == 0)
    {
        printf("Expression evaluation ended in no values! PANIC!\n");
        return 0;
    }
    return this->value_stack.pop();
}