#pragma once

template<typename T>
class cuda_stack
{
    double* values_;
    unsigned int size_;
    unsigned int stack_pointer_ = 0;
public:
    explicit cuda_stack(unsigned int size);
    double peak();
    double pop();
    void push(double value);
    unsigned int count() const;
};
