#ifndef GUARD_H
#define GUARD_H

#include "Timer.h"

enum logical_operator
{
    less_equal = 0,
    greater_equal,
    less,
    greater,
    equal,
    not_equal
};

class guard
{
private:
    logical_operator type_;
    double value_;
    timer* timer_;
public:
    guard(logical_operator type, double value, timer* timer);
    double get_value() const;
    bool validate_guard();
    logical_operator get_type();
};


#endif // GUARD_H