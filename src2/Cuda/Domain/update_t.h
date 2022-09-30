#pragma once
#include "common.h"
#include "timer_t.h"

class update_t : public element
{
private:
    int id_;
    int timer_id_;
    double timer_value_;
public:
    update_t(int id, int timer_id, double timer_value);
    GPU void update_timer(const lend_array<timer_t>* timers) const;
    void accept(visistor& v) override;
    int get_timer_id() const;
    float get_timer_value() const;
    int get_id() const;
};
