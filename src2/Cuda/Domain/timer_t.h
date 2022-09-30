#pragma once
#include "common.h"

class timer_t : public element
{
private:
    int id_;
    double current_time_;
public:
    CPU GPU timer_t(int id, double start_value);
    CPU GPU int get_id() const;
    CPU GPU double get_time() const;
    GPU void set_time(double new_value);
    GPU void add_time(double progression);
    GPU timer_t duplicate() const;
    void accept(visistor& v) override;
};