#pragma once
#include "node_t.h"
#include "update_t.h"

class edge_t : public element
{
private:
    int id_;
    float weight_;
    node_t* dest_;
    constraint_t* guard_;
    array_t<update_t> updates_{0};
public:
    explicit edge_t(int id, float weight, node_t* dest, constraint_t* guard = nullptr);
    GPU float get_weight() const;
    void set_updates(std::list<update_t>* updates);
    GPU bool evaluate_constraints(const lend_array<timer_t>* timers) const;
    void accept(visistor& v) override;
    node_t* get_dest_node() const;
    int get_id() const;
    lend_array<update_t> get_updates();
};
