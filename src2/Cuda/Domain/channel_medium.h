#pragma once

#ifndef CHANNEL_MEDIUM_H
#define CHANNEL_MEDIUM_H

#include "../common/macro.h"
#include "node_t.h"
#include "edge_t.h"

struct model_state;
class simulator_state;

struct channel_listener
{
    model_state* state;
    edge_t* edge;
    CPU GPU void broadcast(simulator_state* sim_state) const; //Assumes the edge is valid
};

struct channel_stack
{
    unsigned count = 0;
    channel_listener* listeners{};

    CPU GPU void add(model_state* state, edge_t*);
    CPU GPU void remove(const node_t* node);
};

class channel_medium
{
    channel_stack* store_;
    unsigned channels_;
    unsigned width_;

    CPU GPU explicit channel_medium(channel_stack* store, unsigned channels, unsigned max_width);
public:
    CPU GPU explicit channel_medium(unsigned channels, unsigned max_width);

    CPU GPU void init(const lend_array<model_state>* states) const;
    CPU GPU void add(model_state* state) const;
    CPU GPU void remove(node_t* node) const; //remove all entries under given channel, which contain this node
    CPU GPU bool listener_exists(unsigned channel_id) const;
    CPU GPU channel_listener* find_listener(unsigned channel_id) const;
    CPU GPU channel_listener* pick_random_valid_listener(
                    unsigned channel_id,
                    simulator_state* state,
                    curandState* r_state) const;
    CPU GPU void clear() const;
};

#endif