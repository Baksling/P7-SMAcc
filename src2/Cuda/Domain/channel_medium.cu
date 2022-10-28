#include "channel_medium.h"
#include "simulator_state.h"

CPU GPU void channel_listener::broadcast(simulator_state* sim_state) const
{
    sim_state->medium->remove(this->state->current_node);
    
    node_t* dest = edge->get_dest();

    //update state
    this->state->current_node = dest;
    this->state->reached_goal = dest->is_goal_node();
    
    this->edge->execute_updates(sim_state);
    
    sim_state->medium->add(this->state);
}

CPU GPU void channel_stack::add(model_state* state, edge_t* edge)
{
    this->listeners[this->count] = channel_listener{state, edge};
    this->count++;
}

CPU GPU void channel_stack::remove(const node_t* node)
{
    if(this->count == 0) return; //no entries

    const channel_listener* end = &this->listeners[this->count -1];

    if(end->state->current_node == node)
    {
        this->count--;
        return;
    }

    //loop over all listeners except end. if this.count = 1, then this is skipped
    for (unsigned i = 0; i < this->count - 1; ++i)
    {
        if(this->listeners[i].state->current_node != node) continue;

        this->listeners[i] = *end; //switch the listener containing 'node to remove' with the end listener
        this->count--; //remove end node, which is now the node to remove
        return;
    }
}

CPU GPU channel_medium::channel_medium(channel_stack* store, const unsigned channels, const unsigned max_width)
{
    this->store_ = store;
    this->channels_ = channels;
    this->width_ = max_width;
}

CPU GPU channel_medium::channel_medium(const unsigned channels, const unsigned max_width)
{
    this->channels_ = channels;
    this->width_ = max_width;

    if(channels == 0 || max_width == 0)
    {
        this->store_ = nullptr;
        return;
    }
    this->store_ = static_cast<channel_stack*>(malloc(channels * sizeof(channel_stack)));
    for (unsigned i = 0; i < channels; ++i)
    {
        this->store_[i] = channel_stack{0,
            static_cast<channel_listener*>(malloc(max_width*sizeof(channel_listener))) };
    }
}

CPU GPU void channel_medium::init(const lend_array<model_state>* states) const
{
    for (int i = 0; i < states->size(); ++i)
    {
        this->add(states->at(i));
    }
}

CPU GPU void channel_medium::add(model_state* state) const
{
    node_t* node = state->current_node;
    const lend_array<edge_t*> edges = node->get_edges();
        
    for (int i = 0; i < edges.size(); ++i)
    {
        edge_t* edge = edges.get(i);
        const unsigned channel_id = edge->get_channel();
        if(channel_id == NO_CHANNEL) continue;

        this->store_[channel_id].add(state, edge);
    }
}

CPU GPU void channel_medium::remove(node_t* node) const
{
    const lend_array<edge_t*> edges = node->get_edges();
        
    for (int i = 0; i < edges.size(); ++i)
    {
        const unsigned channel_id = edges.get(i)->get_channel();
        if(channel_id == NO_CHANNEL) continue;

        this->store_[channel_id].remove(node);
    }
}

CPU GPU bool channel_medium::listener_exists(const unsigned channel_id) const
{
    return this->store_[channel_id].count > 0;
}

CPU GPU channel_listener* channel_medium::find_listener(const unsigned channel_id) const
{
    if (!this->listener_exists(channel_id)) return nullptr;
    const channel_stack* stack = &this->store_[channel_id];
    return &stack->listeners[stack->count - 1 ]; //returns end node
}


CPU GPU channel_listener* channel_medium::pick_random_valid_listener(const unsigned channel_id, simulator_state* state, curandState* r_state) const
{
    if(!listener_exists(channel_id)) return nullptr;
    const channel_stack* stack = &this->store_[channel_id];
    const unsigned start_index = curand(r_state) % stack->count;

    for (unsigned i = 0; i < stack->count; ++i)
    {
        channel_listener* listener = &stack->listeners[(start_index + i) % stack->count];
        if(listener->edge->evaluate_constraints(state))
            return listener;
    }

    return nullptr;
}

CPU GPU void channel_medium::clear() const
{
    for (unsigned i = 0; i < this->channels_; ++i)
    {
        this->store_[i].count = 0;
    }
}



