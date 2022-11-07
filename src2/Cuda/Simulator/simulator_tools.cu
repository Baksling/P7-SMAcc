#include "simulator_tools.h"

#include "../Domain/edge_t.h"
#include "../Domain/simulator_state.h"
#include "../common/allocation_helper.h"
#include "../common/array_t.h"
#include "../common/lend_array.h"


CPU GPU bool simulator_tools::bit_is_set(const unsigned long long* n, const unsigned int i)
{
    return (*n) & (1UL << i);
}


CPU GPU void simulator_tools::set_bit(unsigned long long* n, const unsigned int i)
{
    (*n) |=  (1UL << (i));
}

CPU GPU void simulator_tools::unset_bit(unsigned long long* n, const unsigned int i)
{
    (*n) &= ~(1UL << (i));
}


CPU GPU edge_t* find_valid_edge_heap(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    // return nullptr;
    edge_t** valid_edges = static_cast<edge_t**>(malloc(sizeof(edge_t*) * edges->size()));  // NOLINT(bugprone-sizeof-expression)
    if(valid_edges == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    int valid_count = 0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        valid_edges[i] = nullptr; //clean malloc
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
            valid_edges[valid_count++] = edge;
    }
    
    if(valid_count == 0)
    {
        free(valid_edges);
        return nullptr;
    }
    if(valid_count == 1)
    {
        edge_t* result = valid_edges[0];
        free(valid_edges);
        return result;
    }

    //summed weight
    double weight_sum = 0.0;
    for(int i = 0; i < valid_count; i++)
    {
        weight_sum += valid_edges[i]->get_weight(state);
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state))*weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    for (int i = 0; i < valid_count; ++i)
    {
        edge_t* temp = valid_edges[i];
        r_acc += temp->get_weight(state);
        if(r_val < r_acc)
        {
            free(valid_edges);
            return temp;
        }
    }

    //This should be handled in for loop.
    //This is for safety :)
    edge_t* edge = valid_edges[valid_count - 1];
    free(valid_edges);
    return edge;
}

CPU GPU edge_t* find_valid_edge_fast(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->get(i);
        if(edge->evaluate_constraints(state))
        {
            simulator_tools::set_bit(&valid_edges_bitarray, i);
            valid_edge = edge;
            valid_count++;
        }
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;
    
    //summed weight
    double weight_sum = 0.0;
    for(int i = 0; i  < edges->size(); i++)
    {
        if(simulator_tools::bit_is_set(&valid_edges_bitarray, i))
            weight_sum += edges->get(i)->get_weight(state);
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state)) * weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges->size(); ++i)
    {
        if(!simulator_tools::bit_is_set(&valid_edges_bitarray, i)) continue;

        valid_edge = edges->get(i);
        r_acc += valid_edge->get_weight(state);
        if(r_val < r_acc)
        {
            return valid_edge;
        }
    }
    return valid_edge;
}



CPU GPU edge_t* simulator_tools::choose_next_edge_bit(
    simulator_state* state,
    const lend_array<edge_t*>* edges,
    curandState* r_state)
{
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    double weight_sum = 0.0;
    
    for (int i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->get(i);
        if(edge->is_listener()) continue;
        if(edge->evaluate_constraints(state))
        {
            simulator_tools::set_bit(&valid_edges_bitarray, i);
            valid_edge = edge;
            valid_count++;
            weight_sum += edge->get_weight(state);
        }
    }

    if(valid_count == 0) return nullptr;
    if(valid_count == 1 && valid_edge != nullptr) return valid_edge;

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state)) * weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    valid_edge = nullptr; //reset valid edge !IMPORTANT
    for (int i = 0; i < edges->size(); ++i)
    {
        if(!simulator_tools::bit_is_set(&valid_edges_bitarray, i)) continue;

        valid_edge = edges->get(i);
        r_acc += valid_edge->get_weight(state);
        if(r_val < r_acc) break;
    }
    return valid_edge;
}



// CPU GPU next_edge simulator_tools::choose_next_edge_channel(
//     simulator_state* state,
//     const lend_array<edge_t*>* edges,
//     curandState* r_state)
// {
//     if(edges->size() == 0) return next_edge{nullptr, nullptr};
//     state->edge_stack.clear();
//     
//     edge_t* edge;
//     channel_listener* listener;
//     double weight_sum = 0.0;
//     
//     for (int i = 0; i < edges->size(); ++i)
//     {
//         edge = edges->get(i);
//         listener = nullptr; //reset to sync with 'edge' variable.
//         if(edge->is_listener()) continue; //listener only moved using broadcast
//         if(!edge->evaluate_constraints(state)) continue;
//
//         const unsigned channel = edge->get_channel();
//         if(channel != NO_CHANNEL)
//         {
//             //if nullptr is returned, that means no valid states are available.
//             listener = state->medium->pick_random_valid_listener(channel, state, r_state);
//             if(listener == nullptr) continue; //edge might be valid, but no valid listener. Dont pick this one
//         }
//
//         const double weight = edges->get(i)->get_weight(state);
//         weight_sum += weight; //accumulate sum of valid edges weight
//         state->edge_stack.push(next_edge_weighted{ edge, listener, weight });
//     }
//
//     if(state->edge_stack.count() == 0) return next_edge{nullptr, nullptr};
//     if(state->edge_stack.count() == 1) //if only one valid
//     {
//         edge = state->edge_stack.peak_at()->next;
//         listener = state->edge_stack.peak_at()->listener;
//         return next_edge{edge, listener};
//     }
//
//     //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
//     //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
//     const double r_val = (1.0 - curand_uniform_double(r_state)) * weight_sum;
//     double r_acc = 0.0;
//     
//     //pick the weighted random value.
//     edge = nullptr; //reset valid edge !IMPORTANT
//     listener = nullptr;
//
//     //pick random element of stack
//     while (state->edge_stack.count() > 0)
//     {
//         const next_edge_weighted edge_n = state->edge_stack.pop();
//         edge = edge_n.next;
//         listener = edge_n.listener;
//         r_acc += edge_n.weight;
//
//         if(r_val < r_acc) break;
//     }
//
//     state->edge_stack.clear();
//     return next_edge{edge, listener };
// }


CPU GPU edge_t* simulator_tools::choose_next_edge(simulator_state* state, const lend_array<edge_t*>* edges, curandState* r_state)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->get(0);
        return edge->evaluate_constraints(state)
                ? edge
                : nullptr;
    }

    if(static_cast<unsigned long long>(edges->size()) < sizeof(unsigned long long)*8)
        return find_valid_edge_fast(state, edges, r_state);
    else
        return find_valid_edge_heap(state, edges, r_state);
}