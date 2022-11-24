#include "simulator_tools.h"

#include "../Domain/edge_t.h"
#include "../Domain/simulator_state.h"
#include "../common/lend_array.h"


CPU GPU inline bool bit_is_set(const unsigned long long n, const unsigned int i)
{
    return (n) & (1UL << i);
}


CPU GPU inline void set_bit(unsigned long long* n, const unsigned int i)
{
    (*n) |=  (1UL << (i));
}

CPU GPU inline void unset_bit(unsigned long long* n, const unsigned int i)
{
    (*n) &= ~(1UL << (i));
}


CPU GPU edge_t* find_valid_edge_heap(
    simulator_state* state,
    const lend_array<edge_t>* edges,
    curandState* r_state)
{
    // return nullptr;
    edge_t** valid_edges_arr = static_cast<edge_t**>(malloc(sizeof(void*) * edges->size()));  // NOLINT(bugprone-sizeof-expression)
    if(valid_edges_arr == nullptr) printf("COULD NOT ALLOCATE HEAP MEMORY\n");
    edge_t* valid_edge = nullptr;
    double weight_sum = 0.0;
    unsigned valid_count = 0;
    
    for (unsigned i = 0; i < edges->size(); ++i)
    {
        valid_edges_arr[i] = nullptr; //clean malloc
        edge_t* edge = edges->at(i);
        if(valid_edge->evaluate_constraints(state))
        {
            valid_edge = edge;
            valid_edges_arr[valid_count++] = edge;
            weight_sum += edge->get_weight(state);
        }
    }
    
    if(valid_count == 0 || valid_count == 1)
    {
        free(valid_edges_arr);
        return valid_edge; //is nullptr if == 0 and valid if == 1
    }

    //curand_uniform return ]0.0f, 1.0f], but we require [0.0f, 1.0f[
    //conversion from float to int is floored, such that a array of 10 (index 0..9) will return valid index.
    const double r_val = (1.0 - curand_uniform_double(r_state))*weight_sum;
    double r_acc = 0.0; 

    //pick the weighted random value.
    for (unsigned i = 0; i < valid_count; ++i)
    {
        valid_edge = valid_edges_arr[i];
        r_acc += valid_edge->get_weight(state);
        if(r_val < r_acc) break;
    }

    free(valid_edges_arr);
    return valid_edge;
    
}

CPU GPU edge_t* simulator_tools::choose_next_edge_bit(
    simulator_state* state,
    const lend_array<edge_t>* edges,
    curandState* r_state)
{
    if(static_cast<size_t>(edges->size()) > sizeof(size_t))
    {
        printf("Too many edge options.");
        return find_valid_edge_heap(state, edges, r_state);
    }
    
    unsigned long long valid_edges_bitarray = 0UL;
    unsigned int valid_count = 0;
    edge_t* valid_edge = nullptr;
    double weight_sum = 0.0;
    
    for (unsigned i = 0; i < edges->size(); ++i)
    {
        edge_t* edge = edges->at(i);
        if(edge->is_listener()) continue;
        if(edge->evaluate_constraints(state))
        {
            set_bit(&valid_edges_bitarray, i);
            valid_edge = edge;
            valid_count++;
            const double weight = edge->get_weight(state);
            weight_sum += weight; 
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
    for (unsigned i = 0; i < edges->size(); ++i)
    {
        if(!bit_is_set(valid_edges_bitarray, i)) continue;
        const double weight = edges->at(i)->get_weight(state);
        if(weight <= 0) continue;
        
        valid_edge = edges->at(i);
        r_acc += weight;
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


CPU GPU edge_t* simulator_tools::choose_next_edge(simulator_state* state, const lend_array<edge_t>* edges, curandState* r_state)
{
    //if no possible edges, return null pointer
    if(edges->size() == 0) return nullptr;
    if(edges->size() == 1)
    {
        edge_t* edge = edges->at(0);
        return edge->evaluate_constraints(state)
                ? edge
                : nullptr;
    }

    if(static_cast<size_t>(edges->size()) < sizeof(size_t))
        return choose_next_edge_bit(state, edges, r_state);
    else
        return find_valid_edge_heap(state, edges, r_state);
}