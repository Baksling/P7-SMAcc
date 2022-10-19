#ifndef ALLOCATION_HELPER_H
#define ALLOCATION_HELPER_H

#include <list>
#include <unordered_map>

//Prototype :-D &*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*&*
class node_t;

struct allocation_helper
{
    std::list<void*>* free_list;
    std::unordered_map<node_t*, node_t*>* node_map;
};

#endif