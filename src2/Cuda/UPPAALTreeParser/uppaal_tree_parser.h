#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include <list>
#include "pugixml.hpp"
#include "../Domain/clock_timer_t.h"
#include "../Domain/update_t.h"
#include "../Domain/node_t.h"
#include "../Domain/constraint_t.h"
#include "../Domain/stochastic_model_t.h"
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

class uppaal_tree_parser
{
private:
    int init_node_id_;
    clock_timer_t* timer_list_;
    int timer_amount_ = 0;
    map<string, int> timers_map_;
    list<node_t*>* nodes_ = new list<node_t*>();
    list<list<edge_t>> edge_list_;
    list<list<constraint_t>> guard_list_;
    list<list<constraint_t>> invariance_list_;
    list<list<update_t>> update_list_;
    list<int> branchpoint_nodes;
    int get_timer_id(string expr) const;
    node_t* get_node(int target_id);
    void init_lists(pugi::xml_document* doc);
public:
    uppaal_tree_parser();
    __host__ stochastic_model_t parse_xml(char* file_path);
};
#endif