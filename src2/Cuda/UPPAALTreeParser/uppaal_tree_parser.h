#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include <list>
#include "pugixml.hpp"
#include "../Domain/clock_variable.h"
#include "../Domain/update_t.h"
#include "../Domain/node_t.h"
#include "../Domain/expressions/constraint_t.h"
#include "../Domain/stochastic_model_t.h"
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include "helper_methods.h"
#include "../Domain/edge_t.h"
#include "declaration.h"
#include "declaration_parser.h"
#include "update_parser.h"

#define GPU __device__
#define CPU __host__

using namespace std;
using namespace pugi;
using namespace helper;

class uppaal_tree_parser
{
private:
    int init_node_id_{};
    list<clock_variable>* timer_list_ = new list<clock_variable>();
    list<clock_variable>* var_list_ = new list<clock_variable>();
    int timer_amount_ = 0;
    int var_id_ = 0;
    int clock_id_ = 0;
    declaration_parser dp_;
    template <typename T> void get_guys(const list<string>& expressions, list<T>* t);
    const map<string, int> timers_map_{};
    map<string, int> vars_map_{};
    map<string, int> global_vars_map_{};
    list<node_t*>* nodes_ = new list<node_t*>();
    list<node_t*>* goal_nodes_ = new list<node_t*>();
    list<int> branchpoint_nodes{};
    list<int> start_nodes_{};
    int get_timer_id(const string& expr) const;
    node_t* get_node(const int target_id, const list<node_t*>* arr) const;
    void init_clocks(const pugi::xml_document* doc);
    stochastic_model_t parse_xml(char* file_path);
public:
    uppaal_tree_parser();
    __host__ stochastic_model_t parse(char* file_path);
};
#endif