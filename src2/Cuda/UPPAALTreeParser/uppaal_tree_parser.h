﻿#pragma once

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

struct node_with_system_id
{
private:
    node_t* node_;
    int system_id_;
public:
    node_with_system_id(node_t* node, int system_id_) {this->node_ = node; this->system_id_ = system_id_;}
    node_t* get_node() const {return this->node_;}
    int get_system_id() const {return this->system_id_;}
    bool is_in_system(int system_id) const {return system_id == system_id_;}
};

class uppaal_tree_parser
{
   
private:
    int init_node_id_{};
    list<clock_variable>* timer_list_ = new list<clock_variable>();
    list<clock_variable>* var_list_ = new list<clock_variable>();
    int timer_amount_ = 0;
    int var_id_ = 0;
    int clock_id_ = 0;
    int chan_id_ = 0;
    int system_count = 0;
    declaration_parser dp_;
    template <typename T> void get_guys(const list<string>& expressions, list<T>* t);
    unordered_map<string, int> timers_map_{};
    unordered_map<string, int> vars_map_{};
    unordered_map<string, int> global_vars_map_{};
    unordered_map<int, string>* node_names_ = new unordered_map<int, string>();
    unordered_map<int, node_with_system_id>* nodes_map_= new unordered_map<int, node_with_system_id>();
    list<node_t*>* nodes_ = new list<node_t*>();
    list<node_t*>* goal_nodes_ = new list<node_t*>();
    list<int> branchpoint_nodes{};
    list<int> start_nodes_{};
    list<int> sync_chan_ids_{};
    int get_timer_id(const string& expr) const;
    string is_timer(const string& expr) const;
    node_t* get_node(const int target_id, const list<node_t*>* arr) const;
    void init_clocks(const pugi::xml_document* doc);
    stochastic_model_t parse_xml(char* file_path);
    template<typename T, typename V>
    static void insert_to_map(unordered_map<T, V>* map, const T& key, const V& value)
    {
        if (map->count(key)) (*map)[key] = value;
        else map->insert(std::pair<T, V>(key, value));
    }

public:
    unordered_map<int, string>* get_nodes_with_name() const {return this->node_names_;}
    unordered_map<int, node_with_system_id>* get_subsystems() const {return this->nodes_map_;}
    uppaal_tree_parser();
    __host__ stochastic_model_t parse(string file_path);
};
#endif