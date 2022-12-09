#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include <list>
#include "pugixml.hpp"
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include "helper_methods.h"
#include "string_extractor.h"
#include "declaration.h"
#include "declaration_parser.h"
#include <utility>
#include "variable_expression_evaluator.h"


#include "../common/macro.h"
// #define GPU __device__
// #define CPU __host__

using namespace std;
using namespace pugi;
using namespace helper;

struct node_with_system_id
{
private:
    node* node_;
    int system_id_;
public:
    node_with_system_id(node* node, int system_id_) {this->node_ = node; this->system_id_ = system_id_;}
    node* get_node() const {return this->node_;}
    int get_system_id() const {return this->system_id_;}
    bool is_in_system(int system_id) const {return system_id == system_id_;}
};

class uppaal_xml_parser
{
   
private:
    int init_node_id_{};
    list<clock_var>* vars_list_ = new list<clock_var>();
    int vars_id_ = 0;
    int chan_id_ = 1;
    int system_count_ = 0;
    int update_id_ = 0;
    int edge_id_ = 0;
    declaration_parser dp_;
    template <typename T> void fill_expressions(const list<string>& expressions, list<T>* t);
    unordered_map<string, double> const_local_vars{};
    unordered_map<string, double> const_global_vars{};
    unordered_map<string, int> timers_map_{};
    unordered_map<string, int> global_timers_map_{};
    unordered_map<string, int> vars_map_{};
    unordered_map<string, int> global_vars_map_{};
    unordered_map<int, list<edge>> node_edge_map{};
    unordered_map<int, string>* node_names_ = new unordered_map<int, string>();
    unordered_map<int, node_with_system_id>* nodes_map_= new unordered_map<int, node_with_system_id>();
    list<node*>* nodes_ = new list<node*>();
    list<int> start_nodes_{};
    
    int get_timer_id(const string& expr) const;
    void get_condition_strings(const string& con, string* left, string* op, string* right);
    node* get_node(const int target_id, const list<node*>* arr) const;
    int handle_sync(const string& input) const;
    list<update> handle_assignment(const string& input);
    bool is_if_statement(const string& expr);
    expr* handle_if_statement(const string& input);
    void handle_transitions(const xml_node trans);
    void handle_locations(const xml_node locs);
    arr<node*> after_processing();
    void init_global_clocks(const pugi::xml_document* doc);
    void init_local_clocks(const pugi::xml_node template_node);
    network parse_xml(const char* file_path);
    template<typename T, typename V>
    static void insert_to_map(unordered_map<T, V>* map, const T& key, const V& value)
    {
        if (map->count(key)) (*map)[key] = value;
        else map->insert(std::pair<T, V>(key, value));
    }

public:
    unordered_map<int, string>* get_nodes_with_name() const {return this->node_names_;}
    unordered_map<int, node_with_system_id>* get_subsystems() const {return this->nodes_map_;}
    uppaal_xml_parser();
    network parse(string file_path);
    static bool try_parse_block_threads(const std::string& str, unsigned* out_blocks, unsigned* out_threads);
    static bool try_parse_units(const std::string& str, bool* is_time, double* value);
};
#endif