#include "uppaal_tree_parser.h"

#include "uppaal_tree_parser.h"
#include <iostream>
#include "pugixml.hpp"
#include <list>
#include "../Domain/node_t.h"
#include "../Domain/update_t.h"
#include "../Domain/constraint_t.h"
//#include "../Domain/uneven_list.h"
#include "../Domain/timer_t.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>

#include "../../Node.h"

#define GPU __device__
#define CPU __host__

using namespace std;
using namespace pugi;

constraint_t get_constraint(const string& expr, const int timer_id, const float value)
{
    if(expr.find("<=") != std::string::npos)
        return constraint_t::less_equal_v(timer_id, value);
    if(expr.find(">=") != std::string::npos)
        return constraint_t::greater_equal_v(timer_id, value);
    if(expr.find("==") != std::string::npos)
        return constraint_t::equal_v(timer_id, value);
    if(expr.find('<') != std::string::npos)
        return constraint_t::less_v(timer_id, value);
    if(expr.find('>') != std::string::npos)
        return constraint_t::greater_v(timer_id, value);
    throw "Operand in " + expr + " not found, sad..";
}

constraint_t get_constraint(const string& expr, const int timer_id_1, const int timer_id_2)
{
    if(expr.find("<=") != std::string::npos)
        return constraint_t::less_equal_t(timer_id_1, timer_id_2);
    if(expr.find(">=") != std::string::npos)
        return constraint_t::greater_equal_t(timer_id_1, timer_id_2);
    if(expr.find("==") != std::string::npos)
        return constraint_t::equal_t(timer_id_1, timer_id_2);
    if(expr.find('<') != std::string::npos)
        return constraint_t::less_t(timer_id_1, timer_id_2);
    if(expr.find('>') != std::string::npos)
        return constraint_t::greater_t(timer_id_1, timer_id_2);
    throw "Operand in " + expr + " not found, sad..";
}

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

bool in_array(const char &value, const std::vector<char> &array)
{
    return std::find(array.begin(), array.end(), value) != array.end();
}

int get_expr_value(const string& expr)
{
    string expr_wout_spaces = replace_all(expr, " ", "");
    int index = expr_wout_spaces.length();
    while (true)
    {
        if (index == 0)
        {
            return 0;
        }
        
        if (!in_array(expr_wout_spaces[--index], {'1','2','3','4','5','6','7','8','9','0'}))
        {
            return stoi(expr_wout_spaces.substr(index+1));
        }
    }
}

float get_expr_value_float(const string& expr)
{
    int index = expr.length();
    while (true)
    {
        if (index == 0)
        {
            return stof(expr);
        }
        
        if (in_array(expr[--index], {'=',' ','<','>'}))
        {
            return stof(expr.substr(index+1));
        }
    }
}

template <typename T> T* list_to_arr(list<T> l)
{
    T* arr = (T*)malloc(sizeof(T)*l.size());
    int k = 0;
    for (int const &i: l) {
        arr[k++] = i;
    }
    
    return arr;
}

int xml_id_to_int(string id_string)
{
    return stoi(id_string.replace(0,2,""));
}

void print1(list<edge_t> const &list)
{
    for (auto i: list) {
        std::cout << i.get_weight() << " <-1-> " << i.get_weight()  << std::endl;
    }
}

void print2(list<list<edge_t>> const &list)
{
    for (auto const &i: list) {
        for (auto q: i) {

        }
    }
}

void print_2_guard(list<list<guard_d>> const &list)
{
    // int index = 0;
    // for (auto const &i: list) {
    //     cout << " "<< index++ << ":"<<endl;
    //     for (auto q: i) {
    //         std::cout << q.get_value() << " -> " << q.get_type()  << std::endl;
    //     }
    // }
}

int uppaal_tree_parser::get_timer_id(string expr) const
{
    size_t sub_end = expr.find("=");
    string expr_wout_spaces = replace_all(expr, string(" "), string(""));
    int index = 0;

    while (true)
    {
        if (index == expr.length())
        {
            return expr.length();
        }
        
        if (in_array(expr_wout_spaces[++index], {'<','>','='}))
        {
            break;
        }
    }

    string sub = expr_wout_spaces.substr(0, index);
    
    return timers_map_.count(sub) == 1 ?  timers_map_.at(sub) : throw "sum tin wong";
}

list<string> split_expr(string expr)
{
    list<string> result;
    std::stringstream test(expr);
    std::string segment;
    while(std::getline(test, segment, '&'))
    {
        string s = replace_all(segment, "&", "");
        result.push_back(s);
    }
    return result;
}

void uppaal_tree_parser::init_lists(xml_document* doc)
{
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        string decl = templates.child("declaration").child_value();
        
        size_t clock_start = decl.find("clock");
        
        size_t clock_end = decl.find(";");
        
        //string vars = decl.substr(clock_start, clock_end-1);
        string clocks = decl.substr(clock_start+5,clock_end-1);
        clocks = replace_all(clocks, string(" "), string(""));

        std::stringstream clocks_stream(clocks);
        std::string clock;
        
        int var_amount = 0;
        while(std::getline(clocks_stream, clock, ','))
        {
            string clock_without = clock;
            if (clock.find(';') != std::string::npos)
                clock_without = replace_all(clock, ";", "");
            timers_map_.insert_or_assign(clock_without,var_amount++);
        }

        timer_list_ = static_cast<timer_t*>(malloc(sizeof(timer_t) * var_amount));
        timer_amount_ = var_amount;
        for (int i = 0; i < var_amount; i++)
        {
            timer_list_[i] = timer_t(i, 0);
        }
        
        for (pugi::xml_node locs: templates.children("location"))
        {
            edge_list_.push_back(list<edge_t>());
            invariance_list_.push_back(list<constraint_t>());
        }

        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            string string_id = locs.attribute("id").as_string();
            branchpoint_nodes.push_back(xml_id_to_int(string_id));
            edge_list_.push_back(list<edge_t>());
            invariance_list_.push_back(list<constraint_t>());
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            guard_list_.push_back(list<guard_d>());
            update_list_.push_back(list<update_t>());
        }
    }
}

uppaal_tree_parser::uppaal_tree_parser()
{
}

node_t* uppaal_tree_parser::get_node(int target_id)
{
    for(auto node: nodes_)
    {
        if(node.get_id() == target_id)
            return &node;
    }
    return &nodes_.front();
}

template <typename T> void insert_into_list(list<list<T>>* t_list, int index, T item)
{
    auto l_front = t_list->begin();
    std::advance(l_front, index);
    l_front->emplace_back(item);
}


__host__ stochastic_model_t uppaal_tree_parser::parse_xml(char* file_path)
{
    string path = file_path;
    xml_document doc;

    map<int, list<edge_t>> node_edge_map;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        throw "!";
    }
    
    list<node_t> nodes__;
    int egde_id_ = 0;
    int invariant_id = 0;

    init_lists(&doc);
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            string string_name = locs.child("name").child_value();
            const int node_id = xml_id_to_int(string_id);
            bool is_goal = false;
            
            list<constraint_t> invariants;
            
            if (string_name == "Goal")
                is_goal = true;
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr_string = locs.child("label").child_value();

            list<string> exprs = split_expr(expr_string);
            if (kind == "invariant")
            {
                for(auto expr: exprs)
                {
                    invariants.push_back(get_constraint(expr, get_timer_id(expr), get_expr_value_float(expr)));
                    insert_into_list(&invariance_list_, node_id, get_constraint(expr, get_timer_id(expr), get_expr_value_float(expr)));
                }
            }
            nodes_.push_back(node_t(node_id, false, list_to_arr(invariants), is_goal));
        }

        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            string string_id = locs.attribute("id").as_string();
            const int node_id = xml_id_to_int(string_id);

            nodes_.push_back(node_t(node_id, true));
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            string source = trans.child("source").attribute("ref").as_string();
            string target = trans.child("target").attribute("ref").as_string();

            int source_id = xml_id_to_int(source);
            int target_id = xml_id_to_int(target);
            
            list<constraint_t> guards;
            list<update_t> updates;
            int update_id = 0;
            int edge_id = 0;
            float probability = 1.0f;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr_string = labels.child_value();

                list<string> exprs = split_expr(expr_string);
                if(kind == "guard")
                {
                    for(auto expr: exprs)
                    {
                        if (expr == "")
                            continue;
                        guards.push_back(get_constraint(expr,get_timer_id(expr), get_expr_value_float(expr)));
                        //insert_into_list(&guard_list_, egde_id_, constraint_t(get_timer_id(expr),get_expr_enum(expr),get_expr_value(expr)));
                        
                    }
                }
                else if (kind == "assignment")
                {
                    for(auto expr: exprs)
                    {
                        //updates.push_back(update_t(get_timer_id(expr), get_expr_value(expr)));
                        insert_into_list(&update_list_, egde_id_, update_t(update_id++, get_timer_id(expr),get_expr_value(expr)));
                    }
                }
                else if (kind == "probability")
                {
                    probability = get_expr_value_float(expr_string);
                }
            }
            
            node_t* target_node = get_node(target_id);
            node_edge_map.at(source_id).push_back(edge_t(edge_id++, probability, target_node, list_to_arr(guards)));
            
        }
    }
    
    // auto edges = new uneven_list<edge_d>(&edge_list_, edge_list_.size());
    // auto invariants = new uneven_list<guard_d>(&invariance_list_, invariance_list_.size());
    // auto guards = new uneven_list<guard_d>(&guard_list_, guard_list_.size());
    // auto updates = new uneven_list<update_d>(&update_list_, update_list_.size());
    // list<guard_d> guard_0_;

    for(auto node: nodes_)
    {
        node.set_edges(&node_edge_map.at(node.get_id()));
    }

    //print_2_guard(guard_list_);
    
    //int* branchpoint_nodes_arr = list_to_arr(branchpoint_nodes);
    
    return stochastic_model_t(get_node(init_node_id_), new array_t<timer_t>(timer_list_, timer_amount_));
}

