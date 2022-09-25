#include "UPAALXMLParser.h"
#include <iostream>
#include "pugixml.hpp"
#include <list>
#include "../Cuda/Projekt/node_d.h"
#include "../Cuda/Projekt/update_d.h"
#include "../Cuda/Projekt/guard_d.h"
#include "../Cuda/Projekt/uneven_list.h"
#include "../Cuda/Projekt/timer_d.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <algorithm>

#define GPU __device__
#define CPU __host__

using namespace std;
using namespace pugi;

UPAALXMLParser::UPAALXMLParser() = default;

logical_operator get_expr_enum(const string& expr)
{

    if(expr.find('<') != std::string::npos)
        return logical_operator::less;
    if(expr.find('>') != std::string::npos)
        return logical_operator::greater;
    if(expr.find("<=") != std::string::npos)
        return logical_operator::less_equal;
    if(expr.find(">=") != std::string::npos)
        return logical_operator::greater_equal;
    if(expr.find("==") != std::string::npos)
        return logical_operator::equal;

    throw "Operand in " + expr + " not found, sad..";
}

bool in_array(const char &value, const std::vector<char> &array)
{
    return std::find(array.begin(), array.end(), value) != array.end();
}

int get_expr_value(const string& expr)
{
    int index = expr.length();
    while (true)
    {
        if (index == 0)
        {
            return 0;
        }
        
        if (in_array(expr[--index], {'=',' ','<','>'}))
        {
            cout << index << " " << expr.substr(index+1) << " \n";
            cout.flush();
            return stoi(expr.substr(index+1));
        }
    }
}

template <typename T> auto list_to_arr(list<T> l)
{
    T arr[l.size()];
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


void print1(list<edge_d> const &list)
{
    for (auto i: list) {
        std::cout << i.get_id() << " <-1-> " << i.get_dest_node()  << std::endl;
    }
    cout << "---------------------";
}

void print2(list<list<edge_d>> const &list)
{
    for (auto const &i: list) {
        for (auto q: i) {
            std::cout << q.get_id() << " -> " << q.get_dest_node()  << std::endl;
        }
    }
    
}

void UPAALXMLParser::init_lists(xml_document* doc)
{
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        for (pugi::xml_node locs: templates.children("location"))
        {
            edge_list_.push_back(list<edge_d>());
            invariance_list_.push_back(list<guard_d>());
            
        }

        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            edge_list_.push_back(list<edge_d>());
            invariance_list_.push_back(list<guard_d>());
            
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            guard_list_.push_back(list<guard_d>());
            update_list_.push_back(list<update_d>());
        }
    }
}

template <typename T> void insert_into_list(list<list<T>> t_list, int index, T item)
{
    auto l_front = t_list.begin();
    std::advance(l_front, index);
    l_front->emplace_back(item);
}


__host__ parser_output UPAALXMLParser::parse_xml(timer_d* t, char* file_path, int goal_node_id)
{
    string path = file_path;
    cout << "\nParsing XML data ("+path+").....\n\n";
    xml_document doc;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        throw "!";
    }
    
    list<node_d> nodes__;
    int egde_id_ = 0;

    init_lists(&doc);
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            string string_name = locs.child("name").child_value();
            cout << string_name;
            const int node_id = xml_id_to_int(string_id);
            
            if (string_name == "Goal")
                nodes_.emplace_back(node_id,true);
            else
                nodes_.emplace_back(node_id);
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr = locs.child("label").child_value();

            if (kind == "invariant")
                insert_into_list(invariance_list_, node_id, guard_d(t->get_id(),get_expr_enum(expr),get_expr_value(expr)));
            
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            string source = trans.child("source").attribute("ref").as_string();
            string target = trans.child("target").attribute("ref").as_string();

            int source_id = xml_id_to_int(source);
            int target_id = xml_id_to_int(target);
            
            list<guard_d> guards;
            auto edge_updates = new list<update_d>;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr = labels.child_value();
                
                if(kind == "guard")
                    insert_into_list(guard_list_, source_id, guard_d(t->get_id(),get_expr_enum(expr),get_expr_value(expr)));
                else if (kind == "assignment")
                    insert_into_list(update_list_, source_id, update_d(t->get_id(),get_expr_value(expr)));
            }
            
            insert_into_list(edge_list_, source_id, edge_d(egde_id_, target_id));
            egde_id_ = egde_id_+1;
        }
    }

    list<edge_d> edges_1_;
    edges_1_.emplace_back(0, 1);

    list<edge_d> edges_2_;
    edges_2_.emplace_back(1, 2);

    list<edge_d> edges_3_;
    edges_1_.emplace_back(2,0);
    

    list<list<edge_d>> edge_list;
    edge_list.push_back(edges_1_);
    edge_list.push_back(edges_2_);
    edge_list.push_back(edges_3_);

    //print1(edge_list.front());
    //print1(edge_list_test.front());
    //print2(edge_list_test);
    //print2(edge_list);
    
    auto edges = uneven_list<edge_d>(&edge_list_, edge_list_.size());
    auto invariants = uneven_list<guard_d>(&invariance_list_, invariance_list_.size());
    auto guards = uneven_list<guard_d>(&guard_list_, guard_list_.size());
    auto updates =uneven_list<update_d>(&update_list_, update_list_.size());

    parser_output p_output {
        edges,
    invariants,
    guards,
        updates
    };

    return p_output;
}


