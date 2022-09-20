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

int get_expr_value(const string& expr)
{
    int index = expr.length() - 1;
    while (true)
    {
        if (index == 0)
        {
            return 0;
        }
        if (expr[index] == ' ')
        {
            return stoi(expr.substr(index));
        }
        index = index - 1;
    }
}

template <typename T> void fill_map(map<int, list<T>>* map_of_t, T t, int source_id)
{
    if (!map_of_t->count(source_id))
    {
        list<T> list_temp;
                
        list_temp.emplace_back(t);
        map_of_t->insert(pair<int, list<T>>(source_id, list_temp));
    }
    else
        map_of_t->at(source_id).emplace_back(t);
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

template <typename T> list<list<T>>* convert_map_to_list_list(map<int, list<T>> in, int* size)
{
    *size = 0;
    typename map<int, list<T>>::iterator it;

    auto result = new list<list<T>>();
    
    for (it = in.begin(); it != in.end(); it++)
    {
        result->emplace_back(it->second);
        *size = *size+1;
    }

    return result;
}


__host__ parser_output UPAALXMLParser::parse_xml(timer_d* t, char* file_path, int goal_node_id)
{
    string path = file_path;
    cout << "\nParsing XML data ("+path+").....\n\n";
    
    xml_document doc;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        throw "No XML file, sad..";
    }
    
    list<node_d> nodes__;
    int egde_id_ = 0;
    auto edge_map_ = map<int, list<edge_d>>();
    auto guard_map_ = map<int, list<guard_d>>();
    auto update_map_ = map<int, list<update_d>>();
    auto invariant_map_ = map<int, list<guard_d>>();
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            const int node_id = xml_id_to_int(string_id);
            
            if (node_id == goal_node_id)
                nodes_.emplace_back(node_id,true);
            else
                nodes_.emplace_back(node_id);
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr = locs.child("label").child_value();

            if (kind == "invariant")
                fill_map(&guard_map_, guard_d(t->get_id(),get_expr_enum(expr),get_expr_value(expr)),node_id);
            
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
                    fill_map(&guard_map_, guard_d(t->get_id(),get_expr_enum(expr),get_expr_value(expr)),source_id);
                else if (kind == "assignment")
                    fill_map(&update_map_, update_d(t->get_id(),get_expr_value(expr)),source_id);
            }
            
            fill_map(&edge_map_, edge_d(target_id,egde_id_), source_id);
            
            egde_id_ = egde_id_+1;
        }
    }
    int index_size_invariant = 0;
    int index_size_guard = 0;
    int index_size_edge = 0;
    int index_size_update = 0;
    parser_output p_output {
    uneven_list<edge_d>(convert_map_to_list_list(edge_map_,&index_size_edge), index_size_edge),
    uneven_list<guard_d>(convert_map_to_list_list(invariant_map_,&index_size_invariant), index_size_invariant),
    uneven_list<guard_d>(convert_map_to_list_list(guard_map_,&index_size_guard), index_size_guard),
    uneven_list<update_d>(convert_map_to_list_list(update_map_,&index_size_update), index_size_update)
    };

    return p_output;
}


