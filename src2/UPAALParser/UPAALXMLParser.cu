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
#include <string>
#include <vector>
#include <sstream>

#define GPU __device__
#define CPU __host__

using namespace std;
using namespace pugi;

UPAALXMLParser::UPAALXMLParser() = default;

logical_operator get_expr_enum(const string& expr)
{

    if(expr.find("<=") != std::string::npos)
        return logical_operator::less_equal;
    if(expr.find(">=") != std::string::npos)
        return logical_operator::greater_equal;
    if(expr.find("==") != std::string::npos)
        return logical_operator::equal;
    if(expr.find('<') != std::string::npos)
        return logical_operator::less;
    if(expr.find('>') != std::string::npos)
        return logical_operator::greater;

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
            //cout << "\n" << "AAAAA: " << expr_wout_spaces << "\n"; 
            //cout << "HERE: " << index << " ::: " << expr_wout_spaces.substr(index+1) << " \n";
            //cout.flush();
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
            cout << index << " ::: " << expr.substr(index+1) << " \n";
            cout.flush();
            return stof(expr.substr(index+1));
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
    cout << id_string;
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




int UPAALXMLParser::get_timer_id(string expr) const
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

    for ( const auto &myPair : timers_map_ ) {
        std::cout <<"KEY:"<< myPair.first << ":\n";
    }
    
    //cout << "::::::"<<index<<"::::::TImEr ID:" << sub << "|\n";
    //cout << "CLOCKO::" << (timers_map_.count(sub) == 1 ?  timers_map_.at(sub) : -1) << "\n";
    return timers_map_.count(sub) == 1 ?  timers_map_.at(sub) : throw "sum ting wrong";
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

void UPAALXMLParser::init_lists(xml_document* doc)
{
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        string decl = templates.child("declaration").child_value();
        
        size_t clock_start = decl.find("clock");
        
        size_t clock_end = decl.find(";");
        
        //string vars = decl.substr(clock_start, clock_end-1);
        string t = decl.substr(clock_start+5,clock_end-1);
        t = replace_all(t, string(" "), string(""));

        std::stringstream test(t);
        std::string segment;
        
        
        
        int var_amount = 0;
        while(std::getline(test, segment, ','))
        {
            if (segment.find(';') != std::string::npos)
                segment = replace_all(segment, ";", "");
            timers_map_.insert_or_assign(segment,var_amount);
            
        }

        timer_list_ = (timer_d*)malloc(sizeof(timer_d) * var_amount+1);
        timer_amount_ = var_amount + 1;
        var_amount = 0;
        while(std::getline(test, segment, ','))
        {
            if (segment.find(';') != std::string::npos)
                segment = replace_all(segment, ";", "");
            timer_list_[var_amount] = timer_d(var_amount++, 0);
            
        }
        //cout << "VARS:" << t << " " << clock_start << " " << clock_end;
        
        
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

template <typename T> void insert_into_list(list<list<T>>* t_list, int index, T item)
{
    auto l_front = t_list->begin();
    std::advance(l_front, index);
    l_front->emplace_back(item);
}


__host__ stochastic_model UPAALXMLParser::parse_xml(char* file_path)
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
    int invariant_id = 0;

    init_lists(&doc);
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            string string_name = locs.child("name").child_value();
            const int node_id = xml_id_to_int(string_id);
            
            if (string_name == "Goal")
                continue;
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr_string = locs.child("label").child_value();

            list<string> exprs = split_expr(expr_string);
            if (kind == "invariant")
            {
                for(auto expr: exprs)
                {
                    cout << "\nA_ASDADSD:" <<expr;
                    insert_into_list(&invariance_list_, node_id, guard_d(get_timer_id(expr),get_expr_enum(expr),get_expr_value(expr),invariant_id++));
                }
                //cout << "\nINVARIANT:::" << invariance_list_.begin()->begin()->get_value() << " " << invariant_id++ << "\n";
            }
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
            float probability = 1.0f;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr_string = labels.child_value();

                list<string> exprs = split_expr(expr_string);
                cout << "KIND:::" << kind<<"|\n";
                if(kind == "guard")
                {
                    for(auto expr: exprs)
                    {
                        if (expr == "")
                            continue;
                        cout << "\nGUARDZ:" <<expr<<"|\n";
                        insert_into_list(&guard_list_, egde_id_, guard_d(get_timer_id(expr),get_expr_enum(expr),get_expr_value(expr)));
                        cout << "\nGINFO-  TIMERID: " << get_timer_id(expr) << "::: TYPE: "<<get_expr_enum(expr)<< "::: VAL: " << get_expr_value(expr);
                    }
                    cout << "\n\n\n";
                }
                else if (kind == "assignment")
                {
                    for(auto expr: exprs)
                    {
                        cout << "\nASS:" <<expr<<"\n";
                        insert_into_list(&update_list_, egde_id_, update_d(get_timer_id(expr),get_expr_value(expr)));
                    }
                }
                else if (kind == "probability")
                {
                    cout << "PROP1:" << probability << "\n";
                    probability = get_expr_value_float(expr_string);
                    cout << "PROP2:" << probability << "\n";
                }
            }
            insert_into_list(&edge_list_, source_id, edge_d(egde_id_++, target_id, probability));
            //cout << "::::::::::::::" << edge_list_.begin()->begin()->get_dest_node() << ":::::::"<< edge_list_.begin()->begin()->get_id()<<":::::\n";
        }
    }
    
    //print1(edge_list.front());
    //print1(edge_list_test.front());
    //print2(edge_list_test);
    //print2(edge_list);

    
    //cout << "::::::::::::::" << edge_list_.begin()->begin()->get_val() << ":::::::"<< edge_list_.begin()->begin()->get_id()<<":::::\n";
    auto edges = uneven_list<edge_d>(&edge_list_, edge_list_.size());
    auto invariants = uneven_list<guard_d>(&invariance_list_, invariance_list_.size());
    auto guards = uneven_list<guard_d>(&guard_list_, guard_list_.size());
    auto updates =uneven_list<update_d>(&update_list_, update_list_.size());
    
    cout << "DONE PARSIN' MAN, GL HF";
    return stochastic_model(&edges, &invariants, &guards, &updates, timer_list_, timer_amount_);
}


