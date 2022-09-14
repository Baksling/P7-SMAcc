#include "UPAALXMLParser.h"
#include <iostream>
#include "pugixml.hpp"
#include <list>
#include "../Node.h"
#include "../Edge.h"
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

int xml_id_to_int(string id_string)
{
    return stoi(id_string.replace(0,2,""));
}

node UPAALXMLParser::parse_xml(timer* t, char* file_path)
{
    string path = file_path;
    cout << "\nParsing XML data ("+path+").....\n\n";
    
    xml_document doc;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        throw "No XML file, sad..";
    }
    
    nodes_ = new map<unsigned int, node>;
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            const int node_id = xml_id_to_int(string_id);

            node n(node_id);
            
            if (node_id == 1)
                n = node(node_id,true);
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr = locs.child("label").child_value();

            if (kind == "invariant")
            {
                n.add_invariant(get_expr_enum(expr),get_expr_value(expr),t);
            }
            nodes_->insert(pair<unsigned int,node>(node_id, n));
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            string source = trans.child("source").attribute("ref").as_string();
            string target = trans.child("target").attribute("ref").as_string();

            int source_id = xml_id_to_int(source);
            int target_id = xml_id_to_int(target);
            
            list<guard> guards;
            auto edge_updates = new list<update>;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr = labels.child_value();
                
                if(kind == "guard")
                    guards.emplace_back(get_expr_enum(expr),get_expr_value(expr),t);
                else if (kind == "assignment")
                    edge_updates->emplace_back(t, get_expr_value(expr));
            }
            
            nodes_->at(source_id).add_edge(&nodes_->at(target_id), guards, edge_updates);
        }
    }
   
    return nodes_->at(init_node_id_);
}


