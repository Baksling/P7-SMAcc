#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include "../Cuda/Projekt/node_d.h"
#include "../Cuda/Projekt/edge_d.h"
#include <list>
#include "pugixml.hpp"
#include "../Cuda/Projekt/timer_d.h"
#include "../Cuda/Projekt/update_d.h"
#include "../Cuda/Projekt/uneven_list.h"
#include "../Cuda/Projekt/stochastic_model.h"
#include <map>

class guard_d;
using namespace std;

class UPAALXMLParser
{
private:
    int init_node_id_;
    timer_d* timer_list_;
    int timer_amount_ = 0;
    map<string, int> timers_map_;
    list<node_d> nodes_;
    list<list<edge_d>> edge_list_;
    list<list<guard_d>> guard_list_;
    list<list<guard_d>> invariance_list_;
    list<list<update_d>> update_list_;
    int get_timer_id(string expr) const;
    void init_lists(pugi::xml_document* doc);
public:
    UPAALXMLParser();
    __host__ stochastic_model parse_xml(char* file_path);
};
#endif