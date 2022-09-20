#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include "../Cuda/Projekt/node_d.h"
#include "../Cuda/Projekt/edge_d.h"
#include <list>

#include "../Cuda/Projekt/timer_d.h"
#include "../Cuda/Projekt/update_d.h"
#include "../Cuda/Projekt/uneven_list.h"

class guard_d;
using namespace std;

struct parser_output
{
    uneven_list<edge_d> edge;
    uneven_list<guard_d> invariance;
    uneven_list<guard_d> guard;
    uneven_list<update_d> update;
};


class UPAALXMLParser
{
private:
    int init_node_id_;
    list<node_d> nodes_;
    list<list<edge_d>> edge_list_;
    list<list<edge_d>> edge_list_test;
public:
    UPAALXMLParser();
    __host__ parser_output parse_xml(timer_d* t, char* file_path, int goal_node_id = 1);
};
#endif