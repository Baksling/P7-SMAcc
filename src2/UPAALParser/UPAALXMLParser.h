#pragma once

#ifndef UPAALXMLParser_H
#define UPAALXMLParser_H

#include "../Timer.h"
#include "../Node.h"
#include <map>

class UPAALXMLParser
{
private:
    int init_node_id_;
    map<unsigned int, node>* nodes_;
public:
    UPAALXMLParser();
    node parse_xml(timer* t, char* file_path);
};
#endif