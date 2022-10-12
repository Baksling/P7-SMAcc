#pragma once
#include <list>
#include <string>

#include "declaration.h"

class declaration_parser
{
public:
    static std::list<declaration> parse(std::string declaration);
};
