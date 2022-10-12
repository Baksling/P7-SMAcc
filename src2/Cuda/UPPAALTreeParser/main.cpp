#include <iostream>

#include "declaration_parser.h"

int main(int argc, const char* argv[])
{
    list<declaration> decls = declaration_parser().parse("");

    for (declaration d : decls)
    {
        d.to_string();
    }

    return 0;
}
