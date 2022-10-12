#include "declaration_parser.h"

#include <list>
#include <sstream>
#include <string>



using namespace std;

list<declaration> declaration_parser::parse(string declaration)
{
    std::stringstream clocks_stream(declaration);
    std::string clock;

    while(std::getline(clocks_stream, clock, '\n'))
    {
        cout << clock;
    }
}
