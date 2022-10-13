#include <iostream>
#include "declaration_parser.h"
#include "declaration.h"
#include <fstream>
#include <iostream>
#include <sstream>

string read_from_sample_file(string input)
{
    std::ifstream inFile;
    inFile.open(input); //open the input file

    std::stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    std::string str = strStream.str(); //str holds the content of the file

    return str;
}

int main(int argc, const char* argv[])
{
    
    string test = read_from_sample_file("test.txt");

    try
    {
        list<declaration> decls = declaration_parser().parse(test);
        for (declaration d : decls)
        {
            d.to_string();
        }
    }
    catch (const std::runtime_error &ex)
    {
        cout << "Parse error: " << ex.what() << "\n";
        throw runtime_error("parse error");
    }
    

    

    return 0;
}


