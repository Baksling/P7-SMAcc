#include "common.h"
#include <iostream>
#include "pretty_visitor.h"
#include "../UPPAALTreeParser/uppaal_tree_parser.h"

int main(int argc, char* argv[])
{
    std::cout << "Fuck you\n";

    constraint_t con0 = constraint_t::less_equal_v(0, 10.0f);
    constraint_t con1 = constraint_t::less_equal_v(1, 10.0f);
    constraint_t con2 = constraint_t::greater_equal_v(0, 0.0f);
    
    node_t node0 = node_t(0, false, &con0,false);
    node_t node1 = node_t(1, false, nullptr,true);
    node_t node2 = node_t(2, false, nullptr,false);

    edge_t* edge0_1 = new edge_t(0, 1, &node1, &con1);
    edge_t* edge0_2 = new edge_t(1, 1, &node2, &con2);
    edge_t* edge1_0 = new edge_t(2, 1, &node0, nullptr);

    std::list<edge_t*> node0_lst;
    std::list<edge_t*> node1_lst;
    
    node0_lst.push_back(edge0_1);
    node0_lst.push_back(edge0_2);
    node0.set_edges(&node0_lst);

    node1_lst.push_back(edge1_0);
    node1.set_edges(&node1_lst);


    pretty_visitor visitor;
    if (argc > 1)
    {
        uppaal_tree_parser parser;
        stochastic_model_t model = parser.parse_xml(argv[1]);
        visitor.visit(&model);
    }
    else
    {
        stochastic_model_t model(&node0, nullptr);
        visitor.visit(&model);
    }
    
    std::cout << "bacon\n";

    return 0;
}
