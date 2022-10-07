#include "common.h"
#include <iostream>
#include "pretty_visitor.h"
#include "../UPPAALTreeParser/uppaal_tree_parser.h"
#include "CudaSimulator.h"

int main(int argc, char* argv[])
{
    std::cout << "Fuck you\n";

    constraint_t con0 = constraint_t::less_equal_v(0, 10.0f);
    constraint_t con1 = constraint_t::less_equal_v(1, 10.0f);
    constraint_t con2 = constraint_t::greater_equal_v(0, 0.0f);
    
    node_t node0 = node_t(0, false, &con0,false);
    node_t node1     = node_t(1, false, &con0,false);
    node_t node2 = node_t(2, false, nullptr,true);

    edge_t* edge0_1 = new edge_t(0, 1, &node1, nullptr);
    edge_t* edge0_2 = new edge_t(1, 1, &node2, nullptr);
    edge_t* edge1_0 = new edge_t(2, 1, &node0, nullptr);

    clock_timer_t timer1 = clock_timer_t(0, 0.0);
    clock_timer_t timer2 = clock_timer_t(1, 0.0);

    std::list<clock_timer_t*> clock_lst;
    clock_lst.push_back(&timer1);
    clock_lst.push_back(&timer2);
    
    std::list<edge_t*> node0_lst;
    std::list<edge_t*> node1_lst;
    
    node0_lst.push_back(edge0_1);
    node0_lst.push_back(edge0_2);
    node0.set_edges(&node0_lst);

    node1_lst.push_back(edge1_0);
    node1.set_edges(&node1_lst);


    pretty_visitor visitor;
    stochastic_model_t model(&node0, to_array(&clock_lst));
    if (argc > 1)
    {
        printf("USING PARSER\n");
        uppaal_tree_parser parser;
        model = parser.parse_xml(argv[1]);
    }
    visitor.visit(&model);
    simulation_strategy strategy = {10, 10, 1, 1, 100};
    cuda_simulator::simulate(&model, &strategy);
    
    std::cout << "pully porky\n";

    return 0;
}
