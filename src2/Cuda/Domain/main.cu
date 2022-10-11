
#include "common.h"
#include <iostream>
#include "pretty_visitor.h"
#include "../UPPAALTreeParser/uppaal_tree_parser.h"
// #include "../Simulator/cuda_simulator.h"
// #include "../Simulator/cpu_simulator.h"
#include "../Simulator/simulation_strategy.h"
#include "../Simulator/stochastic_simulator.h"

int main(int argc, char* argv[])
{
    std::cout << "Fuck you\n";

    constraint_t* con0 = constraint_t::less_equal_v(0, 10.0f);
    // constraint_t con1 = constraint_t::less_equal_v(1, 10.0f);
    // constraint_t con2 = constraint_t::greater_equal_v(0, 0.0f);

    array_t<constraint_t*> con0_arr = array_t<constraint_t*>(1);
    con0_arr.arr()[0] = con0;
    node_t node0 = node_t(0, con0_arr, false,false);
    node_t node1     = node_t(1, con0_arr, false,false);
    node_t node2 = node_t(2, array_t<constraint_t*>(0),false,true);

    edge_t* edge0_1 = new edge_t(0, 1, &node1, array_t<constraint_t*>(0));
    edge_t* edge0_2 = new edge_t(1, 1, &node2, array_t<constraint_t*>(0));
    edge_t* edge1_0 = new edge_t(2, 1, &node0, array_t<constraint_t*>(0));

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
        model = parser.parse(argv[1]);
    }
    visitor.visit(&model);
    simulation_strategy strategy = {32, 512, 1000, 1, 1000};

    stochastic_simulator::simulate_cpu(&model, &strategy);
    stochastic_simulator::simulate_gpu(&model, &strategy);
    // cuda_simulator::simulate(&model, &strategy);
    
    std::cout << "pully porky\n";

    return 0;
}
