
#include "common.h"
#include <iostream>
#include "pretty_visitor.h"
#include "domain_analysis_visitor.h"
#include "../UPPAALTreeParser/uppaal_tree_parser.h"

#include "../Simulator/simulation_strategy.h"
#include "../Simulator/stochastic_simulator.h"
#include "argparser.h"

using namespace argparse;

int main(int argc, const char* argv[])
{
    simulation_strategy strategy = {};
    
    ArgumentParser parser("supa_pc_strikes_argina.exe/cuda", "Argument parser example");

    parser.add_argument("-m", "--model", "Model xml file path", false);
    parser.add_argument("-b", "--block", "Number of block", false);
    parser.add_argument("-t", "--threads", "Number of threads", false);
    parser.add_argument("-a", "--amount", "Number of simulations", false);
    parser.add_argument("-c", "--count", "number of times to repeat simulations", false);
    parser.add_argument("-s", "--steps", "maximum number of steps per simulation", false);
    parser.add_argument("-p", "--maxtime", "Maximum number to progress in time (default=100)", false );
    parser.add_argument("-d", "--device", "What simulation to run (GPU (0) / CPU (1) / BOTH (2))", false);
    parser.add_argument("-u", "--cputhread", "The number of threads to use on the CPU", false);
    parser.enable_help();
    auto err = parser.parse(argc, argv);
    
    if (err) {
        std::cout << err << std::endl;
        return -1;
    }

    if (parser.exists("help")) {
        parser.print_help();
        return 0;
    }

    int mode = 0; // 0 = GPU, 1 = CPU, 2 = BOTH


    if (parser.exists("b")) strategy.block_n = parser.get<int>("b");
    if (parser.exists("t")) strategy.threads_n = parser.get<int>("t");
    if (parser.exists("a")) strategy.simulation_amounts = parser.get<unsigned int>("a");
    if (parser.exists("c")) strategy.sim_count = parser.get<int>("c");
    if (parser.exists("s")) strategy.max_sim_steps = parser.get<unsigned int>("s");
    if (parser.exists("p")) strategy.max_time_progression = parser.get<double>("p");
    if (parser.exists("u")) strategy.cpu_threads_n = parser.get<unsigned int>("u");
    if (parser.exists("d")) mode = parser.get<int>("d");
    
    
    std::cout << "Fuck you\n";

    constraint_t* con0 = constraint_t::less_equal_v(0, 10.0f);
    constraint_t* con1 = constraint_t::greater_equal_t(0, 10.0f);
    // constraint_t con1 = constraint_t::less_equal_v(1, 10.0f);
    // constraint_t con2 = constraint_t::greater_equal_v(0, 0.0f);

    array_t<constraint_t*> con0_arr = array_t<constraint_t*>(1);
    array_t<constraint_t*> con1_arr = array_t<constraint_t*>(1);

    con0_arr.arr()[0] = con0;
    con1_arr.arr()[0] = con1;
    node_t node0 = node_t(0, con0_arr, false,false);
    node_t node1 = node_t(1, con0_arr, false,false);
    node_t node2 = node_t(2, array_t<constraint_t*>(0),false,true);

    update_expression* exp1 = update_expression::plus_expression(update_expression::literal_expression(3), update_expression::literal_expression(4));
    update_expression* exp2 = update_expression::minus_expression(update_expression::literal_expression(6), update_expression::literal_expression(4));

    std::list<update_t*> update_lst;

    update_t update1 = update_t(0, 0, true, exp1);
    update_t update2 = update_t(0, 1, true, exp2);

    printf("Depth of update1 expression: %d\n", update_t::get_expression_depth(update1.get_expression_root()));
    
    update_lst.push_back(&update1);
    update_lst.push_back(&update2);

    array_t<update_t*> update_arr = to_array(&update_lst);
    
    edge_t* edge0_1 = new edge_t(0, 1, &node1, con1_arr, update_arr);
    edge_t* edge0_2 = new edge_t(1, 1, &node2, array_t<constraint_t*>(0), update_arr);
    edge_t* edge1_0 = new edge_t(2, 1, &node0, array_t<constraint_t*>(0), update_arr);

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

    array_t<system_variable*> variable_arr = array_t<system_variable*>(0);


    pretty_visitor p_visitor;
    domain_analysis_visitor d_visitor;
    
    stochastic_model_t model(&node0, to_array(&clock_lst), variable_arr);
    if (parser.exists("m"))
    {
        printf("USING PARSER\n");
        uppaal_tree_parser tree_parser;
        string temp = parser.get<string>("m");
        char* writeable = new char[temp.size() + 1];
        std::copy(temp.begin(), temp.end(), writeable);
        writeable[temp.size()] = '\0';
        
        printf("Fuck thi");
        model = tree_parser.parse(writeable);

        delete[] writeable;
    }
    p_visitor.visit(&model);
    d_visitor.visit(&model);
    auto [max_expression, max_updates] = d_visitor.get_results();
    printf("Max exp: %d | Max updates: %d\n", max_expression, max_updates);
    if (mode == 2 || mode == 0)
    {
        cout << "GPU SIMULATIONS STARTED! \n";
        stochastic_simulator::simulate_gpu(&model, &strategy);
        cout << "GPU SIMULATION DONE! \n";
    }
    if (mode > 0)
    {
        cout << "CPU SIMULATION STARTED! \n";
        stochastic_simulator::simulate_cpu(&model, &strategy);
        cout << "CPU SIMULATION DONE! \n";
    }
    
    std::cout << "pully porky\n";

    return 0;
}
