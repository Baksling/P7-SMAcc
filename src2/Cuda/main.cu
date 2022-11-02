
#include <iostream>
#include <filesystem>
#include "Visitors/domain_analysis_visitor.h"
#include "Visitors/pretty_visitor.h"
#include "UPPAALTreeParser/uppaal_tree_parser.h"
#include "Simulator/simulation_strategy.h"
#include "Simulator/stochastic_simulator.h"
#include "common/argparser.h"

#include "Domain/edge_t.h"
#include "Simulator/result_writer.h"

using namespace argparse;


int main(int argc, const char* argv[])
{
    cudaFree(nullptr); //done to load cuda assembly, in case of dynamic linking 
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
    parser.add_argument("-w", "--write", "Write to file (0) / console (1) / both (2)", false);
    parser.add_argument("-o", "--output", "The path to output result file", false);
    parser.add_argument("-y", "--max", "Use max steps or time for limit simulation. (max steps (0) / max time (1) )", false);
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
    string o_path = std::filesystem::current_path();

    int write_mode = -1; // 0 = file, 1 = console, 2 = both


    if (parser.exists("b")) strategy.block_n = parser.get<int>("b");
    if (parser.exists("t")) strategy.threads_n = parser.get<int>("t");
    if (parser.exists("a")) strategy.simulations_per_thread = parser.get<unsigned int>("a");
    if (parser.exists("c")) strategy.simulation_runs = parser.get<int>("c");
    if (parser.exists("s")) strategy.max_sim_steps = parser.get<unsigned int>("s");
    if (parser.exists("p")) strategy.max_time_progression = parser.get<double>("p");
    if (parser.exists("u")) strategy.cpu_threads_n = parser.get<unsigned int>("u");
    if (parser.exists("d")) mode = parser.get<int>("d");
    if (parser.exists("o")) o_path = o_path + parser.get<string>("o");
    if (parser.exists("w")) write_mode = parser.get<int>("w");
    if (parser.exists("y")) strategy.use_max_steps = parser.get<int>("y") == 0;
    
    
    std::cout << "Fuck you\n";

    array_t<clock_variable> variable_arr = array_t<clock_variable>(2);
    variable_arr.arr()[0] = clock_variable(0, 10);
    variable_arr.arr()[1] = clock_variable(1, 5);
    
    array_t<constraint_t*> con0_arr = array_t<constraint_t*>(1);
    con0_arr.arr()[0] = constraint_t::less_equal_v(0, expression::literal_expression(10) );

    
    node_t node0 = node_t(0, con0_arr, false,false);
    node_t node1 = node_t(1, con0_arr, false,false);
    node_t node2 = node_t(2, con0_arr,false,true);

    expression* exp1 = expression::plus_expression(expression::variable_expression(0), expression::variable_expression(1));
    expression* exp2 = expression::minus_expression(expression::variable_expression(1), expression::literal_expression(4));

    std::list<update_t*> update_lst;

    // update_t update1 = update_t(0, 0, false, exp1);
    // update_t update2 = update_t(1, 1, false, exp2);
    //
    // update_lst.push_back(&update1);
    // update_lst.push_back(&update2);

    array_t<update_t*> update_arr = to_array(&update_lst);
    
    edge_t* edge0_1 = new edge_t(0, expression::literal_expression(1), &node1, con0_arr, update_arr);
    edge_t* edge0_2 = new edge_t(1, expression::literal_expression(1), &node2, array_t<constraint_t*>(0), update_arr);
    edge_t* edge1_0 = new edge_t(2, expression::literal_expression(1), &node0, array_t<constraint_t*>(0), update_arr);

    array_t<clock_variable> timer_arr = array_t<clock_variable>(2);
    timer_arr.arr()[0] = clock_variable(0, 0.0);
    timer_arr.arr()[1] = clock_variable(1, 0.0);
    
    std::list<edge_t*> node0_lst;
    std::list<edge_t*> node1_lst;
    
    node0_lst.push_back(edge0_1);
    node0_lst.push_back(edge0_2);
    node0.set_edges(&node0_lst);

    node1_lst.push_back(edge1_0);
    node1.set_edges(&node1_lst);
    
    array_t<node_t> start_nodes = array_t<node_t>(1);
    start_nodes.arr()[0] = node0;

    pretty_visitor p_visitor;
    domain_analysis_visitor d_visitor;

    cout << write_mode % 2 << "HELELELELLELELELELLLLLLLOOOOOOOOOOOOO";
    // 0 = file, 1 = console, 2 = both

    cout << start_nodes.size() << "habahbababahbahba \n";
    
    result_writer r_writer = result_writer(&o_path ,strategy, start_nodes.size(), write_mode > 0, write_mode % 2 == 0);
    
    stochastic_model_t model(start_nodes, timer_arr, variable_arr, 5);
    if (parser.exists("m"))
    {
        uppaal_tree_parser tree_parser;
        string temp = parser.get<string>("m"); 
        char* writeable = new char[temp.size() + 1]; //TODO Move this fuckery inside parser
        std::copy(temp.begin(), temp.end(), writeable);
        writeable[temp.size()] = '\0';
        
        printf("Fuck this");
        model = tree_parser.parse(writeable);

        delete[] writeable;
    }
    p_visitor.visit(&model);
    d_visitor.visit(&model);
    printf("Max exp: %d | Max updates: %d\n", d_visitor.get_max_expression_depth(), d_visitor.get_max_update_width());

    if (mode == 2 || mode == 0)
    {
        cout << "GPU SIMULATIONS STARTED! \n";
        stochastic_simulator::simulate_gpu(&model, &strategy, &r_writer);
        cout << "GPU SIMULATION DONE! \n";
    }
    if (mode > 0)
    {
        cout << "CPU SIMULATION STARTED! \n";
        stochastic_simulator::simulate_cpu(&model, &strategy, &r_writer);
        cout << "CPU SIMULATION DONE! \n";
    }
    
    std::cout << "pully porky\n";

    return 0;
}
