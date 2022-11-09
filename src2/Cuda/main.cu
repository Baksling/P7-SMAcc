
#include <iostream>
#include <filesystem>
#include <fstream>

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
    parser.add_argument("-w", "--write", "Write to file (\n / No output (0) \n / Console Summary (1) \n / File summary (2) \n / Console and File summary (3) \n / Console summary and File data (4) \n / File summary and File data (5) \n / Console summary, File summary, and File data (6) \n / Lite summary (7))", false);
    parser.add_argument("-o", "--output", "The path to output result file", false);
    parser.add_argument("-y", "--max", "Use max steps or time for limit simulation. (max steps (0) / max time (1) )", false);
    parser.add_argument("-v", "--verbose", "Enable pretty print of model (print model (0) / silent(1))", false);
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
    writer_modes write_mode; // 0 = file, 1 = console, 2 = both
    bool verbose = true;

    if (parser.exists("b")) strategy.block_n = parser.get<int>("b");
    if (parser.exists("t")) strategy.threads_n = parser.get<int>("t");
    if (parser.exists("a")) strategy.simulations_per_thread = parser.get<unsigned int>("a");
    if (parser.exists("c")) strategy.simulation_runs = parser.get<unsigned int>("c");
    if (parser.exists("s")) strategy.max_sim_steps = parser.get<unsigned int>("s");
    if (parser.exists("p")) strategy.max_time_progression = parser.get<double>("p");
    if (parser.exists("u")) strategy.cpu_threads_n = parser.get<unsigned int>("u");
    if (parser.exists("d")) mode = parser.get<int>("d");
    if (parser.exists("o")) o_path = o_path + "/" + parser.get<string>("o");
    if (parser.exists("w")) write_mode = static_cast<writer_modes>(parser.get<int>("w"));
    if (parser.exists("y")) strategy.use_max_steps = parser.get<int>("y") == 0;
    if (parser.exists("v")) verbose = parser.get<int>("v") == 0;
    
    
    stochastic_model_t model(array_t<node_t>(0), array_t<clock_variable>(0), array_t<clock_variable>(0),  0);
    
    if (parser.exists("m"))
    {
        uppaal_tree_parser tree_parser;
        string temp = parser.get<string>("m"); 
        char* writeable = new char[temp.size() + 1]; //TODO Move this fuckery inside parser
        std::copy(temp.begin(), temp.end(), writeable);
        writeable[temp.size()] = '\0';
        
        model = tree_parser.parse(writeable);

        delete[] writeable;
    }
    else
    {
        //TODO remove default model
        array_t<clock_variable> variable_arr = array_t<clock_variable>(2);
        variable_arr.arr()[0] = clock_variable(0, 10);
        variable_arr.arr()[1] = clock_variable(1, 5);
        
        array_t<constraint_t*> con0_arr = array_t<constraint_t*>(1);
        con0_arr.arr()[0] = constraint_t::less_equal_v(0, expression::literal_expression(10) );

        node_t node0 = node_t(0, con0_arr, false,false);
        node_t node1 = node_t(1, con0_arr, false,false);
        node_t node2 = node_t(2, con0_arr,false,true);

        std::list<update_t*> update_lst;
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

        model = stochastic_model_t(start_nodes, timer_arr, variable_arr, 5);
    }
    result_writer r_writer = result_writer(
        &o_path ,strategy,
        model.get_models_count(),
        model.get_variable_count(),
        write_mode);
    
    //Computers were not meant to speak.
    //You can speak when spoken to.
    if (verbose)
    {
        pretty_visitor p_visitor;
        domain_analysis_visitor d_visitor;
        p_visitor.visit(&model);
        d_visitor.visit(&model);
        printf("Max exp: %d | Max updates: %d\n", d_visitor.get_max_expression_depth(), d_visitor.get_max_update_width());
    }

    //0 == GPU, 1 == CPU, 2 == BOTH
    if (mode == 2 || mode == 0)
    {
        if (verbose) cout << "GPU SIMULATIONS STARTED! \n";
        stochastic_simulator::simulate_gpu(&model, &strategy, &r_writer, verbose);
        if (verbose) cout << "GPU SIMULATION DONE! \n";
    }
    if (mode > 0)
    {
        if (verbose) cout << "CPU SIMULATION STARTED! \n";
        stochastic_simulator::simulate_cpu(&model, &strategy, &r_writer, verbose);
        if(verbose) cout << "CPU SIMULATION DONE! \n";
    }
    return 0;
}
