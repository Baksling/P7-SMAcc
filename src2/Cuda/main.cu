
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
#include "Simulator/writers/result_writer.h"
#include "Simulator/writers/result_manager.h"

using namespace argparse;

int main(int argc, const char* argv[])
{
    cudaFree(nullptr); //done to load cuda assembly, in case of dynamic linking 
    simulation_strategy strategy = {};
    
    ArgumentParser parser("supa_pc_strikes_argina.exe/cuda", "Argument parser example");

    parser.add_argument("-a", "--amount", "Number of simulations", false);
    parser.add_argument("-b", "--block", "Number of block", false);
    parser.add_argument("-c", "--count", "number of times to repeat simulations", false);
    parser.add_argument("-d", "--device", "What simulation to run (GPU (0) / CPU (1) / BOTH (2))", false);
    parser.add_argument("-f", "--pretty", "Pretty print output file", false);
    parser.add_argument("-i", "--interval", "Interval to store trace. e.g. 10t = trace every 10 simulation seconds / 10s = every 10 step  (default = 1s). \n Parameter '-s' is used to specify the max number of steps to store.", false);
    parser.add_argument("-m", "--model", "Model xml file path", false);
    parser.add_argument("-o", "--output", "The path to output result file", false);
    parser.add_argument("-p", "--max_time", "Maximum number to progress in time (default=100)", false );
    parser.add_argument("-s", "--steps", "maximum number of steps per simulation", false);
    parser.add_argument("-t", "--threads", "Number of threads", false);
    parser.add_argument("-u", "--cputhread", "The number of threads to use on the CPU", false);
    parser.add_argument("-v", "--verbose", "Enable pretty print of model (print model (0) / silent(1))", false);
    parser.add_argument("-w", "--write", "Write mode \n / c = console summary  \n / f = file summary \n / d = file data dump \n / t = trace \n / l = lite summary \n / m = write model to file \n / r = write hit percentage to file", false);
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
    string o_path = "./"; // std::filesystem::current_path();
    int write_mode = 0; // 0 = file, 1 = console, 2 = both
    bool verbose = true;

    if (parser.exists("a")) strategy.simulations_per_thread = parser.get<unsigned int>("a");
    if (parser.exists("b")) strategy.block_n = parser.get<int>("b");
    if (parser.exists("c")) strategy.simulation_runs = parser.get<unsigned int>("c");
    if (parser.exists("d")) mode = parser.get<int>("d");
    if (parser.exists("p")) strategy.max_time_progression = parser.get<double>("p");
    if (parser.exists("s")) strategy.max_sim_steps = parser.get<unsigned int>("s");
    if (parser.exists("t")) strategy.threads_n = parser.get<int>("t");
    if (parser.exists("u")) strategy.cpu_threads_n = parser.get<unsigned int>("u");
    if (parser.exists("v")) verbose = parser.get<int>("v") == 0;
    if (parser.exists("w")) write_mode = result_writer::parse_mode(parser.get<std::string>("w"));
    if (parser.exists("y")) strategy.use_max_steps = parser.get<int>("y") == 0;
    if (parser.exists("o")) o_path = o_path + parser.get<string>("o");
    else o_path = o_path + "output";

    if(write_mode & trace) //Trace settings, only if trace is enabled
    {
        strategy.trace_settings =  result_manager::parse_interval(parser.exists("i")
            ? parser.get<std::string>("i")
            : "1s");
    }
    else strategy.trace_settings.mode = trace_interval::disabled;

    uppaal_tree_parser tree_parser = uppaal_tree_parser();
    stochastic_model_t model(array_t<node_t*>(0), array_t<clock_variable>(0), array_t<clock_variable>(0));
    
    if (parser.exists("m"))
    {
        string input_file_path = parser.get<string>("m"); 
        model = tree_parser.parse(input_file_path);
    }
    else
    {
        //TODO remove default model
        array_t<clock_variable> variable_arr = array_t<clock_variable>(2);
        variable_arr.arr()[0] = clock_variable(0, 10);
        variable_arr.arr()[1] = clock_variable(1, 5);
        
        array_t<constraint_t> con0_arr = array_t<constraint_t>(1);
        con0_arr.arr()[0] = *constraint_t::less_equal_v(0, expression::literal_expression(10) );

        node_t node0 = node_t(0, con0_arr, false,false);
        node_t node1 = node_t(1, con0_arr, false,false);
        node_t node2 = node_t(2, con0_arr,false,true);

        std::list<update_t> update_lst;
        array_t<update_t> update_arr = to_array(&update_lst);
        
        edge_t edge0_1 = edge_t(0, expression::literal_expression(1), &node1, con0_arr, update_arr);
        edge_t edge0_2 = edge_t(1, expression::literal_expression(1), &node2, array_t<constraint_t>(0), update_arr);
        edge_t edge1_0 = edge_t(2, expression::literal_expression(1), &node0, array_t<constraint_t>(0), update_arr);

        array_t<clock_variable> timer_arr = array_t<clock_variable>(2);
        timer_arr.arr()[0] = clock_variable(0, 0.0);
        timer_arr.arr()[1] = clock_variable(1, 0.0);
        
        std::list<edge_t> node0_lst;
        std::list<edge_t> node1_lst;
        
        node0_lst.push_back(edge0_1);
        node0_lst.push_back(edge0_2);
        node0.set_edges(&node0_lst);

        node1_lst.push_back(edge1_0);
        node1.set_edges(&node1_lst);
        
        array_t<node_t*> start_nodes = array_t<node_t*>(1);
        start_nodes.arr()[0] = &node0;

        model = stochastic_model_t(start_nodes, timer_arr, variable_arr);
    }
    result_writer r_writer = result_writer(&o_path ,strategy,
        model.get_models_count(),
        model.get_variable_count(),
        write_mode);

    if(write_mode & trace || write_mode & model_out)
    {
        r_writer.write_model(tree_parser.get_nodes_with_name(),
            tree_parser.get_subsystems());
    }
    
    //Computers were not meant to speak.
    //You can speak when spoken to.
    if (write_mode & pretty_out || verbose)
    {
        pretty_visitor p_visitor = pretty_visitor(verbose, write_mode & pretty_out, o_path + "_pretty_model.txt");
        p_visitor.visit(&model);
    }

    //0 == GPU, 1 == CPU, 2 == BOTH
    if (mode == 2 || mode == 0)
    {
        if (verbose) cout << "GPU SIMULATIONS STARTED! \n";
        r_writer.clear();
        stochastic_simulator::simulate_gpu(&model, &strategy, &r_writer, verbose);
        if (verbose) cout << "GPU SIMULATION DONE! \n";
    }
    if (mode > 0)
    {
        if (verbose) cout << "CPU SIMULATION STARTED! \n";
        r_writer.clear();
        stochastic_simulator::simulate_cpu(&model, &strategy, &r_writer, verbose);
        if(verbose) cout << "CPU SIMULATION DONE! \n";
    }
    return 0;
}
