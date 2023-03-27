#include <string>
#include "simulation_runner.h"

#include "cmath"
#include "UPPAALXMLParser/uppaal_xml_parser.h"

#include "./results/output_writer.h"
#include "./allocations/argparser.h"
#include "common/io_paths.h"
#include "UPPAALXMLParser/abstract_parser.h"
#include "UPPAALXMLParser/smacc_parser.h"

#include "visitors/domain_optimization_visitor.h"
#include "visitors/model_count_visitor.h"
#include "visitors/pretty_print_visitor.h"


enum parser_state
{
    parsed,
    error,
    help
};

parser_state parse_configs(const int argc, const char* argv[], sim_config* config)
{

    argparse::ArgumentParser parser("Cuda stochastic system simulator", "Argument parser example");

    //model and query
    parser.add_argument("-m", "--model", "Model xml file path", false);
    parser.add_argument("-q", "--query", "Check reachability of nodes with names.\n (comma seperated string, eg.: ProcessName0.node0,ProcessName1.node1,ProcessName2.node2)", false);

    //output 
    parser.add_argument("-o", "--output", "The path to output result file without file extension. e.g. './output' ", false);
    parser.add_argument("-w", "--write", "Write mode \n / c = console summary  \n / f = file summary \n / d = file data dump \n / t = trace \n / l = lite summary \n / m = write model to file \n / r = write hit percentage to file", false);
    
    //explicit block/thread specification
    parser.add_argument("-b", "--block", "Specify number of blocks/threads to use.\nIn the format 'blocks,threads' e.g. '32,512' for 32 blocks and 512 threads pr. block", false);

    //number of simulations to run.
    parser.add_argument("-e", "--epsilon", "Probability uncertainty for the results. (default = 0.005)", false);
    parser.add_argument("-a", "--alpha", "Probability of false negatives. Gives the total simulations in combination with epsilon.", false);
    parser.add_argument("-n", "--number", "Specify the total number of simulations to run .\nIf this parameter is specified, epsilon and alpha are ignored", false);
    parser.add_argument("-r", "--repeat", "number of times to repeat simulations. Concats results. default = 1", false);

    //device options
    parser.add_argument("-d", "--device", "Where to run simulation. (GPU (0) / CPU (1) / BOTH (2)). default = 0", false);
    parser.add_argument("-c", "--cputhread", "The number of threads to use on the CPU. ", false);

    //simulation options
    parser.add_argument("-x", "--units", "Maximum number of steps or time to simulate. e.g. 100t for 100 time units or 100s for 100 steps", false);
    parser.add_argument("-s", "--shared", "Attempt to use shared memory in cuda simulation. Will only enable if (threads * 32 > model size)", false);
    parser.add_argument("-j", "--jit", "JIT compile the expressions. Only works for GPU, mutually exclusive with --shared.", false);
    
    
    //other
    parser.add_argument("-v", "--verbose", "Enable pretty print of model (print model (0) / silent(1))", false);
    parser.add_argument("-p", "--print", "Pretty print of model (0 = no print, 1 = before optimisation, 2 = after optimisation). default = 0", false);
    parser.enable_help();

    const auto err = parser.parse(argc, argv);
    if (err) {
        std::cout << err << std::endl;
        return parser_state::error;
    }

    if (parser.exists("help")) {
        parser.print_help();
        return parser_state::help;
    }
    
    config->seed = static_cast<unsigned long long>(time(nullptr));
    
    if(parser.exists("m")) config->paths->model_path = parser.get<std::string>("m");
    else throw argparse::arg_exception('m', "No model argument supplied");

    if(parser.exists("q")) config->paths->query = parser.get<std::string>("q");
    else config->paths->query = "";

    if(parser.exists("o")) config->paths->output_path = parser.get<std::string>("o");
    else config->paths->output_path = "./output";

    if(parser.exists("w")) config->write_mode = output_writer::parse_mode(parser.get<std::string>("w"));
    else config->write_mode = 0;

    if(parser.exists("b"))
    {
        if(!abstract_parser::try_parse_block_threads(
            parser.get<std::string>("b"),
            &config->blocks,
            &config->threads
            ))
                throw argparse::arg_exception('b', "could not parse block/threads. format: 'blocks,threads'. e.g. '32,512'");
    }
    else throw argparse::arg_exception('b', "no block arg supplied");

    double epsilon, alpha;
    size_t total_simulations;
    if(parser.exists("e")) epsilon = parser.get<double>("e");
    else epsilon = 0.005; //default value
    
    if(parser.exists("n"))
    {
        total_simulations = parser.get<size_t>("n");
        alpha = 2 * exp((-2.0) * static_cast<double>(total_simulations) * pow(epsilon,2));
    }
    else if(parser.exists("a"))
    {
        alpha = parser.get<double>("a");
        total_simulations = static_cast<size_t>(ceil((log(2.0) - log(alpha)) / (2*pow(epsilon, 2))));
    }
    else throw argparse::arg_exception('n', "no simulation amount supplied. ");

    if(parser.exists("r")) config->simulation_repetitions = parser.get<unsigned>("r");
    else config->simulation_repetitions = 1;

    if(parser.exists("d")) config->sim_location = static_cast<sim_config::device_opt>(parser.get<int>("d"));
    else config->sim_location = sim_config::device;

    if(parser.exists("c")) config->cpu_threads = parser.get<unsigned>("c");
    else config->cpu_threads = 1;

    if(parser.exists("x"))
    {
        bool is_timer;
        double unit_value = 0.0;
        const bool success = abstract_parser::try_parse_units(parser.get<std::string>("x"), &is_timer, &unit_value);
        if(!success) throw argparse::arg_exception('x', "could not parse unit format. e.g. 100t or 100s");
        config->use_max_steps = !is_timer;
        config->max_steps_pr_sim = static_cast<unsigned>(floor(unit_value));
        config->max_global_progression = unit_value;
    }
    else
    {
        config->use_max_steps = true;
        config->max_steps_pr_sim = 100;
        config->max_global_progression = 100;
    }

    if(parser.exists("v")) config->verbose = parser.get<int>("v");
    else config->verbose = true;

    if(parser.exists("p")) config->model_print_mode = static_cast<sim_config::pretty_print>(parser.get<int>("p"));
    else config->model_print_mode = sim_config::no_print;

    config->use_shared_memory = parser.exists("s");
    config->use_jit = parser.exists("j");
    
    //Insert into config object
    config->simulation_amount = static_cast<unsigned>(ceil(
            static_cast<double>(total_simulations) /
            static_cast<double>((config->blocks * config->threads * config->simulation_repetitions))));
    config->alpha = alpha;
    config->epsilon = epsilon;
    
    return parser_state::parsed;
}

void setup_config(sim_config* config, const network* model,
    const unsigned max_expr_depth,
    const unsigned max_fanout,
    const unsigned node_count)
{
    unsigned track_count = 0;
    for (int i = 0; i < model->variables.size; ++i)
        if(model->variables.store[i].should_track)
            track_count++;

    for (int i = 0; i < model->automatas.size; ++i)
    {
        config->initial_urgent += IS_URGENT(model->automatas.store[i]->type);
        config->initial_committed += model->automatas.store[i]->type == node::committed;
    }

    config->tracked_variable_count = track_count;
    config->network_size = model->automatas.size;
    config->variable_count = model->variables.size;
    config->max_expression_depth = max_expr_depth;
    config->max_edge_fanout = max_fanout;
    config->node_count = node_count;
}

void print_config(const sim_config* config, const size_t model_size)
{
    printf("simulation configuration:\n");
    printf("simulating on model %s\n", config->paths->model_path.c_str());
    printf("running %llu simulations on %d repetitions using parallelism of %d.\n",
        static_cast<unsigned long long>(config->total_simulations()),
        config->simulation_repetitions,
        config->blocks*config->threads);
    printf("Model size: %llu bytes\n", static_cast<unsigned long long>(model_size));
    printf("attempt to use shared memory: %s (possible: %s)\n",
        (config->use_shared_memory ? "Yes" : "No" ),
        (config->can_use_cuda_shared_memory(model_size) ? "Yes" : "No"));
    printf("End criteria: %lf %s\n",
        (config->use_max_steps ? static_cast<double>(config->max_steps_pr_sim) : config->max_global_progression),
        (config->use_max_steps ? "steps" : "time units"));
}


inline bool str_ends_with(const std::string& full_string, const std::string& ending) {
    if (full_string.length() >= ending.length()) {
        return (0 == full_string.compare (full_string.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

abstract_parser* instantiate_parser(const std::string& filepath)
{
    if(str_ends_with(filepath, ".SMAcc"))
    {
        return new smacc_parser();
    }
    else if (str_ends_with(filepath, ".xml"))
    {
        return new uppaal_xml_parser();
    }
    else
    {
        throw std::runtime_error(R"(Could not parse file. Should end in ".SMAcc" or ".xml", but neither was detected)");
    }
}


void run_optimise(network* model, sim_config* config, output_properties* properties)
{
    pretty_print_visitor print_visitor = pretty_print_visitor(&std::cout,
        properties->node_names,
        properties->variable_names);

    if(config->model_print_mode == sim_config::print_model)
        print_visitor.visit(model);

    if(config->verbose) printf("Optimizing...\n");
    domain_optimization_visitor optimizer = domain_optimization_visitor(
        abstract_parser::parse_query(config->paths->query),
        properties->node_network,
        properties->node_names,
        properties->template_names);
    optimizer.optimize(model);

    model_count_visitor count_visitor = model_count_visitor();
    count_visitor.visit(model);

    model_size size_of_model = count_visitor.get_model_size();
    properties->node_map = optimizer.get_node_map();
    
    setup_config(config, model,
        optimizer.get_max_expr_depth(),
        optimizer.get_max_fanout(),
        optimizer.get_node_count());
    
    optimizer.clear();

    if(config->model_print_mode == sim_config::print_reduction)
    {
        print_visitor.clear();
        print_visitor.visit(model);
    }

    if(config->verbose) print_config(config, size_of_model.total_memory_size());
    
    if(config->use_shared_memory) //ensure shared memory is safe
        config->use_shared_memory = config->can_use_cuda_shared_memory(size_of_model.total_memory_size());
        
}

int main(int argc, const char* argv[])
{
    CUDA_CHECK(cudaFree(nullptr));

    io_paths paths = {};
    output_properties properties = {};
    sim_config config = {};
    config.properties = &properties;
    config.paths = &paths;
    parser_state state = parse_configs(argc, argv, &config);
    if(state == error) return -1;
    if(state == help) return 0;
    
    memory_allocator allocator = memory_allocator(
        config.sim_location == sim_config::device || config.sim_location == sim_config::both
        );

    abstract_parser* model_parser = instantiate_parser(config.paths->model_path);
    network model = model_parser->parse(config.paths->model_path);

    properties.node_names = new std::unordered_map<int, std::string>(*model_parser->get_nodes_with_name());
    properties.node_network = new std::unordered_map<int, int>(*model_parser->get_subsystems());
    properties.variable_names = new std::unordered_map<int, std::string>(*model_parser->get_clock_names()); // this can create mem leaks.
    properties.template_names = new std::unordered_map<int, std::string>(*model_parser->get_template_names());

    delete model_parser;
    run_optimise(&model, &config, &properties);

    const bool run_device = (config.sim_location == sim_config::device || config.sim_location == sim_config::both);
    const bool run_host   = (config.sim_location == sim_config::host   || config.sim_location == sim_config::both);
    
    //run simulation
    if(run_device)
    {
        if(config.use_jit)
            simulation_runner::simulate_gpu_jit(&model, &config);
        else
            simulation_runner::simulate_gpu(&model, &config);
    }
    if(run_host)
    {
        if(config.use_jit) throw std::runtime_error("Cannot run simulation on host, when JIT is enabled. Please disable jit(-j) and try again.");
        simulation_runner::simulate_cpu(&model, &config);
    }

    allocator.free_allocations();
    return 0;
}
