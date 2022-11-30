
#include <string>

#include "./allocations/memory_allocator.h"
#include "./allocations/cuda_allocator.h"
#include "macro.h"
#include "automata_engine.cu"
#include "../UPPAALXMLParser/uppaal_xml_parser.h"
#include "./results/output_writer.h"
#include "sim_config.h"
#include "visitors/pretty_print_visitor.h"
#include "./allocations/argparser.h"
#include "visitors/domain_optimization_visitor.h"


class arg_exception : public std::runtime_error
{
public:
    std::string msg_;

    arg_exception(const arg_exception& ex) : runtime_error(ex.msg_) { msg_ = ex.msg_; }

    explicit arg_exception(const char arg_name, const std::string& msg) :  runtime_error(msg)
    {
        std::ostringstream o;
        o << "error on arg '" << arg_name << "', message: " << msg;
        msg_ = o.str();
    }

    ~arg_exception() throw() {}
    const char* what() const throw() {
        return msg_.c_str();
    }
};

sim_config parse_configs(int argc, const char* argv[])
{

    argparse::ArgumentParser parser("Cuda stochastic system simulator", "Argument parser example");

    //model
    parser.add_argument("-m", "--model", "Model xml file path", false);

    //output 
    parser.add_argument("-o", "--output", "The path to output result file without file extension. e.g. './output' ", false);
    parser.add_argument("-w", "--write", "Write mode \n / c = console summary  \n / f = file summary \n / d = file data dump \n / t = trace \n / l = lite summary \n / m = write model to file \n / r = write hit percentage to file", false);
    
    //explicit block/thread specification
    parser.add_argument("-b", "--block", "Specify number of blocks/threads to use.\nIn the format 'blocks,threads' e.g. '32,512' for 32 blocks and 512 threads pr. block", false);

    //number of simulations to run.
    parser.add_argument("-e", "--epsilon", "epsilon value to calculate number of simulations", false);
    parser.add_argument("-a", "--alpha", "Specify number of blocks/threads to use.\nIn the format 'blocks,threads' e.g. '32,512' for 32 blocks and 512 threads pr. block", false);
    parser.add_argument("-n", "--number", "Specify the total number of simulations to run .\nIf this parameter is specified, epsilon and alpha are ignored", false);
    parser.add_argument("-r", "--repeat", "number of times to repeat simulations. Concats results. default = 1", false);

    //device options
    parser.add_argument("-d", "--device", "Where to run simulation. (GPU (0) / CPU (1) / BOTH (2)). default = 0", false);
    parser.add_argument("-c", "--cputhread", "The number of threads to use on the CPU. ", false);

    //simulation options
    parser.add_argument("-x", "--units", "Maximum number of steps or time to simulate. e.g. 100t for 100 time units or 100s for 100 steps", false);

    //other
    parser.add_argument("-v", "--verbose", "Enable pretty print of model (print model (0) / silent(1))", false);
    parser.enable_help();
    auto err = parser.parse(argc, argv);
    
    sim_config config = {};
    config.seed = static_cast<unsigned long long>(time(nullptr));
    size_t total_simulations = 1;
    
    if(parser.exists("m")) config.model_path = parser.get<std::string>("m");
    else throw arg_exception('m', "No model argument supplied");

    if(parser.exists("o")) config.out_path = parser.get<std::string>("o");
    else config.out_path = "./output";

    if(parser.exists("w")) config.write_mode = output_writer::parse_mode(parser.get<std::string>("w"));
    else config.write_mode = 0;

    if(parser.exists("b"))
    {
        if(!uppaal_xml_parser::try_parse_block_threads(
            parser.get<std::string>("b"),
            &config.blocks,
            &config.threads
            ))
                throw arg_exception('b', "could not parse block/threads. format: 'blocks,threads'. e.g. '32,512'");
    }
    else throw arg_exception('b', "no block arg supplied");

    if(parser.exists("n")) total_simulations = parser.get<size_t>("n");
    else if(parser.exists("e") && parser.exists("a"))
    {
        double epsilon = parser.get<double>("e");
        double alpha = parser.get<double>("a");
        total_simulations = static_cast<size_t>(ceil((log(2.0) - log(alpha)) / (2*pow(epsilon, 2))));
    }
    else throw arg_exception('n', "no simulation amount supplied. ");

    if(parser.exists("r")) config.simulation_repetitions = parser.get<unsigned>("r");
    else config.simulation_repetitions = 1;

    if(parser.exists("d")) config.sim_location = static_cast<sim_config::device_opt>(parser.get<int>("d"));
    else config.sim_location = sim_config::device;

    if(parser.exists("c")) config.cpu_threads = parser.get<unsigned>("c");
    else config.cpu_threads = 1;

    if(parser.exists("x"))
    {
        bool is_timer;
        double unit_value = 0.0;
        bool success = uppaal_xml_parser::try_parse_units(parser.get<std::string>("x"), &is_timer, &unit_value);
        if(!success) throw arg_exception('x', "could not parse unit format. e.g. 100t or 100s");
        config.use_max_steps = !is_timer;
        config.max_steps_pr_sim = static_cast<unsigned>(floor(unit_value));
        config.max_global_progression = unit_value;
    }
    else
    {
        config.use_max_steps = true;
        config.max_steps_pr_sim = 100;
        config.max_global_progression = 100;
    }

    if(parser.exists("v")) config.verbose = parser.get<int>("v");
    else config.verbose = true;
    
    config.simulation_amount = static_cast<unsigned>(ceil(
            static_cast<double>(total_simulations) /
            static_cast<double>((config.blocks * config.threads))));
    
    return config;
}

void setup_config(sim_config* config, const automata* model, unsigned max_expr_depth)
{
    unsigned track_count = 0;
    for (int i = 0; i < model->variables.size; ++i)
        if(model->variables.store[i].should_track)
            track_count++;

    config->tracked_variable_count = track_count;
    config->network_size = model->network.size;
    config->variable_count = model->variables.size;
    config->max_expression_depth = max_expr_depth;
}

int main(int argc, const char* argv[])
{
    CUDA_CHECK(cudaFree(nullptr));
    memory_allocator helper = memory_allocator(true);

    sim_config config = parse_configs(argc, argv);
    
    uppaal_xml_parser xml_parser;
    automata model = xml_parser.parse(config.model_path);

    domain_optimization_visitor optimizer = domain_optimization_visitor();
    optimizer.visit(&model);
    
    setup_config(&config, &model, optimizer.get_max_expr_depth());
    
    pretty_print_visitor pretty_visitor = pretty_print_visitor(&std::cout);
    pretty_visitor.visit(&model);
    
    cuda_allocator av = cuda_allocator(&helper);
    const automata* model_d = av.allocate_automata(&model);

    const result_store store = result_store(config.blocks, config.threads, config.simulation_amount,
                                            config.tracked_variable_count,  config.network_size, &helper);
    
    output_writer writer = output_writer(
        &config.out_path,
        config.blocks*config.threads*config.simulation_amount,
        output_writer::parse_mode("c"),
        &model        
        );

    result_store* store_d = nullptr;
    CUDA_CHECK(cudaMalloc(&store_d, sizeof(result_store)));
    CUDA_CHECK(cudaMemcpy(store_d, &store, sizeof(result_store), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(helper.allocate(&config.cache, static_cast<size_t>(config.blocks*config.threads)*thread_heap_size(&config)));
    CUDA_CHECK(helper.allocate(&config.random_state_arr, static_cast<size_t>(config.blocks*config.threads)*sizeof(curandState)));

    sim_config* config_d = nullptr;
    CUDA_CHECK(cudaMalloc(&config_d, sizeof(sim_config)));
    CUDA_CHECK(cudaMemcpy(config_d, &config, sizeof(sim_config), cudaMemcpyHostToDevice));

    printf("pre: %d\n", cudaPeekAtLastError());
    const std::chrono::steady_clock::time_point global_start = std::chrono::steady_clock::now();

    
    
    simulator_gpu_kernel<<<config.blocks, config.threads>>>(model_d, store_d, config_d);
    cudaDeviceSynchronize();
    
    printf("post: %d\n", cudaPeekAtLastError());
    const std::chrono::steady_clock::duration sim_duration = std::chrono::steady_clock::now() - global_start;

    writer.write(&store, sim_duration);
    writer.write_summary(sim_duration);

    helper.free_allocations();
    printf("PULLY PORKY\n");
}
