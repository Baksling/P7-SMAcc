#pragma once
#include "../visitors/domain_optimization_visitor.h"
#include <iostream>
#include <fstream>
#include <nvrtc.h>
#include "jitify.hpp"

#include "../common/sim_config.h"
#include "../allocations/memory_allocator.h"
#include "../results/result_store.h"


#define KERNAL404 "kernal.cu file not found. Please run file_assembler.py from analysis folder and place the 'kernal.cu' file in the same folder as executable." 
#define NVRTC_SAFE_CALL(x)                                        \
do {                                                            \
nvrtcResult result = x;                                       \
if (result != NVRTC_SUCCESS) {                                \
std::cerr << "\n NVRTC failed, error: " << result << ' '          \
<< nvrtcGetErrorString(result) << '\n';           \
exit(1);                                                    \
}                                                             \
} while(0)

#define CU_CHECK(x)                                         \
do {                                                            \
CUresult result = x;                                          \
if (result != CUDA_SUCCESS) {                                 \
const char *msg;                                            \
cuGetErrorName(result, &msg);                               \
std::cerr << "\nerror: " #x " failed with error "           \
<< msg << '\n';                                   \
exit(1);                                                    \
}                                                             \
} while(0)
#define JIT_EXPR_LOCATION "//__SEARCH_TEXT_FOR_JIT__"


#ifndef CUDA_INC_DIR
#define CUDA_INC_DIR "/usr/local/cuda/include/"
#endif


inline std::istream* fallback(const std::string filename, std::iostream& tmp_stream)
{
    std::ifstream file(CUDA_INC_DIR + filename);
    tmp_stream << file.rdbuf() << std::endl;
    file.close();
    
    return &tmp_stream;
}

class jit_compiler
{
public:
    static jitify::KernelInstantiation compile(expr_compiler_visitor* expr_compiler)
    {
        if(expr_compiler == nullptr) throw std::runtime_error("expr visitor is nullptr"); 
        std::ifstream file("kernal.cu");
        if(file.fail()) throw std::runtime_error(KERNAL404);
        std::stringstream buffer;
        buffer << "kernal.cu\n" << file.rdbuf();
        file.close();
        std::string jit_content = buffer.str();
        std::string jit_marker = JIT_EXPR_LOCATION;
        size_t start_pos = jit_content.find(jit_marker);
        if(start_pos == std::string::npos)
            throw std::runtime_error("Could not find the JIT EXPR LOCATION marker in source");
        std::stringstream& compiled_expr = expr_compiler->get_compiled_expressions();
        jit_content = jit_content.replace(start_pos, jit_marker.length(), compiled_expr.str());

        {
            std::ofstream outfile("./JIT.cu", std::ofstream::out);
            outfile << jit_content;
            outfile.flush();
            outfile.close();
        }
        
        try
        {
            static jitify::JitCache kernel_cache;
            jitify::Program program = kernel_cache.program(jit_content,
                {},
                {"--use_fast_math", "-I " CUDA_INC_DIR, "--dopt=on" },
                fallback);
            return program
            .kernel("simulator_gpu_kernel")
            .instantiate();
        }
        catch(std::runtime_error&)
        {
            std::ofstream outfile("./JIT.cu", std::ofstream::out);
            outfile << jit_content;
            outfile.flush();
            outfile.close();
            
            std::cout << "runtime compilation error encountered. Source dumped as './JIT.cu'. Errors: " << std::endl;

            throw;
        }
    }
};
