#pragma once
#include "../visitors/domain_optimization_visitor.h"
#include <iostream>
#include <fstream>
#include <nvrtc.h>
#include "jitify.hpp"

#include "../common/sim_config.h"
#include "../allocations/memory_allocator.h"
#include "../results/result_store.h"


#define KERNAL404 "kernal.cu file not found. Please run kernal_assembler.py from analysis folder and place the 'kernal.cu' file in the same folder as executable." 
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
#define JIT_EXPRESSION_LOCATION "//__SEARCH_MARKER_FOR_JIT_EXPRESSION__"
#define JIT_CONSTRAINT_LOCATION "//__SEARCH_MARKER_FOR_JIT_CONSTRAINT__"
#define JIT_INVARIANT_LOCATION "//__SEARCH_MARKER_FOR_JIT_INVARIANTS__"

#ifndef KERNAL_PATH
#define KERNAL_PATH ("kernal.cu")
#endif

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

inline std::string replace(std::string& content, const std::string& marker, const std::stringstream& new_content)
{
    const size_t start_pos = content.find(marker);
    if(start_pos == std::string::npos)
        throw std::runtime_error("Could not find the JIT EXPR LOCATION marker in source");

    return content.replace(start_pos, marker.length(), new_content.str());
}

class jit_compiler
{
public:
    static jitify::KernelInstantiation compile(jit_compile_visitor* expr_compiler)
    {
        expr_compiler->finalize();

        
        std::ifstream file(KERNAL_PATH);
        if(file.fail()) throw std::runtime_error(KERNAL404);
        std::stringstream buffer;
        buffer << "kernal.cu\n" << file.rdbuf();
        file.close();
        
        std::string jit_content = buffer.str();

        //replace content
        jit_content = replace(jit_content, JIT_EXPRESSION_LOCATION, expr_compiler->get_expr_compilation());
        jit_content = replace(jit_content, JIT_CONSTRAINT_LOCATION, expr_compiler->get_constraint_compilation());
        jit_content = replace(jit_content, JIT_INVARIANT_LOCATION, expr_compiler->get_invariant_compilation());

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
                {"--use_fast_math", "-I " CUDA_INC_DIR, "--dopt=on", "-D __JIT_COMPILING__"},
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
