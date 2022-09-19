cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_Test LANGUAGES CXX CUDA)

add_library(particles STATIC
        ./node_d.cu
        ./node_d.h
        ./edge_d.cu
        ./edge_d.h
        ./update_d.cu
        ./update_d.h
        ./guard_d.cu
        ./guard_d.h
        ./timer_d.cu
        ./timer_d.h
        )

target_compile_features(particles PUBLIC cxx_std_11)

set_target_properties( particles
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(particle_test main.cu)

set_property(TARGET particle_test
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(particle_test PRIVATE particles)
target_compile_options(igl_common INTERFACE /MP $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)