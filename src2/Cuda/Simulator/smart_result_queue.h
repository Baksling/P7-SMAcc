#pragma once

#include "simulation_strategy.h"
#include "../common/macro.h"
#include "../common/allocation_helper.h"
#include "../Domain/simulator_state.h"

#define NO_KEY (0xffffffff)

//queue states
#define RDY (0xffffffff)
#define IN_USE_START (0)


struct data_vector
{
    bool is_node;
    unsigned  sim_id;
    unsigned step;
    int item_id;
    double value;
};


class smart_result_queue
{
private:
    const bool cuda_allocated_;
    unsigned blocks_;
    unsigned block_size_;
    int* state_array_;
    data_vector* data_arr_;

    unsigned queue_pointer = 0;

    void atomic_pick_block(unsigned* qid)
    {
        int* q;
        while(true)
        {
            q = &this->state_array_[queue_pointer];
            if(*q == RDY) break;
            queue_pointer = (queue_pointer+1) % this->blocks_;
        }
        *q = IN_USE_START;
        *qid = queue_pointer;
    }

    void announce_block_full(const unsigned* qid)
    {
        if(*qid >= this->blocks_) printf("cannot announce wack block");
        this->state_array_[*qid] = (-this->state_array_[*qid]);
    }

    void process_vector(const data_vector* vector);

    CPU GPU void write_to_queue(const unsigned sim_id, const unsigned step, const int item_id, const double value, unsigned* qid)
    {
        if(qid == nullptr
            || this->state_array_[*qid] < 0)
        {
            this->atomic_pick_block(qid);
        }
        if(this->state_array_[*qid] >= static_cast<int>(this->block_size_))
        {
            announce_block_full(qid);
            this->atomic_pick_block(qid);
        }
        const unsigned id = *qid;
        constexpr unsigned size = this->block_size_;
        const int index = this->state_array_[id];
        
        this->data_arr_[id * size + index] = data_vector{ sim_id, step,item_id, value };
        this->state_array_[id]++;
    }
    
public:
    smart_result_queue(const unsigned blocks, const unsigned block_size, const unsigned parallelism, const bool cuda) : cuda_allocated_(cuda)
    {
        this->blocks_ = blocks;

        if(parallelism < blocks)
            throw std::runtime_error("Not enough blocks to store results, given the degree of parrallelism");
        
        this->block_size_ = block_size;

        if(blocks == 0 || block_size == 0)
        {
            state_array_ = nullptr;
            data_arr_ = nullptr;
            return;
        }

        allocate(&this->data_arr_, static_cast<unsigned long long>(blocks*block_size)*sizeof(data_vector), cuda);

        int* local_state = new int[blocks];

        for (unsigned i = 0; i < blocks; ++i)
        {
            local_state[i] = RDY;
        }
        
        if(cuda)
        {
            const unsigned long long size = static_cast<unsigned long long>(blocks)*sizeof(int);
            cudaMalloc(&this->state_array_, size);
            cudaMemcpy(this->state_array_, local_state, size, cudaMemcpyHostToDevice);
            free(local_state);
        }
        else this->state_array_ = local_state;
    }

    CPU GPU void write_node(const node_t* node, const simulator_state* state, unsigned* qid)
    {
        write_to_queue(state->sim_id_, state->steps_, node->get_id(), state->global_time_, qid);
    }

    CPU GPU void write_variables(const array_t<clock_variable>* variables, const simulator_state* state, unsigned* qid)
    {
        for (int i = 0; i < variables->size(); ++i)
        {
            write_to_queue(state->sim_id_, state->steps_, i, variables->at(i)->get_time(), qid);
        }
    }


    CPU void check_offloads(const bool* stop_call)
    {
        unsigned i = 0;
        while(!(*stop_call))
        {
            i = (i + 1) % this->blocks_;
            if(this->state_array_[i] < 0 && this->state_array_[i] != static_cast<int>(RDY)) continue;
            data_vector* local_vector = nullptr;

            cudaMemcpy(local_vector,
                &this->data_arr_[static_cast<unsigned>(i*this->blocks_*this->block_size_)],
                sizeof(data_vector)*this->block_size_,
                cudaMemcpyDeviceToHost);

            this->state_array_[i] = RDY;

            process_vector(local_vector);
        }
    }
};