#ifndef RESULT_STORE
#define RESULT_STORE


#include "../engine/Domain.h"
#include "../allocations/memory_allocator.h"


class memory_allocator;

struct estimate_target
{
    enum estimate_types
    {
        reachability,
        min_est,
        max_est
    } type;
    int variable_id;
};


struct result_wrapper
{
private:
    bool owns_pointers_;
    void* source_p_;
public:
    explicit result_wrapper(
        const bool owns_pointers,
        void* free_p,
        estimate_target target,
        int* h0_hypothesis,
        int* h1_hypothesis,
        double* estimates,
        int n_parallelism);

    estimate_target target;
    int* h0_hypothesis;
    int* h1_hypothesis;
    double* estimates;
    int n_parallelism;
    
    void free_internals() const;
};

#define USE_CHERNOFF_BOUND (-1.0)
#define IS_CHERNOFF_BOUND(x) ((x) <= 0.0)
class result_store_2
{
    bool use_cuda_;
    estimate_target est_target_{};
    
    int n_parallelism_;
    int sims_per_entry_;
    double certainty_;
    
    double* estimates_; //per thread (n_parallelism) 
    int* h0_hypothesis_;
    int* h1_hypothesis_;


    inline void store_estimate(const unsigned idx, const state* state) const
    {
        switch(est_target_.type)
        {
        case estimate_target::reachability: return;
            //TODO fix
        case estimate_target::min_est:
            estimates_[idx] += state->variables.store[est_target_.variable_id].max_value - 1; break;
        case estimate_target::max_est:
            estimates_[idx] += state->variables.store[est_target_.variable_id].max_value; break;
        }
    }

    inline size_t get_size() const
    {
        return (sizeof(double) + 2*sizeof(int)) * n_parallelism_;
    }
    
public:
    explicit result_store_2(memory_allocator* allocator,
        const estimate_target est_target,
        const int sims_per_entry,
        const int n_parallelism)
    {
        this->use_cuda_ = allocator->use_cuda;
        this->sims_per_entry_ = sims_per_entry;
        this->n_parallelism_ = n_parallelism;
        this->est_target_ = est_target;
        this->certainty_ = 1.0;

        void* point;
        allocator->allocate(&point, get_size());

        this->h0_hypothesis_ = static_cast<int*>(point);
        point = &h0_hypothesis_[n_parallelism];
        this->h1_hypothesis_ = static_cast<int*>(point);
        point = &h1_hypothesis_[n_parallelism];
        estimates_ = static_cast<double*>(point);
    }
    
    // CPU bool hypothesis_concluded() const
    // {
    //     if(IS_CHERNOFF_BOUND(certainty_)) return false;
    //     if(est_target_.type == estimate_target::reachability) return false;
    //
    //     result_wrapper results = load_results();
    //     
    //     results.free_internals();
    // }

    void write_result(const unsigned idx,  const state* sim) const
    {
        store_estimate(idx, sim);
        
        const bool reached = IS_GOAL(sim->query_state);
        int* hypothesis = reached ? this->h0_hypothesis_ : this->h1_hypothesis_;

        hypothesis[idx]++;
    }

    result_wrapper load_results() const
    {
        if(use_cuda_)
        {
            return result_wrapper
            {
                false,
                h0_hypothesis_, //original entry point
                est_target_,
                h0_hypothesis_,
                h1_hypothesis_,
                estimates_,
                n_parallelism_
            };
        }

        void* og_point = malloc(get_size());
        cudaMemcpy(og_point, h0_hypothesis_, get_size(), cudaMemcpyDeviceToHost);
        
        void* point = og_point;
        int* h0 = static_cast<int*>(point);
        point = &h0[n_parallelism_];
        int* h1 = static_cast<int*>(point);
        point = &h1[n_parallelism_];
        double* estimates = static_cast<double*>(point);

        return result_wrapper
        {
            true,
            og_point,
            est_target_,
            h0,
            h1,
            estimates,
            n_parallelism_
        };
    }
};

#endif