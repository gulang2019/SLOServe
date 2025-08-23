#pragma once

#include <algorithm>
#include <vector> 
#include <iostream>
#include <memory>

#define MAX_BS 16384

struct Request{
    int id; 
    bool is_new_req; 
    double ddl;
    int input_length;
    double profit;
    int mem;
    int tpot_idx;
    
    Request(int id, 
    bool is_new_req, 
    double ddl, 
    int input_length, 
    double profit,
    int mem, 
    int tpot_idx): 
        id(id),
        is_new_req(is_new_req), 
        ddl(ddl), 
        input_length(input_length),
        profit(profit), 
        mem(mem), 
        tpot_idx(tpot_idx) {}
};

struct ReqBatch {
    int id; 
    bool is_prefill;
    int n;
    ReqBatch(
        int id, bool is_prefill, int n
    ): id(id), is_prefill(is_prefill), n(n) {}
};

struct Batch {
    std::vector<ReqBatch> req_batches;
    int prefill_bs;
    int next = 1;
};

// struct Batch {
//     int bs = 0;
//     int n_batch = 0;
//     std::vector<int> sd_sizes;

//     Batch() = default;

//     Batch(int bs, int n_batch) 
//         : bs(bs), n_batch(n_batch) {}

//     Batch(int bs, int n_batch, const std::vector<int>& sd_sizes_)
//         : bs(bs), n_batch(n_batch), sd_sizes(sd_sizes_) {}

//     Batch(int bs, int n_batch, std::vector<int>&& sd_sizes_)
//         : bs(bs), n_batch(n_batch), sd_sizes(std::move(sd_sizes_)) {}
// };


std::ostream& operator << (std::ostream& o, Request& req);
std::ostream& operator << (std::ostream& o, Batch& req);
std::ostream& operator << (std::ostream& o, ReqBatch& req);

class BatchPlanner {
protected:
    const char* name;
    std::vector<double> hardware_params;
    std::vector<double> tpots;
    int max_bs;
    bool continuous;
    
    double batch_to_time(int n, size_t decode_steps = 1);
    int time_to_batch(double t, size_t decode_steps = 1);

public:
    BatchPlanner(
        const char* name, 
        const std::vector<double>& tpots,
        const std::vector<double>& hardware_params,
        bool fixed_bs, bool continuous
    );
    virtual ~BatchPlanner() = default;

    /**
     * @return int: the extra token batches available for prefills; <0 indicates the decode SLO cannot be satisfied
     * @return double: the time elasped;
     * @return batches: the future batch schedules; it must guarantee that the decode SLOs are satisfied.
     */
    virtual std::tuple<int, double, std::vector<Batch> > plan(
        double t,
        const std::vector<Request>& reqs,
        bool decode_only = false
    ) = 0;
    // virtual std::vector<Batch> plan_decode_only(
    //     const std::vector<Request>& reqs
    // );
    size_t n_tiers() const {return tpots.size();}
    friend class PromaxScheduler;
};

class SDBatchPlanner: public BatchPlanner{
    double alpha;
    int max_sd_size;
    bool fixed_spec;
    double spec_sample(int n);

public: 
    SDBatchPlanner(
        const std::vector<double>& tpots,
        const std::vector<double>&  hardware_params,
        bool fixed_bs,
        double alpha, 
        int max_sd_size, 
        bool fixed_spec,
        bool continuous 
    ): BatchPlanner("SDBatchPlanner", tpots, hardware_params, fixed_bs, continuous),
     alpha(alpha), max_sd_size(max_sd_size), fixed_spec(fixed_spec){}
    std::tuple<int, double, std::vector<Batch> > plan(
        double t,
        const std::vector<Request>& reqs,
        bool decode_only
    ) override;

    // std::vector<Batch> plan_decode_only(
    //     const std::vector<Request>& reqs
    // ) override;
    
};

class ARBatchPlanner: public BatchPlanner{
public: 
    ARBatchPlanner(
        const std::vector<double>& tpots,
        const std::vector<double>&  hardware_params,
        bool fixed_bs, bool continuous):
        BatchPlanner("ARBatchPlanner", tpots, hardware_params, fixed_bs, continuous) {}
    
    std::tuple<int, double, std::vector<Batch> > plan(
        double t,
        const std::vector<Request>& reqs,
        bool decode_only
    ) override;
};

class PromaxScheduler {
    bool continuous = false;
    bool _verbose;

    std::unique_ptr<BatchPlanner> planner;

    void _batch_impl(
        const std::vector<Request>& reqs,
        const std::vector<bool>& is_accepted, 
        std::vector<Batch>& batches
    );

    std::tuple<bool, std::vector<bool>, 
        std::vector<Batch> > _admission_control(
        std::vector<Request>& reqs,
        const int M,
        double current_time
    );

public: 
    PromaxScheduler() = default;
    
    PromaxScheduler(
        bool continuous
    ): continuous(continuous) {}

    PromaxScheduler& set_ar_planner(
        std::vector<double>& tpots,
        std::vector<double>& hardware_params,
        bool fixed_bs
    ) {
        planner = std::make_unique<ARBatchPlanner>(
            tpots, hardware_params, fixed_bs, continuous
        );
        return *this;
    }

    PromaxScheduler& set_sd_planner(
        std::vector<double>& tpots,
        std::vector<double>& hardware_params,
        bool fixed_bs,
        double alpha, 
        int max_sd_size = 15,
        bool fixed_spec = false
    ) {
        planner = std::make_unique<SDBatchPlanner>(
            tpots, hardware_params, fixed_bs,
            alpha, max_sd_size, fixed_spec, continuous
        );
        return *this;
    }

    std::tuple<bool, std::vector<int>
        , std::vector<Batch> > schedule(
        std::vector<Request>& reqs,
        int M,
        double current_time,
        bool verbose
    );
};