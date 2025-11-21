#pragma once

#include <algorithm>
#include <vector> 
#include <iostream>
#include <memory>
#include <string>

#define MAX_BS 16384

struct Request{
    std::string id; 
    bool is_new_req; 
    double ddl;
    int input_length;
    int n_computed_tokens;
    double profit;
    int mem;
    int tpot_idx;
    int prefill_mem;
    int prefill_device_id; 
    int decode_device_id;
    bool prefill_only;
    
    Request() = default;

    Request(std::string id, 
    bool is_new_req, 
    double ddl, 
    int input_length,
    int n_computed_tokens,
    double profit,
    int mem, 
    int tpot_idx,
    int prefill_mem = -1,
    int prefill_device_id = 0,
    int decode_device_id = 0,
    bool prefill_only = false): 
        id(id),
        is_new_req(is_new_req), 
        ddl(ddl), 
        input_length(input_length),
        n_computed_tokens(n_computed_tokens),
        profit(profit), 
        mem(mem), 
        tpot_idx(tpot_idx),
        prefill_mem(prefill_mem),
        prefill_device_id(prefill_device_id),
        decode_device_id(decode_device_id),
        prefill_only(prefill_only) {}
};

struct ReqBatch {
    std::string id;     
    bool is_prefill;
    int n;
    
    ReqBatch(
        std::string id, bool is_prefill, int n
    ): id(id), is_prefill(is_prefill), n(n) {}
};

struct Batch {
    std::vector<ReqBatch> req_batches;
    int prefill_bs;
    int next = 1;
    double estimated_time = 0.0;

    int batch_size() const {
        int n = 0;
        for (auto& req_batch: req_batches) {
            n += req_batch.n;
        }
        return n;
    }
    
    int max_decode_size() const {
        int max_decode_size = 0;
        for (auto& req_batch: req_batches) {
            if (req_batch.is_prefill) continue;
            max_decode_size = std::max(max_decode_size, req_batch.n);
        }
        return max_decode_size;
    }
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
    bool fixed_bs;
    bool continuous;
    
    double batch_to_time(int n_tokens, size_t n_reqs = 1, size_t n_past_tokens = 0, size_t decode_steps = 1);
    int time_to_batch(double t, size_t n_reqs = 1, size_t n_past_tokens = 0, size_t decode_steps = 1);

public:
    BatchPlanner(
        const char* name, 
        const std::vector<double>& tpots,
        const std::vector<double>& hardware_params,
        bool fixed_bs, size_t max_bs, bool continuous
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
        bool decode_only = false,
        double finish_time = 1e9
    ) = 0;
    // virtual std::vector<Batch> plan_decode_only(
    //     const std::vector<Request>& reqs
    // );
    size_t n_tiers() const {return tpots.size();}
    friend class AdmCtrlScheduler;
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
        size_t max_bs,
        double alpha, 
        int max_sd_size, 
        bool fixed_spec,
        bool continuous 
    ): BatchPlanner("SDBatchPlanner", tpots, hardware_params, fixed_bs, max_bs, continuous),
     alpha(alpha), max_sd_size(max_sd_size), fixed_spec(fixed_spec) {
     }
    std::tuple<int, double, std::vector<Batch> > plan(
        double t,
        const std::vector<Request>& reqs,
        bool decode_only,
        double finish_time
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
        bool fixed_bs, size_t max_bs, bool continuous):
        BatchPlanner("ARBatchPlanner", tpots, hardware_params, fixed_bs, max_bs, continuous) {

        }
    
    std::tuple<int, double, std::vector<Batch> > plan(
        double t,
        const std::vector<Request>& reqs,
        bool decode_only,
        double finish_time
    ) override;
};

class AdmCtrlScheduler {
protected:
    std::string mode;
    bool continuous = false;
    bool _verbose;

    std::unique_ptr<BatchPlanner> planner;

    void _batch_impl(
        const std::vector<Request>& reqs,
        const std::vector<bool>& is_accepted,
        std::vector<Batch>& batches
    );

    bool _check_slo_violation(
        const std::vector<Request>& reqs,
        const std::vector<bool>& is_accepted,
        const std::vector<Batch>& batches,
        double current_time
    );


    std::tuple<bool, std::vector<bool>, 
        std::vector<Batch> > _admission_control_fcfs(
        std::vector<Request>& reqs,
        const int M,
        double current_time
    );

    std::tuple<bool, std::vector<bool>, 
        std::vector<Batch> > _admission_control_dp(
        std::vector<Request>& reqs,
        const int M,
        double current_time
    );

    std::tuple<bool, std::vector<bool>, 
        std::vector<Batch> > _admission_control_edf(
        std::vector<Request>& reqs,
        const int M,
        double current_time
    );

public: 
    AdmCtrlScheduler(): mode("fcfs"), continuous(false) {}
    
    AdmCtrlScheduler(
        std::string mode,
        bool continuous = false
    ): mode(mode), continuous(continuous) {}

    AdmCtrlScheduler& set_ar_planner(
        std::vector<double>& tpots,
        std::vector<double>& hardware_params,
        bool fixed_bs,
        size_t max_bs = 16384
    ) {
        std::cout << "[AdmCtrlScheduler] setting ARBatchPlanner with tpots: ";
        for (double t : tpots) {
            std::cout << t << " ";
        }
        std::cout << " and hardware_params: ";
        for (double h : hardware_params) {
            std::cout << h << " ";
        }
        std::cout << "max BS: " << max_bs << ", fixed_bs: " << fixed_bs;
        std::cout << std::endl;
        planner = std::make_unique<ARBatchPlanner>(
            tpots, hardware_params, fixed_bs, max_bs, continuous
        );
        return *this;
    }

    AdmCtrlScheduler& set_sd_planner(
        std::vector<double>& tpots,
        std::vector<double>& hardware_params,
        bool fixed_bs,
        size_t max_bs,
        double alpha, 
        int max_sd_size = 15,
        bool fixed_spec = false
    ) {
        planner = std::make_unique<SDBatchPlanner>(
            tpots, hardware_params, fixed_bs, max_bs,
            alpha, max_sd_size, fixed_spec, continuous
        );
        return *this;
    }

    std::tuple<bool, std::vector<std::string>
        , std::vector<Batch> > schedule(
        std::vector<Request>& reqs,
        int M,
        double current_time,
        bool verbose
    );
};

struct RequestOutput {
    bool admitted;
    int prefill_device_id;
    int decode_device_id;
};


class AdmCtrlRouter {
    int n_devices;
    std::unique_ptr<BatchPlanner> planner;

public:
    AdmCtrlRouter(int n_devices,
        std::vector<double> hardware_params,
        double tpot): 
        n_devices(n_devices) {
            planner = std::make_unique<ARBatchPlanner>(
                std::vector<double>{tpot}, hardware_params, false, 16384, false
            );
        }
    
    std::tuple<std::vector<RequestOutput>, std::vector<std::vector<Batch> > > schedule(
        std::vector<Request>& reqs,
        std::vector<int>& Ms,
        double current_time,
        bool verbose
    );
};