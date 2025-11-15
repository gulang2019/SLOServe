#ifdef NDEBUG
#undef NDEBUG
#endif

#include "promax_spec_sch.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <list>

const double INFINITE_TIME = 100.;

std::ostream& operator << (std::ostream& o, Request& req) {
    o  << "id " << req.id << " is_new_req " << req.is_new_req 
        << " ddl "  << req.ddl
        << " iL "  << req.input_length
        << " profit " << req.profit 
        << " mem " << req.mem 
        << " tpot_idx " << req.tpot_idx;
    return o;
}

AdmCtrlScheduler::AdmCtrlScheduler(
    std::vector<double>& tpots,
    std::vector<double>& hardware_params,
    double spec_decode_alpha,
    int max_spec_decode_size
): tpots(tpots), 
   hardware_params(hardware_params),
   spec_decode_alpha(spec_decode_alpha),
   max_spec_decode_size(max_spec_decode_size) {
    std::cout << "Backend:";
    std::cout << "Tpots: ";
    for (double t : tpots) {
        std::cout << t << " ";
    }
    std::cout << "\nHardware Params: ";
    for (double h : hardware_params) {
        std::cout << h << " ";
    }
    std::cout << "\nSpec Decode Alpha: " << spec_decode_alpha
              << "\nMax Spec Decode Size: " << max_spec_decode_size
            << std::endl;
}


std::tuple<int, double, Batch> AdmCtrlScheduler::f(
    double t,
    const std::vector<int>& n_reqs
) {
    int max_prefill_tokens = 0;
    double elasped_time = 0.;
    Batch best_batch(0, 0);

    std::vector<int> sd_sizes(n_reqs.size());
    for (int i = 0; i < n_reqs.size(); i++) {
        for (int j = 1; j <= max_spec_decode_size; j += 1) {
            std::vector<int> spec_decode_sizes;
            double unit_time = t;
            int n_decode_tokens = 0;
            int max_sd_size = 0;
            for (int k = 0; k < n_reqs.size(); k++) {
                int spec_decode_size = std::min((int)std::ceil(std::log((tpots[k] - tpots[i]) + 
                std::pow(spec_decode_alpha, j * (tpots[i]/tpots[k]))) 
                / std::log(spec_decode_alpha)), max_spec_decode_size);
                assert(spec_decode_size >= 0);
                max_sd_size = std::max(max_sd_size, spec_decode_size);
                spec_decode_sizes.push_back(spec_decode_size);
                unit_time = std::min(unit_time, spec_sample(spec_decode_size) * tpots[k]);
                n_decode_tokens += spec_decode_size * n_reqs[k];
                sd_sizes[k] = spec_decode_size;
            }
            int cur_n_batch = (int)std::floor(t / unit_time);
            double cur_elasped_time = cur_n_batch * unit_time;
            int cur_bs = time_to_batch(unit_time, max_sd_size);
            if (cur_bs < 0) continue;
            auto token_per_batch = cur_bs - n_decode_tokens;
            auto cur_prefill_tokens = token_per_batch * cur_n_batch;
            if (cur_prefill_tokens >= max_prefill_tokens) {
                elasped_time = cur_elasped_time;
                best_batch = Batch(cur_bs, cur_n_batch, sd_sizes);
                max_prefill_tokens = cur_prefill_tokens;
            }
        }
    }

    return {max_prefill_tokens, elasped_time, best_batch};
}

// Define the State struct
struct State {
    int i;
    double profit;
    int mem;
    int n_prefill_tokens;
    double finish_time;
    Batch batch;
    State* last;

    State(int i, double p, int m, 
            int n, double f, 
        State* last = nullptr)
        : i(i), 
        profit(p), 
        mem(m), 
        n_prefill_tokens(n), 
        finish_time(f), 
        last(last) {}
};

std::ostream& operator << (std::ostream& o, State& s) {
    o << "i " << s.i << " profit " << s.profit << " mem " << s.mem << " n_prefill_tokens " << s.n_prefill_tokens << " finish_time " << s.finish_time << " batch " << s.batch << std::endl;
    if (s.last == nullptr)
        o << "last is null" << std::endl;
    else o << *s.last << std::endl;
    return o;
}

std::ostream& operator << (std::ostream& o, BatchSch& batch_sch) {
    o << "BatchSch(" << std::endl;
    for (auto& b: batch_sch.batches) o << b << std::endl;
    o << ")" << std::endl;
    return o;
}

std::ostream& operator << (std::ostream& o, ReqBatchSch& req) {
    o << "ReqBatchSch(id=" << req.id << ", is_prefill=" << req.is_prefill << ", n=" << req.n << ")";
    return o;
}

std::ostream& operator << (std::ostream& o, Batch& req) {
    o << "Batch(bs=" << req.bs << ", #batch=" << req.n_batch << ", sd_sizes:{";
    for (auto& sd_size: req.sd_sizes) o << sd_size << ",";
    o << "})";
    return o;
}

bool dominates(const State& r1, const State& r2) {
    // r1 is better than r2 if:
    // - r1.profit >= r2.profit (maximize)
    // - r1.mem <= r2.mem (minimize)
    // - r1.n_prefill_tokens <= r2.n_prefill_tokens (minimize)
    // - r1.finish_time <= r2.finish_time (minimize)
    // - And at least one comparison is strict
    return (r1.profit >= r2.profit &&
            r1.mem <= r2.mem &&
            r1.n_prefill_tokens <= r2.n_prefill_tokens) &&
           (r1.profit > r2.profit ||
            r1.mem < r2.mem ||
            r1.n_prefill_tokens < r2.n_prefill_tokens);
}

// Pareto frontier calculation
#include <list>
#include <iterator>

void _pareto_max(std::list<State>& states) {
    for (auto it1 = states.begin(); it1 != states.end(); ++it1) {
        bool is_dominated = false;
        for (auto it2 = states.begin(); it2 != states.end(); ++it2) {
            if (it1 != it2 && dominates(*it2, *it1)) {
                is_dominated = true;
                break;
            }
        }
        if (is_dominated) {
            it1 = states.erase(it1);  // Erase returns the next valid iterator
            --it1;  // Step back to counter the increment in the for-loop
        }
    }
}

struct Serializer {
    std::vector<int> _offsets;
    Serializer(const std::vector<int>& offsets) {
        _offsets.push_back(1);
        for (auto& offset: offsets) {
            _offsets.push_back(_offsets.back() * offset);
        }
    }

    int size() const {
        return _offsets.back();
    }

    int t2i(const std::vector<int>& tuple) {
        int idx = 0;
        for (size_t i = 0; i < tuple.size(); ++i) {
            idx += tuple[i] * _offsets[i];
        }
        return idx;
    }

    std::vector<int> i2t(int idx) {
        std::vector<int> tuple;
        for (size_t i = 1; i < _offsets.size(); i ++) {
            tuple.push_back((idx % _offsets[i]) / _offsets[i-1]);
        }
        return tuple;
    }

};

bool operator <= (const std::vector<int>& v1, const std::vector<int>& v2) {
    assert(v1.size() == v2.size());
    for (int i = 0; i < v1.size(); ++i) {
        if (v1[i] > v2[i]) return false;
    }
    return true;
}

std::tuple<bool, std::vector<bool>, 
    std::vector<Batch> > AdmCtrlScheduler::_admission_control(
    std::vector<Request>& reqs,
    int M,
    double current_time
){
    std::vector<
        std::vector<
        std::list<State> > > dp;
    dp.resize(reqs.size()+1);

    int tot_old_req = 0;
    for (auto& req: reqs) {
        tot_old_req += 1 - req.is_new_req;
    }
    
    int last_old_req = 0;
    dp[0].resize(1);
    dp[0][0].push_back(State(
           -1,
           0,
           0,
           0,
           current_time,
           nullptr 
    ));
    State* best_state = nullptr;

    std::vector<int> req_cnts(tpots.size(), 0);
    std::vector<int> n_olds(tpots.size(), 0);
    int n_old = 0;
    for (int i = 0; i < (int) reqs.size(); ++i) {
        req_cnts[reqs[i].tpot_idx] += 1;
    } 
    Serializer S(req_cnts);

    for (int n = 1; n <= reqs.size(); n ++) {
        // dp[n].resize(n+1);
        dp[n].resize(S.size());
        auto& req = reqs[n-1];
        dp[n][0].push_back(State(
           -1,
           0,
           0,
           0,
           current_time,
           nullptr 
        ));
        // for (int r = n_old + 1; r <= n; r ++) {
        for (int r = 1; r < S.size(); r ++) {
            auto n_reqs = S.i2t(r);
            // if we cannot satisfy the decoding requirement, break
            if (std::get<1>(f(INFINITE_TIME, n_reqs)) == 0.) break;
            n_reqs[req.tpot_idx] -= 1;
            if (!(n_reqs <= n_olds)) continue;
            auto& states = dp.at(n).at(r);
            int n_accs = 0;
            for (int i = 0; i < n_reqs.size(); i+= 1) 
                n_accs += n_reqs[i];
            for (int m = std::max(last_old_req, n_accs - 1); m < n; m += 1) {
                for (auto& state: dp.at(m).at(S.t2i(n_reqs))) {
                    int n_prefill_tokens;
                    double elapsed_time;
                    Batch batch;
                    std::tie(n_prefill_tokens, elapsed_time, batch) =\
                     f(req.ddl - state.finish_time, n_reqs);

                    State s = State(
                        /*.i*/ n,
                        /*.profit =*/ state.profit + req.profit,
                        /*.mem =*/ state.mem + req.mem,
                        /*.prefill_tokens =*/ state.n_prefill_tokens + n_prefill_tokens - req.input_length,
                        /*.finish_time =*/ state.finish_time + elapsed_time,
                        /*.last = */ &state
                    );

                    s.batch = batch;
                    if ((s.mem > M) || (s.n_prefill_tokens < 0)) continue;
                    // states.push_back(s);
                    states.push_back(s);

                    if ((tot_old_req == (n_old + 1 - req.is_new_req)) and (
                        !best_state or best_state->profit < s.profit
                    )) {
                        best_state = &states.back();
                    }

                }
            }
            _pareto_max(states);
        }
        if (!req.is_new_req) {
            n_olds[req.tpot_idx] += 1;
            n_old += 1;
            last_old_req = n;
        }
    }

    std::vector<bool> accept_request;
    std::vector<Batch> batches;
    accept_request.resize(reqs.size(), false);

    if (not best_state) 
        return {false, accept_request, batches};
    // std::cout << *best_state << std::endl;
    auto state = best_state;
    while (state and state->i != -1){
        if (not (state->i >=1 && state->i <= accept_request.size())) {
            std::cout << "state->i " << state->i << std::endl;
        }
        assert(state->i >=1 && state->i <= accept_request.size());
        accept_request[state->i-1] = true;
        // std::cout << "insert " << state-> batch << std::endl;
        batches.insert(batches.begin(), state->batch);
        state = state->last;
    }

    for (int i = 0; i < reqs.size(); ++i) {
        assert(reqs[i].is_new_req or accept_request[i]);
    }

    return {true, accept_request, batches};
}

std::vector<BatchSch> AdmCtrlScheduler::_batch_impl(
    std::vector<Request>& reqs,
    std::vector<bool>& is_accepted, 
    std::vector<Batch>& batches
){
    /*1. We realize the batch first. */
    assert(reqs.size() == is_accepted.size());

    std::vector<int> acc_indices;

    std::vector<int> n_reqs(tpots.size(), 0);

    for (int i = 0; i < reqs.size(); i++) {
        if (is_accepted[i]) {
            acc_indices.push_back(i);
            n_reqs[reqs[i].tpot_idx] ++;
        }
    }
    assert(acc_indices.size() == batches.size());

    int scheduled_idx = 0;
    
    std::vector<BatchSch> batch_schs;
    for (int i = 0; i < acc_indices.size(); ++i) {
        while ((scheduled_idx < acc_indices.size()) 
            and (reqs[acc_indices[scheduled_idx]].input_length == 0)){
            scheduled_idx += 1;
        }
        if (scheduled_idx == acc_indices.size())
            break;
        auto& batch = batches[i];
        for (int j = 0; j < batch.n_batch; j++) {
            if (scheduled_idx == acc_indices.size())
                break;
            BatchSch sch;
            sch.bs = batch.bs;
            sch.repeat = false;
            std::vector<bool> mask(acc_indices.size(), false);
            for (int k = 0; k < i; k++){
                auto& prior_req = reqs[acc_indices[k]];
                int sd_size = batch.sd_sizes[prior_req.tpot_idx];
                sch.batches.push_back(ReqBatchSch(prior_req.id, false, sd_size));
                sch.bs -= sd_size;
            }
            assert(sch.bs >= 0);
            while (sch.bs > 0 and scheduled_idx < acc_indices.size()) {
                // std::cout << "Looping..." << std::endl;
                auto& prior_req = reqs[acc_indices[scheduled_idx]];
                auto prefill_scheduled = std::min(prior_req.input_length,
                                    sch.bs);
                if (prefill_scheduled > 0) {
                    sch.batches.push_back(ReqBatchSch(prior_req.id, true, prefill_scheduled));
                    prior_req.input_length -= prefill_scheduled;
                    sch.bs -= prefill_scheduled;
                }
                if (prior_req.input_length == 0) 
                    scheduled_idx ++;
            }
            // std::cout << "SCH " << sch << std::endl;
            batch_schs.emplace_back(sch);
        }   
        assert(scheduled_idx >= i);
    }


    // here, we have a repeated final batch for decoding operations
    int _;
    double elasped_time;
    Batch batch;
    std::tie(_, elasped_time, batch) = f(INFINITE_TIME, n_reqs);
    assert(elasped_time > 0.);

    BatchSch sch;
    for (int i = 0; i < acc_indices.size(); i++) {
        auto& req = reqs[acc_indices[i]];
        auto sd_size = batch.sd_sizes[req.tpot_idx];
        sch.batches.push_back(ReqBatchSch(
            req.id, 
            false, 
            sd_size
        ));
        batch.bs -= sd_size;        
    }
    assert(batch.bs >= 0);
    int additional_bs = batch.bs / acc_indices.size();
    if (additional_bs > 0) {
        for (auto& batch: sch.batches) 
            batch.n = std::min(max_spec_decode_size, additional_bs + batch.n);
    }
    sch.repeat = true;
    batch_schs.push_back(sch);
    return batch_schs;
}

std::tuple<bool, std::vector<int>, std::vector<BatchSch> > AdmCtrlScheduler::schedule(
        std::vector<Request>& reqs,
        int M,
        double current_time,
        bool verbose
){
    _verbose = verbose;

    if (_verbose) {
        std::cout << "Requests:" << std::endl;
        for (int i = 0; i < reqs.size(); ++i) {
            std::cout << reqs[i] << std::endl;
        }
    }

    sort(reqs.begin(), reqs.end(), [](Request& r1, Request& r2){
        return r1.ddl < r2.ddl; 
    });
    bool is_feasible;
    std::vector<bool> is_accepted;
    std::vector<Batch> batches;
    std::tie(is_feasible, is_accepted, batches) = _admission_control(reqs, M, current_time);

    if (not is_feasible) {
        return {false, {}, {}};
    }

    if (_verbose) {
        for (int i = 0; i < reqs.size(); ++i) {
            std::cout <<(is_accepted[i]? "ACC":"REJ") << " " << reqs[i] << std::endl;
        }

        std::cout << "batches:" << std::endl;
        for (auto & batch: batches) {
            std::cout << batch << std::endl;
        }
    }

    std::vector<int> acc_ids;
    for (int i = 0 ; i < is_accepted.size(); i++) {
        if (is_accepted[i]) 
            acc_ids.push_back(reqs[i].id);
    }

    std::vector<BatchSch> schs = _batch_impl(reqs, is_accepted, batches);
    if (_verbose) {
        std::cout << "schedules" << std::endl;
        for (auto& sch: schs) {
            std::cout << sch << std::endl;
        }
    }
    /*2. Realize the batch & Maximize the utilization*/
    return {true, acc_ids, schs};
}

double AdmCtrlScheduler::spec_sample(int n) {
    assert(n >= 1);
    return (1 - std::pow(spec_decode_alpha, n)) / (1 - spec_decode_alpha);
}

double AdmCtrlScheduler::batch_to_time(int n, size_t decode_steps) {
    assert(n >= 0);
    double t = 0;
    for (int i = 0; i < hardware_params.size(); i += 3){
        double k1 = hardware_params[i];
        double k2 = hardware_params[i+1];
        double b = hardware_params[i+2];
        t = std::max(k1 * n + k2 * decode_steps + b, t);
    }
    return t;
}

int AdmCtrlScheduler::time_to_batch(double t, size_t decode_steps) {
    double bs = 16384;
    for (int i = 0; i < hardware_params.size(); i += 3){
        double k1 = hardware_params[i];
        double k2 = hardware_params[i+1];
        double b = hardware_params[i+2];
        bs = std::min((t - b - k2 * decode_steps) / k1, bs);
    }
    if (bs < 0) return -1;
    return int(bs);
}

/*
When one request is yielding more tokens than needed, we can actually degrade the service.
Multi-tier services. 
TPOT
*/

