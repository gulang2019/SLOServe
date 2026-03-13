#ifdef NDEBUG
#undef NDEBUG
#endif

#include "promax_spec_sch.h"
#include "timer.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <list>
#include <queue>
#include <unordered_set>
#include <unordered_map>

const double INFINITE_TIME = 1.;

std::ostream& operator << (std::ostream& o, Request& req) {
    o  << "id " << req.id << " is_new_req " << req.is_new_req 
        << " ddl "  << req.ddl
        << " iL "  << req.input_length
        << " profit " << req.profit 
        << " mem " << req.mem 
        << " tpot_idx " << req.tpot_idx;
    return o;
}

BatchPlanner::BatchPlanner(
    const char* name, 
    const std::vector<double>& tpots,
    const std::vector<double>& hardware_params,
    bool fixed_bs, bool continuous
): name(name), hardware_params(hardware_params), tpots(tpots), continuous(continuous) {
    assert(hardware_params.size() % 3 == 0);
    for (size_t i = 0; i < tpots.size()-1; ++i) 
        assert(tpots[i] <= tpots[i+1]);
    max_bs = fixed_bs? time_to_batch(tpots[0]):MAX_BS;
}

double BatchPlanner::batch_to_time(int n, size_t decode_steps) {
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

int BatchPlanner::time_to_batch(double t, size_t decode_steps) {
    if (batch_to_time(0, decode_steps) > t) return 0;
    double bs = 16384;
    for (int i = 0; i < hardware_params.size(); i += 3){
        double k1 = hardware_params[i];
        double k2 = hardware_params[i+1];
        double b = hardware_params[i+2];
        bs = std::min((t - b - k2 * decode_steps) / k1, bs);
    }
    // std::cout << "time2batch, t:" << t << ", decode_steps:" << decode_steps << std::endl;
    assert(bs > 0);
    return int(bs);
}

// std::vector<Batch> BatchPlanner::plan_decode_only(
//     const std::vector<Request>& reqs
// ) {
//     int max_prefill_tokens;
//     double elasped_time;
//     std::vector<Batch> batches;
//     std::tie(max_prefill_tokens, elasped_time, batches) = plan(
//         INFINITE_TIME, reqs
//     );
//     assert(max_prefill_tokens >= 0);
//     assert(batches.size());
//     batches.back().next = -batches.size() + 1;
//     return batches; 
// }

std::tuple<int, double, std::vector<Batch> > 
    SDBatchPlanner::plan(
    double t,
    const std::vector<Request>& reqs,
    bool decode_only
) {
    std::vector<int> n_reqs(n_tiers());
    for (auto& req: reqs){
        n_reqs[req.tpot_idx] ++;
    }
    
    if (not reqs.size()){
        Batch batch;
        batch.prefill_bs = time_to_batch(t);
        if (batch.prefill_bs) 
            return {batch.prefill_bs, batch_to_time(batch.prefill_bs), {batch}};
        return {0, 0., {}};
    }
    
    int prefill_tokens_best = 0;
    int decode_tokens_best = 0;
    double elasped_time_best = 0;
    std::vector<int> sd_sizes_best(n_reqs.size());
    int repeat_best = 0;

    // std::cout << "is continous: " << continuous << std::endl;
    std::vector<int> sd_sizes(n_reqs.size());
    // enumerate on the level that touches the tighest bound.
    for (int i = 0; i < n_reqs.size(); i++) {
        if (!n_reqs[i]) continue;
        int min_spec_decode_size = fixed_spec? max_sd_size:1;
        // enumerate on sd size for this level.
        for (int j = min_spec_decode_size; j <= max_sd_size; j += 1) {
            std::vector<int> spec_decode_sizes;
            double unit_time = t;
            int n_decode_tokens = 0;
            bool valid = true;
            // for all other levels, found the sd size that satisfy the SLO while not making it the bottleneck.
            for (int k = 0; k < n_reqs.size(); k++) {
                if (!n_reqs[k]) continue; 
                // std::cout << "tpot[k]" << tpots[k] 
                //     << ", tpots[i]: " << tpots[i] 
                //     << ", alpha: " << alpha
                //     << ", j: " << j << std::endl;
                double v = ((tpots[k] - tpots[i]) + std::pow(alpha, j) * tpots[i]) /tpots[k];
                int spec_decode_size = (int)std::ceil(std::log(v)
                / std::log(alpha));
                spec_decode_size = std::max(spec_decode_size, 
                    min_spec_decode_size);
                if (spec_decode_size > max_sd_size) {
                    valid = false;
                    break; 
                }
                assert((spec_decode_size >= 0) and (spec_decode_size <= max_sd_size));
                spec_decode_sizes.push_back(spec_decode_size);
                unit_time = std::min(unit_time, std::floor(spec_sample(spec_decode_size)) * tpots[k]);
                n_decode_tokens += spec_decode_size * n_reqs[k];
                sd_sizes[k] = spec_decode_size;
            }
            if (!valid) continue;
            int n_decode_steps = 0;
            for (auto& spec_decode_size: sd_sizes) 
                n_decode_steps = std::max(n_decode_steps, spec_decode_size);
            int cur_bs = time_to_batch(unit_time, n_decode_steps);
            cur_bs = std::min(cur_bs, max_bs);
            auto token_per_batch = cur_bs - n_decode_tokens;
            if (token_per_batch < 0) continue;
            if (decode_only) {
                unit_time = batch_to_time(n_decode_tokens, n_decode_steps);
                token_per_batch = 0;
            }
            int cur_n_batch = (int)std::floor(t / unit_time);
            double cur_elasped_time;
            int cur_prefill_tokens;
            if (continuous) {                
                cur_elasped_time = t;
                cur_prefill_tokens = t / unit_time * token_per_batch;
            }
            else {
                cur_elasped_time = cur_n_batch * unit_time;
                cur_prefill_tokens = cur_n_batch * token_per_batch;
            }
            if ((!decode_only && (cur_prefill_tokens >= prefill_tokens_best)) or 
                decode_only && (n_decode_tokens >= decode_tokens_best)) {
                elasped_time_best = cur_elasped_time;
                // best_batch = Batch(cur_bs, cur_n_batch, sd_sizes);
                sd_sizes_best = sd_sizes;
                repeat_best = decode_only? 1:cur_n_batch;
                prefill_tokens_best = cur_prefill_tokens;
                decode_tokens_best = n_decode_tokens;
            }
        }
    }

    // assert(best_repeat >= 1);
    std::vector<Batch> batches;
    if (repeat_best > 0) {
        Batch batch;
        batch.prefill_bs = prefill_tokens_best / repeat_best;
        for (auto& req: reqs)
            batch.req_batches.push_back(ReqBatch(req.id, false, sd_sizes_best[req.tpot_idx]));
        batches.resize(repeat_best, batch);    
    }

    return {prefill_tokens_best, elasped_time_best, batches};
}

double SDBatchPlanner::spec_sample(int n) {
    assert(n >= 1);
    return (1 - std::pow(alpha, n)) / (1 - alpha);
}

struct ReqDDL{
    int id;
    double tpot;
    double ddl;  
    ReqDDL(int id, double tpot, double ddl): id(id), tpot(tpot), ddl(ddl){}
};

struct ReqDDLCMP{
    bool operator () (const ReqDDL& r1, const ReqDDL& r2) {
        return r1.ddl > r2.ddl;
    }
};

std::tuple<int, double, std::vector<Batch> > 
    ARBatchPlanner::plan(
    double t,
    const std::vector<Request>& reqs,
    bool decode_only
) {
    // if (decode_only) {
    //     std::cout << "Here!" << std::endl;
    // }
    // 1. find the request with the tighest tpot;
    if (not reqs.size()){
        Batch batch;
        batch.prefill_bs = time_to_batch(t);
        if (batch.prefill_bs) 
        return {batch.prefill_bs, batch_to_time(batch.prefill_bs), {batch}};
        return {0, 0., {}};
    }

    // int lowest_tpot_idx = tpots.size();
    // for (auto& req: reqs) 
    // lowest_tpot_idx = std::min(lowest_tpot_idx, req.tpot_idx);
    // double tightest_tpot = 100;
    // for (auto& req: reqs)
    //     tightest_tpot = std::min(tightest_tpot, tpots[req.tpot_idx]);
    // double overhead = batch_to_time(1);

    // int min_tokens_required = 0;
    // for (auto& req: reqs) {
    //     min_tokens_required += (int)std::floor(t / tpots[req.tpot_idx]);
    // }
    // int n_batches = 0;
    // int max_tokens = 0;
    // for (int cur_n_batches = std::max((int)std::floor(t / tightest_tpot), 1); 
    //         cur_n_batches <= std::floor(t / overhead); cur_n_batches ++) {
    //     double per_batch_time = std::min(t / cur_n_batches, tightest_tpot);
    //     auto n_tokens = time_to_batch(per_batch_time) * cur_n_batches;
    //     if (n_tokens >= std::max(min_tokens_required, max_tokens)) {
    //         n_batches = cur_n_batches;
    //         max_tokens = n_tokens;
    //     }
    // }

    // if (n_batches == 0) {
    //     return {0, 0., {}};
    // }

    // size_t n_batches = int(std::floor(t / tpots[lowest_tpot_idx]));
    // std::cout << "plan t:" << t << ", lowest_tpot_idx:" << lowest_tpot_idx << ",reqs:";
    // for (auto& req: reqs) std::cout << req.id << ",";
    // std::cout << "#batch:" << n_batches << std::endl;
    std::priority_queue<ReqDDL, std::vector<ReqDDL>, ReqDDLCMP> req_ddls; 
    for (auto& req: reqs) 
        req_ddls.push(ReqDDL(req.id, tpots[req.tpot_idx], tpots[req.tpot_idx]));
    // exit(0);

    assert(req_ddls.size());
    double ddl = 0;
    bool feasible = true;
    int max_prefill_tokens = 0;
    double elasped_time = 0;
    
    std::vector<Batch> batches;
    while (ddl <= t and feasible) {
        int bs = time_to_batch(std::min(t, req_ddls.top().ddl) - ddl);
        if (!bs) break;
        double batch_time = batch_to_time(bs);
        ddl += batch_time;
        elasped_time = ddl;
        batches.push_back({});
        batches.back().prefill_bs = bs;
        std::vector<ReqDDL> new_req_ddls;
        while(not req_ddls.empty() and \
        (req_ddls.top().ddl - req_ddls.top().tpot) < ddl){
            auto req_ddl = req_ddls.top();
            req_ddls.pop();
            assert(req_ddl.ddl >= ddl);
            batches.back().req_batches.push_back(ReqBatch(
                req_ddl.id, false, 1
            ));
            if (--batches.back().prefill_bs < 0) {
                feasible = false;
                break; 
            }
            req_ddl.ddl += req_ddl.tpot;
            new_req_ddls.push_back(req_ddl);
        }
        max_prefill_tokens += batches.back().prefill_bs;
        for (auto& req_ddl: new_req_ddls) 
            req_ddls.push(req_ddl);
    }

    if (not feasible) {
        return {0, 0., {}};
    }

    // double per_batch_time = std::min(t / n_batches, tightest_tpot);
    // int bs = time_to_batch(per_batch_time);
    // per_batch_time = batch_to_time(bs);
    // assert(bs > 0);
    // int max_prefill_tokens = bs * n_batches;
    // std::vector<Batch> batches(n_batches);
    // double elasped_time = n_batches;
    // for (auto& b: batches) b.prefill_bs = bs;
    // double cur_time = per_batch_time;
    // for (auto& b: batches) {
    //     // cur_time <= req_ddls.top().ddl
    //     while (b.prefill_bs and 
    //         req_ddls.top().ddl < cur_time) {
    //         auto req_ddl = req_ddls.top();
    //         req_ddls.pop();
    //         b.req_batches.push_back(ReqBatch(req_ddl.id, false, 1));
    //         b.prefill_bs --;
    //         max_prefill_tokens --;
    //         req_ddl.ddl += req_ddl.tpot;
    //         req_ddls.push(req_ddl);
    //         assert(req_ddl.ddl >= cur_time);
    //     }
    //     cur_time += per_batch_time;
    // }

    if (decode_only) {
        std::unordered_map<int, int> rid2id;
        for (size_t id = 0; id < reqs.size(); ++id){
            rid2id[reqs[id].id] = id;
        }
        
        int extra_token_offset = 0;
        for (size_t bid = 0; bid < batches.size(); ++bid) {
            std::vector<bool> in_batches(reqs.size(), false);
            auto& b = batches[bid];
            for (auto& req_sch: b.req_batches) {
                in_batches[rid2id[req_sch.id]] = true;
            }
            size_t i;
            for (i = 0; i < reqs.size() and (b.prefill_bs > 0); i++) {
                auto tmp = (extra_token_offset + i) % in_batches.size();
                if (!in_batches[tmp]) {
                    b.req_batches.push_back(ReqBatch(reqs[tmp].id, false, 1));
                    b.prefill_bs --;
                }
            }
            extra_token_offset += i;
        }
    }
    return {max_prefill_tokens, elasped_time, batches}; 
}

// Define the State struct
struct State {
    int i;
    double profit;
    int mem;
    int n_prefill_tokens;
    double finish_time;
    std::vector<Batch> batches;
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
    o << "State(i=" << s.i << ",profit=" << s.profit << ",mem=" << s.mem << ",n_prefill_tokens=" 
        << s.n_prefill_tokens << ",finish_time=" << s.finish_time << ",batches=";
    for (auto& b: s.batches)
        o << b << ",";
    std::cout << ",last_state=";
    // if (s.last == nullptr)
    //     o << "last is null";
    // else o << *s.last;
    o << ")";
    return o;
}

std::ostream& operator << (std::ostream& o, Batch& batch) {
    o << "Batch(" << std::endl;
    for (auto& b: batch.req_batches) o << b << std::endl;
    o << "next:" << batch.next << ", prefill: " << batch.prefill_bs;
    o << ")" << std::endl;
    return o;
}

std::ostream& operator << (std::ostream& o, ReqBatch& req) {
    o << "ReqBatch(id=" << req.id << ", is_prefill=" << req.is_prefill << ", n=" << req.n << ")";
    return o;
}

template<typename T>
std::ostream& operator << (std::ostream& o, std::vector<T>& xs) {
    for (auto& x: xs) o << x << ",";
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

// all smaller 
bool operator <= (const std::vector<int>& v1, const std::vector<int>& v2) {
    assert(v1.size() == v2.size());
    for (int i = 0; i < v1.size(); ++i) {
        if (v1[i] > v2[i]) return false;
    }
    return true;
}

void generate_vectors(const std::vector<int>& a, const std::vector<int>& b, 
                      std::vector<int>& c, size_t index, std::vector<std::vector<int>>& result) {
    // Base case: if we've filled all positions in `c`
    if (index == a.size()) {
        result.push_back(c); // Add the generated vector `c` to the results
        return;
    }

    // Recursive case: Iterate over all possible values for c[index] (a[index] to b[index])
    for (int val = a[index]; val <= b[index]; ++val) {
        c[index] = val; // Set the current index of `c`
        generate_vectors(a, b, c, index + 1, result); // Recurse to the next index
    }
}

std::tuple<bool, std::vector<bool>, 
    std::vector<Batch> > PromaxScheduler::_admission_control(
    std::vector<Request>& reqs,
    const int M,
    double current_time
){
    assert(planner);
    Timer timer;
    timer.start();
    std::vector<
        std::unordered_map<int, std::list<State> > > dp;
    dp.resize(reqs.size()+1);

    int tot_old_req = 0;
    for (auto& req: reqs) {
        tot_old_req += 1 - req.is_new_req;
    }
    
    int last_old_req = 0;
    dp[0][0].push_back(State(
           -1,
           0,
           0,
           0,
           current_time,
           nullptr 
    ));
    State* best_state = nullptr;

    std::vector<int> req_cnts(planner->n_tiers(), 0);
    std::vector<int> n_old_acc_reqs(planner->n_tiers(), 0);
    std::vector<int> n_prev_reqs(planner->n_tiers(), 0);
    int n_old = 0;
    for (int i = 0; i < (int) reqs.size(); ++i) {
        req_cnts[reqs[i].tpot_idx] += 1;
    }
    for (int i = 0; i < planner->n_tiers(); ++i) 
        req_cnts[i] += 1;
    Serializer S(req_cnts);
    // std::cout << "S.size()" << S.size() << std::endl;

    // std::cout << "req_cnts:" << req_cnts << std::endl;
    for (size_t n = 1; n <= reqs.size(); n ++) {
        if (n == 33) {
            std::cout << "I am here!" << std::endl;
        }  
        // std::cout << "n:" << n << std::endl;
        timer("before generate vector");
        // dp[n].resize(n+1);
        // dp[n].resize(S.size());
        timer("resize vector");
        auto& req = reqs[n-1];
        // std::cout << "processing, " << req << std::endl;
        dp[n][0].push_back(State(
           -1,
           0,
           0,
           0,
           current_time,
           nullptr
        ));
        // for (int r = n_old + 1; r <= n; r ++) {
        std::vector<std::vector<int> > candidates;
        std::vector<int> tmp(planner->n_tiers());
        
        timer("generate vector");

        // Recursively generate all candidate between n_old_acc_reqs and n_prev_reqs 
        generate_vectors(n_old_acc_reqs, n_prev_reqs, tmp, 0, candidates);

        for (auto& n_reqs: candidates) {
            n_reqs[req.tpot_idx] += 1;
            int r = S.t2i(n_reqs);
            n_reqs[req.tpot_idx] -= 1;
            
            timer("first body");

            auto& states = dp[n][r];

            // std::cout << "n:" << n << ", #reqs:" << n_reqs;
            // std::cout << std::endl;
            int m = 0;
            {
                std::vector<int> tmp_n_reqs(planner->n_tiers(), 0);
                int k = 0;
                while (m < n) {
                    while (tmp_n_reqs[k] >= n_reqs[k])
                        k++;
                    if (k >= planner->n_tiers()) 
                        break;
                    m++;
                    tmp_n_reqs[reqs[m-1].tpot_idx] ++;
                }
                // m = n, or k >= planner->n_tiers()
                assert(k >= planner->n_tiers());
            }
            m = std::max(m, last_old_req);

            timer("first body ends");
            
            // std::cout << "local_n_reqs" << local_n_reqs << std::endl;
            int r_ = S.t2i(n_reqs);

            for (; m < n; m += 1) {
                if (!dp.at(m).count(r_))
                    continue;
                // std::cout << "access " << m << ", " << dp.at(m).size() << ", idx: " << S.t2i(n_reqs) << std::endl;  
                for (auto& state: dp.at(m).at(r_)) {
                    timer("inner dp");
                    int n_prefill_tokens;
                    double elapsed_time;
                    std::vector<Batch> batches;
                    std::vector<Request> acc_reqs;
                    for (auto p_state = &state; p_state->i >= 0; p_state = p_state->last) {
                        acc_reqs.push_back(reqs[p_state->i-1]);
                    }
                    std::tie(n_prefill_tokens, elapsed_time, batches) =\
                     planner->plan(req.ddl - state.finish_time, acc_reqs);
                    timer("plan");
                    
                    if (n_prefill_tokens < 0) continue;
                    // assert(n_prefill_tokens >= 0);
                    // std::cout << "n_prefill_tokens: " << n_prefill_tokens 
                    // << "elasped_time, " << elapsed_time << "batch: " << batch << std::endl;
                    State s(
                        /*.i =*/ n,
                        /*.profit =*/ state.profit + req.profit,
                        /*.mem =*/ state.mem + (req.is_new_req? req.mem:0),
                        /*.prefill_tokens =*/ state.n_prefill_tokens + n_prefill_tokens - req.input_length,
                        /*.finish_time =*/ state.finish_time + elapsed_time,
                        /*.last =*/ &state
                    );

                    if ((s.mem > M) || (s.n_prefill_tokens < 0)) continue;
                    acc_reqs.push_back(req);

                    timer("decode");

                    // the decode cannot be served;
                    if (std::get<1>(planner->plan(INFINITE_TIME, acc_reqs)) == 0.) 
                        continue; 
                        
                    // if (! std::get<0>(planner->plan_decode_only(acc_reqs))) 
                    //     continue;
                    
                    // std::cout << "Req<" << reqs[n-1].id << ">depends on";
                    // std::cout << "Req<" << (m == 0? -1 : reqs[m-1].id) << ">, state<";
                    // for (auto& x: n_reqs) std::cout << x << ",";
                    // std::cout << "> w/ State " << s << std::endl;
                    

                    states.push_back(s);
                    states.back().batches = batches;
                    timer("finish inner");
                }
            }
            _pareto_max(states);
            timer("pareto max");
            for (auto& s: states) {
                if ((tot_old_req == (n_old + 1 - req.is_new_req)) and (
                    !best_state or best_state->profit < s.profit
                )) {
                    best_state = &states.back();
                }
            }
            timer("calc best state");
        }
        n_prev_reqs[req.tpot_idx] += 1;
        if (!req.is_new_req) {
            n_old_acc_reqs[req.tpot_idx] += 1;
            n_old += 1;
            last_old_req = n;
        }
    }
    std::vector<bool> accept_request;
    std::vector<Batch> batches;
    accept_request.resize(reqs.size(), false);

    timer("dp");
    if (not best_state) 
        return {false, accept_request, batches};
    auto state = best_state;
    while (state and state->i != -1){
        if (not (state->i >=1 && state->i <= accept_request.size())) {
            std::cout << "state->i " << state->i << std::endl;
        }
        assert(state->i >=1 && state->i <= accept_request.size());
        accept_request[state->i-1] = true;
        // std::cout << "insert " << state-> batch << std::endl;
        batches.insert(batches.begin(), state->batches.begin(), state->batches.end());
        state = state->last;
    }

    for (int i = 0; i < reqs.size(); ++i) {
        assert(reqs[i].is_new_req or accept_request[i]);
    }
    timer("all");
    // timer.display();
    return {true, accept_request, batches};
}

void PromaxScheduler::_batch_impl(
    const std::vector<Request>& reqs,
    const std::vector<bool>& is_accepted, 
    std::vector<Batch>& batches
){
    // while(not batches.back().req_batches.size())
    //     batches.pop_back();
    /*1. We realize the batch first. */
    assert(reqs.size() == is_accepted.size());

    std::queue<std::pair<int, int> > remain_prefills;
    for (size_t i = 0; i < reqs.size(); i++) {
        if (is_accepted[i] and reqs[i].input_length > 0) 
            remain_prefills.push({reqs[i].id, reqs[i].input_length});
    }
    

    size_t bid = 0;
    while (remain_prefills.size()) {
        int req_id;
        int prefill_tokens;
        std::tie(req_id, prefill_tokens) = remain_prefills.front();
        remain_prefills.pop();
        while (prefill_tokens and bid < batches.size()) {
            auto& batch = batches[bid];
            int n = std::min(batch.prefill_bs, prefill_tokens);
            batch.req_batches.push_back(ReqBatch(req_id, true, n));
            prefill_tokens -= n;
            batch.prefill_bs -= n;
            if (batch.prefill_bs == 0)
                bid ++;
        }
        assert(prefill_tokens == 0);
    }
    // If the last batch has part for prefill, we advance the bid;
    if (bid < batches.size() and (batches[bid].req_batches.size()
        and batches[bid].req_batches.back().is_prefill)) 
        bid++;
    batches.erase(batches.begin() + bid, batches.end());

    std::vector<Request> accepted_reqs;
    for (size_t i = 0; i < reqs.size(); ++i) {
      if (is_accepted[i]) accepted_reqs.push_back(reqs[i]);
    }
    int prefill_tokens;
    double elasped_time;
    std::vector<Batch> last_batches;
    std::tie(prefill_tokens, elasped_time, last_batches) 
        = planner->plan(INFINITE_TIME, accepted_reqs, true);
    if (last_batches.size())
        last_batches.back().next = -last_batches.size() + 1;
    assert(elasped_time >= 0);
    batches.insert(batches.end(), last_batches.begin(), last_batches.end());
}

std::tuple<bool, std::vector<int>, std::vector<Batch> > PromaxScheduler::schedule(
    std::vector<Request>& reqs,
    int M,
    double current_time,
    bool verbose
){
    Timer timer;
    timer.start();
    _verbose = verbose;

    if (verbose) {
        std::cout << "Planner: " << planner->name << std::endl;
        std::cout << "Continuous: " << planner->continuous << std::endl;
        std::cout << "Tpots: ";
        for (double t : planner->tpots) {
            std::cout << t << " ";
        }
        std::cout << "\nHardware Params: ";
        for (double h : planner->hardware_params) {
            std::cout << h << " ";
        }
        std::cout << "max_bs: " << planner->max_bs << std::endl;
        std::cout << "\nAvailable Memory: " << M
                << "\nCurrent Time: " << current_time
                << "\nRequests:\n";

        for (const auto& req : reqs) {
            std::cout << req.id << " " << req.is_new_req << " " << req.ddl << " "
                    << req.input_length << " " << req.profit << " " << req.mem
                    << " " << req.tpot_idx << "\n";
        }
    }

    sort(reqs.begin(), reqs.end(), [](Request& r1, Request& r2){
        return r1.ddl < r2.ddl; 
    });

    if (_verbose) {
        std::cout << "Requests:" << std::endl;
        for (auto& req: reqs)
            std::cout << req << std::endl;
    }
    
    bool is_feasible;
    std::vector<bool> is_accepted;
    std::vector<Batch> batches;
    std::tie(is_feasible, is_accepted, batches) = _admission_control(reqs, M, current_time);
    timer("admission_control");

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
        std::cout << "**************" << std::endl;
    }

    std::vector<int> acc_ids;
    for (int i = 0 ; i < is_accepted.size(); i++) {
        assert(reqs[i].is_new_req or is_accepted[i]);
        if (is_accepted[i]) 
            acc_ids.push_back(reqs[i].id);
    }

    if (!continuous)
        _batch_impl(reqs, is_accepted, batches);
    if (_verbose) {
        std::cout << "schedules" << std::endl;
        for (auto& batch: batches) {
            std::cout << batch << std::endl;
        }
    }
    timer("batch_impl");
    // timer.display();
    /*2. Realize the batch & Maximize the utilization*/
    return {true, acc_ids, batches};
}