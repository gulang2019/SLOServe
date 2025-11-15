#include <queue>

#include "adm_ctrl.h"

#define INFINITE_TIME 1

std::vector<RequestOutput> AdmCtrlRouter::schedule(
    std::vector<Request>& reqs,
    std::vector<int>& Ms,
    double current_time,
    bool verbose
) {
    
    reqs = std::sort(reqs.begin(), reqs.end(), [](Request& r1, Request& r2){
        return r1.ddl < r2.ddl; 
    });

    for (auto& req: reqs) {
        req.ddl -= current_time;
    }

    // std::vector<bool> is_accepted;
    std::vector<RequestOutput> request_outputs;

    struct State {
        int mem;
        int n_prefill_tokens;
        double finish_time;
        std::vector<Request> acc_reqs;

        State(int mem, int n_prefill_tokens, double finish_time, 
            const std::vector<Request>& acc_reqs)
            : mem(mem), n_prefill_tokens(n_prefill_tokens), \
            finish_time(finish_time), acc_reqs(acc_reqs) {}
    };

    std::vector<State> states;
    for (int i = 0; i < n_devices; i++) {
        states.push_back({Ms[i], 0.0, 0.0, {}});
    }

    auto acc_req_fn = [&](const State& current_state, const Request& req, State& new_state, bool is_prefill) {
        // If the request is a prefill request, we need some prefill budget;
        if ((is_prefill and (current_state.n_prefill_tokens < req.input_length))) {
            int new_prefill_budget;
            double new_elapsed_time; 
            std::vector<Batch> new_batches;
            std::tie(new_prefill_budget, new_elapsed_time, new_batches) = \
                planner->plan(req.ddl - current_state.finish_time, current_state.acc_reqs, false, current_state.finish_time);
            
            new_state.n_prefill_tokens += new_prefill_budget;
            new_state.finish_time += new_elapsed_time;
        }

        if (is_prefill) {
            new_state.n_prefill_tokens -= req.input_length;
            new_state.mem -= req.prefill_mem;
        }
        else {
            new_state.acc_reqs.push_back(req);
            new_state.mem -= req.mem;
        }
    };

    auto is_valid_fn = [](const State&state) {
        return (state.n_prefill_tokens >= 0) and (state.mem >= 0);
    };

    auto is_compatible_fn = [&](
        const State& state, const DeviceState& device_state, bool is_prefill) {
        State new_state = state;
        acc_req_fn(state, req, new_state, is_prefill);

        if (not is_valid_fn(new_state)) {
            return false; 
        }

        // 2. check compatibility with the following requests;
        for (auto& [j, isP, ddl]: device_state.rids_isP_ddl) {
            auto& req_j = reqs[j];
            acc_req_fn(new_state, req_j, new_state, isP);
            if (not is_valid_fn(new_state)) {
                return false;
            }
        }

        if (std::get<1>(planner->plan(INFINITE_TIME, new_state.acc_reqs)) == 0.) {
            return false;
        }

        return true;
    };

    struct DeviceState {
        double earlist_decode_start_time;
        std::vector<std::tuple<int, bool, double> > rids_isP_ddl;
        int n_decode_reqs;
    };

    std::vector<DeviceState> device_states(n_devices);
    for (size_t i = 0; i < n_devices; i++) {
        device_states[i].i = 0;
        device_states[i].earlist_decode_start_time = INFINITE_TIME;
    }

    for (size_t i = 0; i < reqs.size(); i++) {
        auto& req = reqs[i];
        if (not req.is_new_req) {
            device_states[req.prefill_device_id].rids_isP_ddl.push_back({i, true, req.ddl});
            device_states[req.decode_device_id].rids_isP_ddl.push_back({i, false, req.ddl});
        }
    }

    std::vector<int> device_ids;
    for (int i = 0; i < n_devices; i++) {
        device_ids.push_back(i);
    }

    for (auto& req: reqs) {
        if (req.is_new_req) {
            int decode_device_id = -1;
            int prefill_device_id = -1;

            std::sort(device_ids.begin(), device_ids.end(), [&](int d1, int d2) {
                return device_states[d1].earlist_decode_start_time < device_states[d2].earlist_decode_start_time;
            });

            
            for (auto& device_id: device_ids) {
                if (is_compatible_fn(states[device_id], device_states[device_id], false)) {
                    decode_device_id = device_id;
                    break;
                }
            }
            
            if (decode_device_id == -1) {
                request_outputs.push_back({false, -1, -1});
                continue;
            }
            
            std::sort(device_ids.begin(), device_ids.end(), [&](int d1, int d2) {
                return device_states[d1].n_decode_reqs < device_states[d2].n_decode_reqs;
            });

            for (auto& device_id: device_ids) {
                // 1. try to accept the request;
                if (is_compatible_fn(states[device_id], device_states[device_id], false)) {
                    prefill_device_id = device_id;
                    break;
                }
            }

            if (prefill_device_id == -1) {
                request_outputs.push_back({false, -1, -1});
                continue;
            }

            acc_req_fn(states[prefill_device_id], req, states[prefill_device_id], true);
            acc_req_fn(states[decode_device_id], req, states[decode_device_id], false);

            request_outputs.push_back({true, prefill_device_id, decode_device_id});

            auto& device_state = device_states[decode_device_id];
            device_state.earlist_decode_start_time = std::min(device_state.earlist_decode_start_time, req.ddl);
            device_state.n_decode_reqs += 1;
        }
        else {
            acc_req_fn(states[req.prefill_device_id], req, states[req.prefill_device_id], true);
            acc_req_fn(states[req.decode_device_id], req, states[req.decode_device_id], false);
            device_states[req.prefill_device_id].rids_isP_ddl.pop_front();
            device_states[req.decode_device_id].rids_isP_ddl.pop_front();
        }
    }

    return request_outputs;
}
