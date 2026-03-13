#include "adm_ctrl.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

struct ScheduleDump {
    double timestamp = 0.0;
    int did = -1;
    std::string mode = "normal";
    std::string scheduler_mode = "edf_sim";
    bool scheduler_fifo_fair = false;
    bool scheduler_continuous = false;
    std::string planner_type = "ar";
    bool planner_fixed_bs = false;
    int planner_max_bs = 16384;
    std::vector<double> tpots;
    std::vector<double> hardware_params;
    int M = 0;
    double current_time = 0.0;
    double max_time = 10.0;

    bool observed_is_feasible = false;
    double observed_schedule_elapsed_s = 0.0;
    double observed_total_elapsed_s = 0.0;
    std::vector<std::string> observed_accepted_ids;
    std::vector<Request> reqs;
};

namespace {

bool expect_label(std::istream& in, const std::string& expected) {
    std::string got;
    if (!(in >> got)) {
        return false;
    }
    return got == expected;
}

bool load_dump(const std::string& path, ScheduleDump& dump, std::string& err) {
    std::ifstream in(path);
    if (!in.is_open()) {
        err = "failed to open dump file: " + path;
        return false;
    }

    std::string magic;
    if (!std::getline(in, magic) || magic != "SLOPACKER_SCHEDULE_DUMP_V1") {
        err = "invalid dump format (magic mismatch)";
        return false;
    }

    if (!expect_label(in, "timestamp")) { err = "missing timestamp"; return false; }
    in >> dump.timestamp;
    if (!expect_label(in, "did")) { err = "missing did"; return false; }
    in >> dump.did;
    if (!expect_label(in, "mode")) { err = "missing mode"; return false; }
    in >> std::quoted(dump.mode);
    if (!expect_label(in, "scheduler_mode")) { err = "missing scheduler_mode"; return false; }
    in >> std::quoted(dump.scheduler_mode);
    if (!expect_label(in, "scheduler_fifo_fair")) { err = "missing scheduler_fifo_fair"; return false; }
    in >> dump.scheduler_fifo_fair;
    if (!expect_label(in, "scheduler_continuous")) { err = "missing scheduler_continuous"; return false; }
    in >> dump.scheduler_continuous;
    if (!expect_label(in, "planner_type")) { err = "missing planner_type"; return false; }
    in >> std::quoted(dump.planner_type);
    if (!expect_label(in, "planner_fixed_bs")) { err = "missing planner_fixed_bs"; return false; }
    in >> dump.planner_fixed_bs;
    if (!expect_label(in, "planner_max_bs")) { err = "missing planner_max_bs"; return false; }
    in >> dump.planner_max_bs;

    size_t n_tpots = 0;
    if (!expect_label(in, "tpots")) { err = "missing tpots"; return false; }
    in >> n_tpots;
    dump.tpots.resize(n_tpots);
    for (size_t i = 0; i < n_tpots; ++i) in >> dump.tpots[i];

    size_t n_hw = 0;
    if (!expect_label(in, "hardware_params")) { err = "missing hardware_params"; return false; }
    in >> n_hw;
    dump.hardware_params.resize(n_hw);
    for (size_t i = 0; i < n_hw; ++i) in >> dump.hardware_params[i];

    if (!expect_label(in, "M")) { err = "missing M"; return false; }
    in >> dump.M;
    if (!expect_label(in, "current_time")) { err = "missing current_time"; return false; }
    in >> dump.current_time;
    std::string next_label;
    if (!(in >> next_label)) { err = "missing observed_is_feasible"; return false; }
    if (next_label == "max_time") {
        in >> dump.max_time;
        if (!expect_label(in, "observed_is_feasible")) { err = "missing observed_is_feasible"; return false; }
    } else if (next_label != "observed_is_feasible") {
        err = "unexpected label after current_time: " + next_label;
        return false;
    }
    in >> dump.observed_is_feasible;
    if (!expect_label(in, "observed_schedule_elapsed_s")) { err = "missing observed_schedule_elapsed_s"; return false; }
    in >> dump.observed_schedule_elapsed_s;
    if (!expect_label(in, "observed_total_elapsed_s")) { err = "missing observed_total_elapsed_s"; return false; }
    in >> dump.observed_total_elapsed_s;

    size_t n_acc = 0;
    if (!expect_label(in, "observed_accepted_ids")) { err = "missing observed_accepted_ids"; return false; }
    in >> n_acc;
    dump.observed_accepted_ids.clear();
    dump.observed_accepted_ids.reserve(n_acc);
    for (size_t i = 0; i < n_acc; ++i) {
        std::string label;
        std::string id;
        if (!(in >> label) || label != "accepted_id") {
            err = "missing accepted_id row";
            return false;
        }
        in >> std::quoted(id);
        dump.observed_accepted_ids.push_back(id);
    }

    size_t n_reqs = 0;
    if (!expect_label(in, "reqs")) { err = "missing reqs"; return false; }
    in >> n_reqs;
    dump.reqs.clear();
    dump.reqs.reserve(n_reqs);
    for (size_t i = 0; i < n_reqs; ++i) {
        std::string line;
        Request req;
        if (!std::getline(in >> std::ws, line)) {
            err = "missing req row";
            return false;
        }
        std::istringstream row(line);
        std::string label;
        if (!(row >> label) || label != "req") {
            err = "missing req row";
            return false;
        }
        row >> std::quoted(req.id)
            >> req.is_new_req
            >> req.ddl
            >> req.input_length
            >> req.n_computed_tokens
            >> req.profit
            >> req.mem
            >> req.tpot_idx
            >> req.prefill_mem
            >> req.prefill_device_id
            >> req.decode_device_id
            >> req.prefill_only
            >> req.arrival_time;
        if (!row.good()) {
            err = "failed parsing req row";
            return false;
        }
        req.max_tokens = req.prefill_only ? 0 : -1;
        if (!(row >> req.max_tokens)) {
            row.clear();
        }
        dump.reqs.push_back(req);
    }

    return true;
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <dump.txt> [repeat]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        print_usage(argv[0]);
        return 2;
    }
    const std::string dump_path = argv[1];
    int repeat = 1;
    if (argc == 3) {
        repeat = std::max(1, std::stoi(argv[2]));
    }

    ScheduleDump dump;
    std::string err;
    if (!load_dump(dump_path, dump, err)) {
        std::cerr << "Load dump failed: " << err << "\n";
        return 1;
    }
    if (dump.planner_type != "ar") {
        std::cerr << "Unsupported planner_type: " << dump.planner_type << " (expected ar)\n";
        return 1;
    }

    AdmCtrlScheduler scheduler(dump.scheduler_mode, 16, dump.scheduler_fifo_fair, dump.scheduler_continuous);
    auto tpots = dump.tpots;
    auto hardware = dump.hardware_params;
    scheduler.set_ar_planner(tpots, hardware, dump.planner_fixed_bs, static_cast<size_t>(dump.planner_max_bs));

    bool replay_is_feasible = false;
    std::vector<std::string> replay_accepted_ids;
    std::vector<Batch> replay_batches;
    double elapsed_sum_s = 0.0;
    double elapsed_min_s = 1e18;
    double elapsed_max_s = 0.0;

    for (int i = 0; i < repeat; ++i) {
        std::vector<Request> reqs = dump.reqs;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto [is_schedule_feasible, out] = scheduler.schedule(reqs, dump.current_time, dump.max_time, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        elapsed_sum_s += elapsed;
        elapsed_min_s = std::min(elapsed_min_s, elapsed);
        elapsed_max_s = std::max(elapsed_max_s, elapsed);

        replay_batches = std::move(out);
        std::cout << "Iteration " << (i + 1) << "/" << repeat
                  << ": replay_is_feasible=" << replay_is_feasible
                  << " replay_accepted_n=" << replay_accepted_ids.size()
                  << " elapsed_s=" << elapsed
                  << " replay_batches=" << replay_batches.size()
                  << std::endl;
    }

    std::unordered_set<std::string> observed_set(dump.observed_accepted_ids.begin(), dump.observed_accepted_ids.end());
    std::unordered_set<std::string> replay_set(replay_accepted_ids.begin(), replay_accepted_ids.end());
    bool accepted_match = observed_set == replay_set;
    double replay_avg = elapsed_sum_s / repeat;

    std::cout << "==== input ====\n";
    std::cout << "dump_path " << dump_path << "\n";
    std::cout << "did " << dump.did << " mode " << dump.mode << "\n";
    std::cout << "scheduler_mode " << dump.scheduler_mode
              << " fifo_fair " << dump.scheduler_fifo_fair
              << " continuous " << dump.scheduler_continuous << "\n";
    std::cout << "planner_type " << dump.planner_type
              << " fixed_bs " << dump.planner_fixed_bs
              << " max_bs " << dump.planner_max_bs << "\n";
    std::cout << "tpots_n " << dump.tpots.size() << " tpots";
    for (double x : dump.tpots) std::cout << " " << x;
    std::cout << "\n";
    std::cout << "hardware_params_n " << dump.hardware_params.size() << " hardware_params";
    for (double x : dump.hardware_params) std::cout << " " << x;
    std::cout << "\n";
    std::cout << "n_reqs " << dump.reqs.size() << " M " << dump.M
              << " max_time " << dump.max_time << "\n";
    int n_new = 0;
    int n_running = 0;
    int n_prefill_only = 0;
    for (const auto& req : dump.reqs) {
        if (req.is_new_req) ++n_new;
        else ++n_running;
        if (req.prefill_only) ++n_prefill_only;
    }
    std::cout << "n_new_reqs " << n_new
              << " n_running_reqs " << n_running
              << " n_prefill_only " << n_prefill_only << "\n";
    const size_t kPrintReqs = 8;
    std::cout << "req_samples_n " << std::min(kPrintReqs, dump.reqs.size()) << "\n";
    for (size_t i = 0; i < dump.reqs.size() && i < kPrintReqs; ++i) {
        const auto& req = dump.reqs[i];
        std::cout << "req[" << i << "]"
                  << " id=" << req.id
                  << " is_new=" << req.is_new_req
                  << " ddl=" << req.ddl
                  << " input_len=" << req.input_length
                  << " n_computed=" << req.n_computed_tokens
                  << " mem=" << req.mem
                  << " prefill_mem=" << req.prefill_mem
                  << " prefill_only=" << req.prefill_only
                  << " arrival=" << req.arrival_time
                  << " max_tokens=" << req.max_tokens
                  << "\n";
    }
    std::cout << "==== replay result ====\n";
    std::cout << "observed_is_feasible " << dump.observed_is_feasible
              << " replay_is_feasible " << replay_is_feasible << "\n";
    std::cout << "observed_accepted_n " << dump.observed_accepted_ids.size()
              << " replay_accepted_n " << replay_accepted_ids.size()
              << " accepted_match " << accepted_match << "\n";
    std::cout << "observed_schedule_elapsed_s " << dump.observed_schedule_elapsed_s << "\n";
    std::cout << "replay_schedule_elapsed_avg_s " << replay_avg << "\n";
    std::cout << "replay_schedule_elapsed_min_s " << elapsed_min_s << "\n";
    std::cout << "replay_schedule_elapsed_max_s " << elapsed_max_s << "\n";
    if (dump.observed_schedule_elapsed_s > 0.0) {
        std::cout << "replay_vs_observed_ratio " << (replay_avg / dump.observed_schedule_elapsed_s) << "\n";
    }
    std::cout << "replay_batches " << replay_batches.size() << "\n";
    for (auto& batch : replay_batches) {
        std::cout << "batch prefill_bs=" << batch.prefill_bs
                  << " next=" << batch.next
                  << " est_time=" << batch.estimated_time
                  << " req_batches=" << batch.req_batches.size() << "\n";
        for (auto& req_batch : batch.req_batches) {
            std::cout << "  req_batch id=" << req_batch.id
                      << " is_prefill=" << req_batch.is_prefill
                      << " n=" << req_batch.n << "\n";
        }
    }
    return 0;
}
