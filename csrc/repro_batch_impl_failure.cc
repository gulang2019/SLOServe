#include "adm_ctrl.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct BatchImplDump {
    std::string failing_req_id;
    int remaining_prefill_tokens = 0;
    size_t failing_batch_index = 0;
    std::vector<Request> reqs;
    std::vector<bool> is_accepted;
    std::vector<Batch> batches;

    bool has_schedule_config = false;
    std::string schedule_mode = "fcfs";
    bool schedule_fifo_fair = false;
    bool schedule_continuous = false;
    int schedule_M = 16384;
    double schedule_current_time = 0.0;
    bool schedule_verbose = false;

    bool has_planner_config = false;
    std::string planner_type = "ar";  // ar | sd
    std::vector<double> planner_tpots;
    std::vector<double> planner_hardware;
    bool planner_fixed_bs = false;
    size_t planner_max_bs = 16384;
    double planner_sd_alpha = 0.8;
    int planner_sd_max_sd_size = 15;
    bool planner_sd_fixed_spec = false;
};

namespace {

bool expect_label(std::istream& in, const std::string& expected) {
    std::string got;
    if (!(in >> got)) {
        return false;
    }
    return got == expected;
}

bool load_dump(const std::string& path, BatchImplDump& dump, std::string& err) {
    std::ifstream in(path);
    if (!in.is_open()) {
        err = "failed to open dump file: " + path;
        return false;
    }

    std::string magic;
    if (!std::getline(in, magic) ||
        (magic != "ADM_CTRL_BATCH_IMPL_DUMP_V1" && magic != "ADM_CTRL_BATCH_IMPL_DUMP_V2")) {
        err = "invalid dump format (magic mismatch)";
        return false;
    }

    std::string label;
    while (in >> label) {
        if (label == "schedule_mode") {
            in >> std::quoted(dump.schedule_mode);
            dump.has_schedule_config = true;
        } else if (label == "schedule_fifo_fair") {
            in >> dump.schedule_fifo_fair;
            dump.has_schedule_config = true;
        } else if (label == "schedule_continuous") {
            in >> dump.schedule_continuous;
            dump.has_schedule_config = true;
        } else if (label == "schedule_M") {
            in >> dump.schedule_M;
            dump.has_schedule_config = true;
        } else if (label == "schedule_current_time") {
            in >> dump.schedule_current_time;
            dump.has_schedule_config = true;
        } else if (label == "schedule_verbose") {
            in >> dump.schedule_verbose;
            dump.has_schedule_config = true;
        } else if (label == "planner_type") {
            in >> dump.planner_type;
            dump.has_planner_config = true;
        } else if (label == "planner_fixed_bs") {
            in >> dump.planner_fixed_bs;
            dump.has_planner_config = true;
        } else if (label == "planner_max_bs") {
            in >> dump.planner_max_bs;
            dump.has_planner_config = true;
        } else if (label == "planner_tpots") {
            size_t n = 0;
            in >> n;
            dump.planner_tpots.resize(n);
            for (size_t i = 0; i < n; ++i) in >> dump.planner_tpots[i];
            dump.has_planner_config = true;
        } else if (label == "planner_hardware") {
            size_t n = 0;
            in >> n;
            dump.planner_hardware.resize(n);
            for (size_t i = 0; i < n; ++i) in >> dump.planner_hardware[i];
            dump.has_planner_config = true;
        } else if (label == "planner_sd_alpha") {
            in >> dump.planner_sd_alpha;
            dump.has_planner_config = true;
        } else if (label == "planner_sd_max_sd_size") {
            in >> dump.planner_sd_max_sd_size;
            dump.has_planner_config = true;
        } else if (label == "planner_sd_fixed_spec") {
            in >> dump.planner_sd_fixed_spec;
            dump.has_planner_config = true;
        } else if (label == "planner_name" || label == "planner_continuous") {
            // Present in V2 dumps but not needed by reproducer.
            std::string ignored;
            if (label == "planner_name") in >> std::quoted(ignored);
            else in >> ignored;
        } else if (label == "failing_req_id") {
            in >> std::quoted(dump.failing_req_id);
            break;
        } else {
            err = "unexpected header label: " + label;
            return false;
        }
    }
    if (dump.failing_req_id.empty()) {
        err = "missing failing_req_id";
        return false;
    }

    if (!expect_label(in, "remaining_prefill_tokens")) {
        err = "missing remaining_prefill_tokens";
        return false;
    }
    in >> dump.remaining_prefill_tokens;

    if (!expect_label(in, "failing_batch_index")) {
        err = "missing failing_batch_index";
        return false;
    }
    in >> dump.failing_batch_index;

    size_t n_reqs = 0;
    if (!expect_label(in, "reqs")) {
        err = "missing reqs";
        return false;
    }
    in >> n_reqs;
    dump.reqs.clear();
    dump.reqs.reserve(n_reqs);
    for (size_t i = 0; i < n_reqs; ++i) {
        Request req;
        std::string label;
        if (!(in >> label) || label != "req") {
            err = "missing req row";
            return false;
        }
        in >> std::quoted(req.id)
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
        if (!in.good()) {
            err = "failed parsing req row";
            return false;
        }
        dump.reqs.push_back(req);
    }

    size_t n_acc = 0;
    if (!expect_label(in, "accepted")) {
        err = "missing accepted";
        return false;
    }
    in >> n_acc;
    dump.is_accepted.clear();
    dump.is_accepted.reserve(n_acc);
    for (size_t i = 0; i < n_acc; ++i) {
        std::string label;
        bool accepted = false;
        if (!(in >> label >> accepted) || label != "acc") {
            err = "missing acc row";
            return false;
        }
        dump.is_accepted.push_back(accepted);
    }

    size_t n_batches = 0;
    if (!expect_label(in, "batches")) {
        err = "missing batches";
        return false;
    }
    in >> n_batches;
    dump.batches.clear();
    dump.batches.reserve(n_batches);
    for (size_t i = 0; i < n_batches; ++i) {
        std::string label;
        Batch batch;
        size_t n_req_batches = 0;
        if (!(in >> label) || label != "batch") {
            err = "missing batch row";
            return false;
        }
        in >> batch.prefill_bs >> batch.next >> batch.estimated_time >> n_req_batches;
        if (!in.good()) {
            err = "failed parsing batch row";
            return false;
        }
        for (size_t j = 0; j < n_req_batches; ++j) {
            ReqBatch req_batch("", false, 0);
            if (!(in >> label) || label != "req_batch") {
                err = "missing req_batch row";
                return false;
            }
            in >> std::quoted(req_batch.id) >> req_batch.is_prefill >> req_batch.n;
            if (!in.good()) {
                err = "failed parsing req_batch row";
                return false;
            }
            batch.req_batches.push_back(req_batch);
        }
        dump.batches.push_back(batch);
    }

    if (dump.reqs.size() != dump.is_accepted.size()) {
        err = "dump mismatch: reqs.size() != is_accepted.size()";
        return false;
    }
    return true;
}

struct ReproConfig {
    std::string mode = "fcfs";
    bool fifo_fair = false;
    bool continuous = false;
    std::string planner = "ar";  // ar | sd
    std::vector<double> tpots;
    std::vector<double> hardware;
    bool fixed_bs = false;
    size_t max_bs = 16384;

    // sd planner only
    double alpha = 0.8;
    int max_sd_size = 15;
    bool fixed_spec = false;

    int M = 16384;
    double current_time = 0.0;
    bool verbose = true;
};

std::vector<double> parse_csv_doubles(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) {
            continue;
        }
        out.push_back(std::stod(tok));
    }
    return out;
}

bool parse_bool01(const std::string& s) {
    if (s == "1" || s == "true" || s == "True") return true;
    if (s == "0" || s == "false" || s == "False") return false;
    throw std::invalid_argument("invalid bool value: " + s + " (expected 0/1/true/false)");
}

void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <dump_file> [options]\n\n"
        << "Options:\n"
        << "  --mode <fcfs|dp|edf|edf_sim>\n"
        << "  --fifo-fair <0|1>\n"
        << "  --continuous <0|1>\n"
        << "  --planner <ar|sd>\n"
        << "  --tpots <v1,v2,...>               (optional, overrides dump)\n"
        << "  --hardware <k1,k2,...>            (optional, overrides dump)\n"
        << "  --fixed-bs <0|1>\n"
        << "  --max-bs <int>\n"
        << "  --alpha <double>                  (sd only)\n"
        << "  --max-sd-size <int>               (sd only)\n"
        << "  --fixed-spec <0|1>                (sd only)\n"
        << "  --M <int>\n"
        << "  --current-time <double>\n"
        << "  --verbose <0|1>\n";
}

bool parse_args(int argc, char** argv, std::string& dump_file, ReproConfig& cfg, std::string& err) {
    if (argc < 2) {
        err = "missing dump_file";
        return false;
    }
    dump_file = argv[1];

    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_val = [&](const std::string& k) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + k);
            }
            return std::string(argv[++i]);
        };

        try {
            if (key == "--mode") cfg.mode = need_val(key);
            else if (key == "--fifo-fair") cfg.fifo_fair = parse_bool01(need_val(key));
            else if (key == "--continuous") cfg.continuous = parse_bool01(need_val(key));
            else if (key == "--planner") cfg.planner = need_val(key);
            else if (key == "--tpots") cfg.tpots = parse_csv_doubles(need_val(key));
            else if (key == "--hardware") cfg.hardware = parse_csv_doubles(need_val(key));
            else if (key == "--fixed-bs") cfg.fixed_bs = parse_bool01(need_val(key));
            else if (key == "--max-bs") cfg.max_bs = static_cast<size_t>(std::stoll(need_val(key)));
            else if (key == "--alpha") cfg.alpha = std::stod(need_val(key));
            else if (key == "--max-sd-size") cfg.max_sd_size = std::stoi(need_val(key));
            else if (key == "--fixed-spec") cfg.fixed_spec = parse_bool01(need_val(key));
            else if (key == "--M") cfg.M = std::stoi(need_val(key));
            else if (key == "--current-time") cfg.current_time = std::stod(need_val(key));
            else if (key == "--verbose") cfg.verbose = parse_bool01(need_val(key));
            else {
                throw std::invalid_argument("unknown option: " + key);
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    return true;
}

void apply_dump_defaults(const BatchImplDump& dump, ReproConfig& cfg) {
    if (dump.has_schedule_config) {
        cfg.mode = dump.schedule_mode;
        cfg.fifo_fair = dump.schedule_fifo_fair;
        cfg.continuous = dump.schedule_continuous;
        cfg.M = dump.schedule_M;
        cfg.current_time = dump.schedule_current_time;
        cfg.verbose = dump.schedule_verbose;
    }

    if (dump.has_planner_config) {
        cfg.planner = dump.planner_type;
        cfg.fixed_bs = dump.planner_fixed_bs;
        cfg.max_bs = dump.planner_max_bs;
        if (cfg.tpots.empty()) cfg.tpots = dump.planner_tpots;
        if (cfg.hardware.empty()) cfg.hardware = dump.planner_hardware;
        cfg.alpha = dump.planner_sd_alpha;
        cfg.max_sd_size = dump.planner_sd_max_sd_size;
        cfg.fixed_spec = dump.planner_sd_fixed_spec;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string dump_file;
    ReproConfig cfg;
    std::string arg_err;
    if (!parse_args(argc, argv, dump_file, cfg, arg_err)) {
        std::cerr << "Argument error: " << arg_err << std::endl;
        print_usage(argv[0]);
        return 2;
    }

    BatchImplDump dump;
    std::string err;
    if (!load_dump(dump_file, dump, err)) {
        std::cerr << "Failed to load dump: " << err << std::endl;
        return 1;
    }

    std::cout
        << "Loaded dump with reqs=" << dump.reqs.size()
        << ", accepted=" << dump.is_accepted.size()
        << ", batches=" << dump.batches.size()
        << ", failing_req_id=" << dump.failing_req_id
        << ", remaining_prefill_tokens=" << dump.remaining_prefill_tokens
        << ", failing_batch_index=" << dump.failing_batch_index
        << std::endl;

    apply_dump_defaults(dump, cfg);
    if (cfg.tpots.empty() || cfg.hardware.empty()) {
        std::cerr << "Missing planner tpots/hardware. Provide --tpots and --hardware." << std::endl;
        return 2;
    }

    std::cout
        << "Running admission_control() + schedule() with mode=" << cfg.mode
        << ", fifo_fair=" << cfg.fifo_fair
        << ", continuous=" << cfg.continuous
        << ", planner=" << cfg.planner
        << ", M=" << cfg.M
        << ", current_time=" << cfg.current_time
        << ", verbose=" << cfg.verbose
        << std::endl;

    AdmCtrlScheduler scheduler(cfg.mode, 16, cfg.fifo_fair, cfg.continuous);
    if (cfg.planner == "ar") {
        scheduler.set_ar_planner(cfg.tpots, cfg.hardware, cfg.fixed_bs, cfg.max_bs);
    } else if (cfg.planner == "sd") {
        scheduler.set_sd_planner(
            cfg.tpots, cfg.hardware, cfg.fixed_bs, cfg.max_bs,
            cfg.alpha, cfg.max_sd_size, cfg.fixed_spec);
    } else {
        std::cerr << "Invalid planner: " << cfg.planner << " (expected ar|sd)" << std::endl;
        return 2;
    }

    // This may intentionally assert if the original bug is reproduced.
    auto reqs = dump.reqs;
    auto [is_feasible, is_accepted] = scheduler.admission_control(
        reqs, cfg.M, cfg.current_time);
    std::vector<Request> accepted_reqs;
    accepted_reqs.reserve(reqs.size());
    size_t accepted_count = 0;
    for (size_t i = 0; i < reqs.size() && i < is_accepted.size(); ++i) {
        if (!is_accepted[i]) {
            continue;
        }
        accepted_reqs.push_back(reqs[i]);
        ++accepted_count;
    }
    auto [schedule_feasible, batches] = is_feasible
        ? scheduler.schedule(accepted_reqs, cfg.current_time, 1, cfg.verbose)
        : std::make_tuple(false, std::vector<Batch>{});
    std::cout
        << "admission_control() returned is_feasible=" << is_feasible
        << ", schedule_is_feasible=" << schedule_feasible
        << ", accepted=" << accepted_count
        << ", batches=" << batches.size()
        << std::endl;

    return 0;
}
