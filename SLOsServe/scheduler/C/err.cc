#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <tuple>
#include <chrono>
#include <assert.h>
#include "promax_spec_sch.h"

// Assuming the Request struct is already defined
// struct Request {
//     int id;
//     bool is_new_req;
//     float ddl;
//     int input_length;
//     float profit;
//     int mem;
//     int tpot_idx;

//     Request(int id, bool is_new_req, float ddl, int input_length, float profit, int mem, int tpot_idx)
//         : id(id), is_new_req(is_new_req), ddl(ddl), input_length(input_length), profit(profit), mem(mem), tpot_idx(tpot_idx) {}
// };

int main(int argc, char** argv) {
    const char* filename = "error.in";
    if (argc == 2)
        filename = argv[1];
    std::cout << "file:" << filename << ":";
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Variables to read
    std::vector<double> tpots;
    std::vector<double> hardware_params;
    double spec_decode_alpha;
    int max_spec_decode_size;
    int n_avail;
    double current_time;
    int num_requests;
    std::vector<Request> requests;

    // Helper function to parse a line of numbers into a vector
    auto parse_vector = [](std::string line) -> std::vector<double> {
        std::stringstream ss(line);
        int n;
        ss >> n;
        std::vector<double> result(n);
        for (int i = 0; i < n; ++i) {
            ss >> result[i];
        }
        return result;
    };

    // Read tpots
    std::string line;
    std::getline(infile, line);
    tpots = parse_vector(line);

    // Read hardware_params
    std::getline(infile, line);
    hardware_params = parse_vector(line);

    // Read spec_decode_alpha and max_spec_decode_size
    infile >> spec_decode_alpha >> max_spec_decode_size;

    // Read n_avail, current_time, and num_requests
    infile >> n_avail >> current_time >> num_requests;

    // Read each request
    for (int i = 0; i < num_requests; ++i) {
        int id, input_length, mem, tpot_idx;
        bool is_new_req;
        double ddl, profit;
        infile >> id >> is_new_req >> ddl >> input_length >> profit >> mem >> tpot_idx;
        requests.emplace_back(id, is_new_req, ddl, input_length, profit, mem, tpot_idx);
    }

    // Close the file
    infile.close();

    // Debug output
    // std::cout << "Tpots: ";
    // for (double t : tpots) {
    //     std::cout << t << " ";
    // }
    // std::cout << "\nHardware Params: ";
    // for (double h : hardware_params) {
    //     std::cout << h << " ";
    // }
    // std::cout << "\nSpec Decode Alpha: " << spec_decode_alpha
    //           << "\nMax Spec Decode Size: " << max_spec_decode_size
    //           << "\nAvailable Memory: " << n_avail
    //           << "\nCurrent Time: " << current_time
    //           << "\nRequests:\n";

    // for (const auto& req : requests) {
    //     std::cout << req.id << " " << req.is_new_req << " " << req.ddl << " "
    //               << req.input_length << " " << req.profit << " " << req.mem
    //               << " " << req.tpot_idx << "\n";
    // }

    PromaxScheduler scheduler(false);

    // scheduler.set_sd_planner(tpots, 
    //     hardware_params, 
    //     false, 
    //     spec_decode_alpha, 
    //     max_spec_decode_size,
    //     false);
    
    scheduler.set_ar_planner(
        tpots, 
        hardware_params, 
        false
    );

    // assert(false);

    bool feasible;
    std::vector<int> acc_ids;
    std::vector<Batch> accepted_batches;
    std::tie(feasible, acc_ids, accepted_batches) = scheduler.schedule(requests, n_avail, current_time, false);
    std::cout << "feasible: " << feasible  << std::endl;
    // for (auto& batch: accepted_batches) {
    //     std::cout << batch << std::endl;
    // }
    // std::cout << "acc: ";
    // for (auto id: acc_ids) {
    //     std::cout << id << ",";
    // }
    // std::cout << std::endl;

    // auto start = std::chrono::high_resolution_clock::now();
    // std::tie(feasible, acc_ids, accepted_batches) = scheduler.schedule(requests, n_avail, current_time, false);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;

    // std::cout << duration.count() << std::endl;

    return 0;
}
