#include <promax_spec_sch.h>
#include <iostream>
#include <chrono>  // For timing utilities


int main() {
    std::vector<double> tpots = {4,};
    std::vector<double> hardware_params = {0.0, 0., 4, 1., 0., 0.0};
    double spec_decode_alpha = 0.8;
    int max_spec_decode_size = 6;

    AdmCtrlScheduler scheduler;

    


    std::vector<Request> reqs = {
        Request(0, true, 8, 1, 1, 3, 0),
        Request(1, true, 12, 2, 20, 3, 0),
        Request(2, true, 2, 2, 10, 3, 0),
        Request(3, true, 2, 2, 1, 3, 0),
        Request(4, false, 0, 0, 1, 3, 0),
        Request(5, false, 0, 0, 1, 3, 0),
        // Request(6, false, 4, 2, 1, 3, 0),
    };
    std::sort(reqs.begin(), reqs.end(), [](Request& r1, Request& r2){
        return r1.ddl < r2.ddl;
    });
    auto start = std::chrono::high_resolution_clock::now();
    bool is_feasible = false;
    std::vector<int> acc_ids;
    std::vector<Batch> batches;
    // scheduler.set_ar_planner(tpots, hardware_params, false);
    scheduler.set_sd_planner(tpots, hardware_params, false, 0.9, 10, false);
    std::tie(is_feasible, acc_ids, batches) = scheduler.schedule(reqs, 100, 0., true);
    auto end = std::chrono::high_resolution_clock::now();
    

    // Calculate the elapsed time
    // auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Display the elapsed time
    // std::cout << "Elapsed time in seconds: " << elapsed_seconds.count() << " s\n";
    // std::cout << "Elapsed time in milliseconds: " << elapsed_milliseconds.count() << " ms\n";
    std::cout << "Elapsed time in microseconds: " << elapsed_microseconds.count() << " µs\n";
    std::cout << "is_feasible:" << is_feasible << std::endl;
    std::cout << "batches:";

    for (auto& batch: batches) {
        std::cout << batch << std::endl;
    }
    std::cout << std::endl;

    return 0;
}