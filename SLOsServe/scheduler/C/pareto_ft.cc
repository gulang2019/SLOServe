#include <vector>
#include <iostream>
#include <algorithm>

// Define the Resource struct
struct Resource {
    double profit;
    int mem;
    int n_prefill_tokens;
    double finish_time;

    Resource(double p, int m, int n, double f)
        : profit(p), mem(m), n_prefill_tokens(n), finish_time(f) {}
};

// Check if one Resource dominates another
bool dominates(const Resource& r1, const Resource& r2) {
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
void _pareto_max(std::vector<Resource>& resources) {
    std::vector<Resource> frontier;

    for (size_t i = 0; i < resources.size(); ++i) {
        bool is_dominated = false;
        for (size_t j = 0; j < resources.size(); ++j) {
            if (i != j && dominates(resources[j], resources[i])) {
                is_dominated = true;
                break;
            }
        }
        if (!is_dominated) {
            frontier.push_back(resources[i]);
        }
    }

    // Replace input vector with Pareto frontier
    resources = frontier;
}

int main() {
    // Example resources
    std::vector<Resource> resources = {
        {10.5, 100, 50, 2.5},
        {15.0, 200, 40, 2.0},
        {12.0, 150, 60, 1.8},
        {10.0, 100, 50, 3.0},
        {18.0, 250, 30, 2.2}
    };

    // Calculate Pareto frontier
    _pareto_max(resources);

    // Print the Pareto frontier
    std::cout << "Pareto Frontier:\n";
    for (const auto& r : resources) {
        std::cout << "Profit: " << r.profit
                  << ", Mem: " << r.mem
                  << ", Prefill Tokens: " << r.n_prefill_tokens
                  << ", Finish Time: " << r.finish_time << "\n";
    }

    return 0;
}
