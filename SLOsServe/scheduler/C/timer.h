#pragma once
#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::unordered_map<std::string, double> times;
    std::string last_s = "START";

    double current_time() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        return elapsed.count();
    }

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        times.clear();
        last_s = "START";
    }

    void operator()(const std::string& s) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;

        std::string key = last_s + "->" + s;
        times[key] += elapsed.count();  // Accumulate time for this key
        start_time = now;
        last_s = s;
    }

    void display() const {
        std::vector<std::pair<std::string, double>> sorted_times(times.begin(), times.end());
        double e2e_time = 0;

        // Calculate total time
        for (const auto& [key, time] : sorted_times) {
            e2e_time += time;
        }

        // Sort by time
        std::sort(sorted_times.begin(), sorted_times.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        // Display results
        std::cout << "Time Breakdown:\n";
        for (const auto& [key, time] : sorted_times) {
            double fraction = e2e_time > 0 ? time / e2e_time : 0;
            std::cout << key << ": " << std::fixed << std::setprecision(3) << fraction << "\n";
        }
    }
};
