#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of STL containers

#include "promax_spec_sch.h"  // Include your header file

namespace py = pybind11;

// Pybind11 module definition
PYBIND11_MODULE(promax, m) {
    m.doc() = "Pybind11 bindings for PromaxScheduler and associated data structures";

    // Bind the Request struct
    py::class_<Request>(m, "Request")
        .def(py::init<int, bool, double, int, double, int, int>(),
             py::arg("id"), py::arg("is_new_req"), py::arg("ddl"), py::arg("input_length"),
             py::arg("profit"), py::arg("mem"), py::arg("tpot_idx"))
        .def_readwrite("id", &Request::id)
        .def_readwrite("is_new_req", &Request::is_new_req)
        .def_readwrite("ddl", &Request::ddl)
        .def_readwrite("input_length", &Request::input_length)
        .def_readwrite("profit", &Request::profit)
        .def_readwrite("mem", &Request::mem)
        .def_readwrite("tpot_idx", &Request::tpot_idx)
        .def("__repr__", [](const Request& req) {
            return "<Request id=" + std::to_string(req.id) +
                   " is_new_req=" + std::to_string(req.is_new_req) +
                   " ddl=" + std::to_string(req.ddl) +
                   " input_length=" + std::to_string(req.input_length) +
                   " profit=" + std::to_string(req.profit) +
                   " mem=" + std::to_string(req.mem) +
                   " tpot_idx=" + std::to_string(req.tpot_idx) + ">";
        });

    // Bind the ReqBatch struct
    py::class_<ReqBatch>(m, "ReqBatch")
        .def(py::init<int, bool, int>(),
             py::arg("id"), py::arg("is_prefill"), py::arg("n"))
        .def_readwrite("id", &ReqBatch::id)
        .def_readwrite("is_prefill", &ReqBatch::is_prefill)
        .def_readwrite("n", &ReqBatch::n)
        .def("__repr__", [](const ReqBatch& rbs) {
            return "<ReqBatch id=" + std::to_string(rbs.id) +
                   " is_prefill=" + std::to_string(rbs.is_prefill) +
                   " n=" + std::to_string(rbs.n) + ">";
        });

    // Bind the Batch struct
    py::class_<Batch>(m, "Batch")
        .def(py::init<>())  // Default constructor
        .def_readwrite("req_batches", &Batch::req_batches)
        .def_readwrite("prefill_bs", &Batch::prefill_bs)
        .def_readwrite("next", &Batch::next)
        .def("__repr__", [](const Batch& bs) {
            return "<Batch " + std::to_string(bs.req_batches.size()) +
                   "#bs, prefill_bs=" + std::to_string(bs.prefill_bs) +
                   " next=" + std::to_string(bs.next) + ">";
        });

    // Bind the Batch struct
    // py::class_<Batch>(m, "Batch")
    //     .def(py::init<>())  // Default constructor
    //     .def(py::init<int, int, std::vector<int>>(),
    //          py::arg("bs"), py::arg("n_batch"), py::arg("sd_sizes"))
    //     .def_readwrite("bs", &Batch::bs)
    //     .def_readwrite("n_batch", &Batch::n_batch)
    //     .def_readwrite("sd_sizes", &Batch::sd_sizes)
    //     .def("__repr__", [](const Batch& batch) {
    //         return "<Batch bs=" + std::to_string(batch.bs) +
    //                " n_batch=" + std::to_string(batch.n_batch) +
    //                " sd_sizes_size=" + std::to_string(batch.sd_sizes.size()) + ">";
    //     });

    // Bind the PromaxScheduler class
    py::class_<PromaxScheduler>(m, "PromaxScheduler")
        .def(py::init<>())
        .def("set_ar_planner", &PromaxScheduler::set_ar_planner,
            py::arg("tpots"), py::arg("hardware_params"), py::arg("fixed_bs"))
        .def("set_sd_planner", &PromaxScheduler::set_sd_planner,
            py::arg("tpots"), py::arg("hardware_params"), py::arg("fixed_bs"), 
            py::arg("alpha"), py::arg("max_sd_size"), py::arg("fixed_spec"))
        .def("schedule", &PromaxScheduler::schedule,
             py::arg("reqs"), py::arg("M"), py::arg("current_time"), py::arg("verbose"))
        .def("__repr__", [](const PromaxScheduler& scheduler) {
            return "<PromaxScheduler>";
        });
}
