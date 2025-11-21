#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of STL containers

#include "adm_ctrl.h"  // Include your header file

namespace py = pybind11;

// Pybind11 module definition
PYBIND11_MODULE(SLOsServe_C, m) {
    m.doc() = "Pybind11 bindings for AdmCtrlScheduler and associated data structures";

    // Bind the Request struct
    py::class_<Request>(m, "Request")
        .def(py::init<std::string, bool, double, int, int, double, int, int, int, int, int, bool>(),
             py::arg("id"), py::arg("is_new_req"), py::arg("ddl"), py::arg("input_length"), py::arg("n_computed_tokens"),
             py::arg("profit"), py::arg("mem"), py::arg("tpot_idx"), py::arg("prefill_mem") = -1,
             py::arg("prefill_device_id") = -1, py::arg("decode_device_id") = -1, py::arg("prefill_only") = false)
        .def_readwrite("id", &Request::id)
        .def_readwrite("is_new_req", &Request::is_new_req)
        .def_readwrite("ddl", &Request::ddl)
        .def_readwrite("input_length", &Request::input_length)
        .def_readwrite("n_computed_tokens", &Request::n_computed_tokens)
        .def_readwrite("profit", &Request::profit)
        .def_readwrite("mem", &Request::mem)
        .def_readwrite("tpot_idx", &Request::tpot_idx)
        .def_readwrite("prefill_mem", &Request::prefill_mem)
        .def_readwrite("prefill_device_id", &Request::prefill_device_id)
        .def_readwrite("decode_device_id", &Request::decode_device_id)
        .def_readwrite("prefill_only", &Request::prefill_only)
        .def("__repr__", [](const Request& req) {
            return "<Request id=" + req.id +
                   " is_new_req=" + std::to_string(req.is_new_req) +
                   " ddl=" + std::to_string(req.ddl) +
                   " input_length=" + std::to_string(req.input_length) +
                   " n_computed_tokens=" + std::to_string(req.n_computed_tokens) +
                   " profit=" + std::to_string(req.profit) +
                   " mem=" + std::to_string(req.mem) +
                   " tpot_idx=" + std::to_string(req.tpot_idx) +
                   " prefill_mem=" + std::to_string(req.prefill_mem) +
                   " prefill_device_id=" + std::to_string(req.prefill_device_id) +
                   " decode_device_id=" + std::to_string(req.decode_device_id) +
                   " prefill_only=" + std::to_string(req.prefill_only) + ">";
        });

    // Bind the ReqBatch struct
    py::class_<ReqBatch>(m, "ReqBatch")
        .def(py::init<std::string, bool, int>(),
             py::arg("id"), py::arg("is_prefill"), py::arg("n"))
        .def_readwrite("id", &ReqBatch::id)
        .def_readwrite("is_prefill", &ReqBatch::is_prefill)
        .def_readwrite("n", &ReqBatch::n)
        .def("__repr__", [](const ReqBatch& rbs) {
            return "<ReqBatch id=" + rbs.id +
                   " is_prefill=" + std::to_string(rbs.is_prefill) +
                   " n=" + std::to_string(rbs.n) + ">";
        });

    // Bind the Batch struct
    py::class_<Batch>(m, "Batch")
        .def(py::init<>())  // Default constructor
        .def_readwrite("req_batches", &Batch::req_batches)
        .def_readwrite("prefill_bs", &Batch::prefill_bs)
        .def_readwrite("next", &Batch::next)
        .def_readwrite("estimated_time", &Batch::estimated_time)
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

    // Bind the AdmCtrlScheduler class
    py::class_<AdmCtrlScheduler>(m, "AdmCtrlScheduler")
        .def(py::init<>())
        .def(py::init<std::string, bool>(), py::arg("mode"), py::arg("continuous"))
        .def("set_ar_planner", &AdmCtrlScheduler::set_ar_planner,
            py::arg("tpots"), py::arg("hardware_params"), py::arg("fixed_bs"), py::arg("max_bs") = 16384)
        .def("set_sd_planner", &AdmCtrlScheduler::set_sd_planner,
            py::arg("tpots"), py::arg("hardware_params"), py::arg("fixed_bs"), 
            py::arg("alpha"), py::arg("max_sd_size"), py::arg("fixed_spec"), py::arg("max_bs") = 16384)
        .def("schedule", &AdmCtrlScheduler::schedule,
             py::arg("reqs"), py::arg("M"), py::arg("current_time"), py::arg("verbose"))
        .def("__repr__", [](const AdmCtrlScheduler& scheduler) {
            return "<AdmCtrlScheduler>";
        });

    // Bind the AdmCtrlRouter class
    py::class_<AdmCtrlRouter>(m, "AdmCtrlRouter")
        .def(py::init<int, std::vector<double>, double>(),
             py::arg("n_devices"), py::arg("hardware_params"), py::arg("tpot"))
        .def("schedule", &AdmCtrlRouter::schedule,
             py::arg("reqs"), py::arg("Ms"), py::arg("current_time"), py::arg("verbose"))
        .def("__repr__", [](const AdmCtrlRouter& router) {
            return "<AdmCtrlRouter>";
        });
    
    // Bind the RequestOutput struct
    py::class_<RequestOutput>(m, "RequestOutput")
        .def_readwrite("admitted", &RequestOutput::admitted)
        .def_readwrite("prefill_device_id", &RequestOutput::prefill_device_id)
        .def_readwrite("decode_device_id", &RequestOutput::decode_device_id)
        .def("__repr__", [](const RequestOutput& output) {
            return "<RequestOutput admitted=" + std::to_string(output.admitted) +
                   " prefill_device_id=" + std::to_string(output.prefill_device_id) +
                   " decode_device_id=" + std::to_string(output.decode_device_id) + ">";
        });
}
