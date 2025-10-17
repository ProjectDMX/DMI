// Slice parsing utilities

#include "native_engine_internal.h"

namespace monitoring {

SliceSpec NativeMonitoringEngine::Impl::parse_slice_py(py::object obj) {
  SliceSpec spec;
  if (obj.is_none()) { spec.mode = SliceMode::Identity; return spec; }
  // int index
  if (py::isinstance<py::int_>(obj)) {
    spec.mode = SliceMode::Int; spec.int_value = obj.cast<int64_t>(); return spec;
  }
  // python slice
  if (py::hasattr(obj, "start") && py::hasattr(obj, "stop") && py::hasattr(obj, "step")) {
    spec.mode = SliceMode::Range;
    py::object start = obj.attr("start");
    py::object stop = obj.attr("stop");
    py::object step = obj.attr("step");
    if (!start.is_none()) spec.start = start.cast<int64_t>();
    if (!stop.is_none()) spec.stop = stop.cast<int64_t>();
    if (!step.is_none()) spec.step = step.cast<int64_t>();
    return spec;
  }
  // list/tuple of indices
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    spec.mode = SliceMode::Array;
    if (py::isinstance<py::list>(obj)) {
      spec.indices = obj.cast<std::vector<int64_t>>();
    } else {
      auto tup = obj.cast<py::tuple>();
      spec.indices.reserve(tup.size());
      for (auto it : tup) spec.indices.push_back(it.cast<int64_t>());
    }
    return spec;
  }
  // transformer_lens Slice-like
  if (py::hasattr(obj, "mode") && py::hasattr(obj, "slice")) {
    std::string mode = obj.attr("mode").cast<std::string>();
    py::object data = obj.attr("slice");
    if (mode == "identity") { spec.mode = SliceMode::Identity; return spec; }
    if (mode == "int") { spec.mode = SliceMode::Int; spec.int_value = data.cast<int64_t>(); return spec; }
    if (mode == "slice") {
      spec.mode = SliceMode::Range;
      if (py::hasattr(data, "start")) {
        py::object start = data.attr("start"); if (!start.is_none()) spec.start = start.cast<int64_t>();
        py::object stop = data.attr("stop"); if (!stop.is_none()) spec.stop = stop.cast<int64_t>();
        py::object step = data.attr("step"); if (!step.is_none()) spec.step = step.cast<int64_t>();
      }
      return spec;
    }
    if (mode == "array") {
      spec.mode = SliceMode::Array;
      if (py::isinstance<py::list>(data)) {
        spec.indices = data.cast<std::vector<int64_t>>();
      } else if (py::isinstance<py::tuple>(data)) {
        auto tup = data.cast<py::tuple>();
        spec.indices.reserve(tup.size());
        for (auto it : tup) spec.indices.push_back(it.cast<int64_t>());
      }
      return spec;
    }
  }
  // Fallback
  spec.mode = SliceMode::Identity;
  return spec;
}

TaskSpec NativeMonitoringEngine::Impl::parse_task_tuple(const py::tuple& task_tuple) {
  TORCH_CHECK(task_tuple.size() == 6,
              "Native backend expects task tuple of length 6, got ",
              task_tuple.size());

  TaskSpec spec;
  spec.tensor = task_tuple[0].cast<at::Tensor>();
  spec.slice_dim = task_tuple[1].cast<int64_t>();
  spec.remove_batch_dim = task_tuple[2].cast<bool>();
  spec.can_slice = task_tuple[3].cast<bool>();

  spec.slice = parse_slice_tuple(task_tuple[4].cast<py::tuple>());

  py::object device_obj = task_tuple[5];
  if (!device_obj.is_none()) {
    spec.target_device = device_obj.cast<c10::Device>();
  }
  return spec;
}

SliceSpec NativeMonitoringEngine::Impl::parse_slice_tuple(const py::tuple& slice_tuple) {
  TORCH_CHECK(slice_tuple.size() > 0, "Slice tuple cannot be empty");
  SliceSpec spec;
  // Fast int-coded modes: 0=identity, 1=int, 2=slice, 3=array
  if (py::isinstance<py::int_>(slice_tuple[0])) {
    int mode_code = slice_tuple[0].cast<int>();
    switch (mode_code) {
      case 0: spec.mode = SliceMode::Identity; return spec;
      case 1: {
        TORCH_CHECK(slice_tuple.size() >= 2, "Slice int mode needs value");
        spec.mode = SliceMode::Int;
        spec.int_value = slice_tuple[1].cast<int64_t>();
        return spec;
      }
      case 2: {
        spec.mode = SliceMode::Range;
        py::object start = slice_tuple.size() > 1 ? slice_tuple[1] : py::none();
        py::object stop = slice_tuple.size() > 2 ? slice_tuple[2] : py::none();
        py::object step = slice_tuple.size() > 3 ? slice_tuple[3] : py::none();
        if (!start.is_none()) spec.start = start.cast<int64_t>();
        if (!stop.is_none()) spec.stop = stop.cast<int64_t>();
        if (!step.is_none()) spec.step = step.cast<int64_t>();
        return spec;
      }
      case 3: {
        spec.mode = SliceMode::Array;
        if (slice_tuple.size() > 1) {
          py::object values_obj = slice_tuple[1];
          if (py::isinstance<py::tuple>(values_obj)) {
            auto values_tuple = values_obj.cast<py::tuple>();
            spec.indices.reserve(values_tuple.size());
            for (auto item : values_tuple) spec.indices.push_back(item.cast<int64_t>());
          } else if (py::isinstance<py::list>(values_obj)) {
            spec.indices = values_obj.cast<std::vector<int64_t>>();
          }
        }
        return spec;
      }
      default: break;
    }
  }
  // String-coded modes
  std::string mode = slice_tuple[0].cast<std::string>();
  if (mode == "identity") { spec.mode = SliceMode::Identity; return spec; }

  if (mode == "int") {
    TORCH_CHECK(slice_tuple.size() >= 2,
                "Slice tuple with mode=int must include value");
    spec.mode = SliceMode::Int;
    spec.int_value = slice_tuple[1].cast<int64_t>();
    return spec;
  }

  if (mode == "slice") {
    spec.mode = SliceMode::Range;
    py::object start = slice_tuple.size() > 1 ? slice_tuple[1] : py::none();
    py::object stop = slice_tuple.size() > 2 ? slice_tuple[2] : py::none();
    py::object step = slice_tuple.size() > 3 ? slice_tuple[3] : py::none();
    if (!start.is_none()) spec.start = start.cast<int64_t>();
    if (!stop.is_none()) spec.stop = stop.cast<int64_t>();
    if (!step.is_none()) spec.step = step.cast<int64_t>();
    return spec;
  }

  if (mode == "array") {
    spec.mode = SliceMode::Array;
    if (slice_tuple.size() > 1) {
      py::object values_obj = slice_tuple[1];
      if (py::isinstance<py::list>(values_obj)) {
        spec.indices = values_obj.cast<std::vector<int64_t>>();
      } else if (py::isinstance<py::tuple>(values_obj)) {
        auto values_tuple = values_obj.cast<py::tuple>();
        spec.indices.reserve(values_tuple.size());
        for (auto item : values_tuple) { spec.indices.push_back(item.cast<int64_t>()); }
      } else if (!values_obj.is_none()) {
        spec.indices.push_back(values_obj.cast<int64_t>());
      }
    }
    return spec;
  }

  spec.mode = SliceMode::Identity;
  return spec;
}

}  // namespace monitoring

