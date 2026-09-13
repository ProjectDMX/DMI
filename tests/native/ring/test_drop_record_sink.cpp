// Check fixed timing origin and prove disabled measurements never read a clock.
#include "drop_record_sink.h"
#include <ATen/ATen.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unistd.h>

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static ring::RecordEnvelope record() {
    ring::RecordDescriptor descriptor;
    descriptor.layout = "tensor";
    ring::PayloadSlice slice;
    slice.dtype = static_cast<int32_t>(at::kFloat);
    slice.logical_shape = {2};
    descriptor.rows = {{{std::string("test"), slice}}};
    return {std::move(descriptor), at::tensor({1.f, 2.f}).view(at::kByte)};
}

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("dmi-drop-clock-" + std::to_string(::getpid()));
    try {
        dmx_host::RecordSchema schema;
        dmx_host::RecordLayout layout;
        layout.name = "tensor";
        layout.table = "tensors";
        layout.columns = {{"id", dmx_host::RecordCellType::STRING},
            {"payload", dmx_host::RecordCellType::TENSOR, "dtype", "shape", "bytes"}};
        schema.layouts = {layout};
        for (bool ring : {false, true}) {
            dmx_host::DropRecordSink sink(schema, root.string(), ring ? 1 : 0, false, ring,
                []() -> int64_t { throw std::runtime_error("disabled clock was called"); });
            sink.iteration_start(1);
            sink.submit(record());
            sink.ring_metrics(1, {{"payload_reserved_bytes", 8}});
            sink.iteration_end(1);
            sink.close();
        }
        int calls = 0;
        int64_t tick = 1000;
        dmx_host::DropRecordSink sink(schema, root.string(), 2, true, false,
            [&] { ++calls; return tick; });
        check(calls == 0, "constructor read the clock");
        sink.submit(record());
        check(calls == 0, "setup record initialized the origin");
        sink.iteration_start(11);
        tick = 1200;
        sink.submit(record());
        tick = 1300;
        sink.iteration_end(11);
        tick = 1500;
        sink.iteration_start(12);
        tick = 1600;
        sink.submit(record());
        tick = 1700;
        sink.iteration_end(12);
        sink.close();
        check(calls == 6, "unexpected timing calls");
        std::ifstream file(sink.path());
        const std::string text((std::istreambuf_iterator<char>(file)), {});
        check(text.find("\"iteration\":11,\"time_ns\":0") != std::string::npos, "first start is not zero");
        check(text.find("\"iteration\":12,\"time_ns\":500") != std::string::npos, "origin reset");
        check(text.find("\"cpu_arrival_ns\":600") != std::string::npos, "arrival uses wrong origin");
        check(text.find("\"time_ns\":700,\"duration_ns\":200") != std::string::npos, "iteration duration wrong");
        std::filesystem::remove_all(root);
        std::cout << "drop sink clock checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
