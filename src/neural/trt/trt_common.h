#pragma once

#ifdef USE_TENSORRT

#include <memory>
#include <stdexcept>

#include "NvInfer.h"
#include "utils/format.h"
#include "utils/log.h"

namespace trt {

struct InferDeleter {
    template <typename T> void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T> using InferPtr = std::unique_ptr<T, InferDeleter>;

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity severity = Severity::kERROR) : reportable_severity_(severity) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportable_severity_) {
            return;
        }

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                LOGGING << "[F] " << msg << std::endl;
                break;
            case Severity::kERROR:
                LOGGING << "[E] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                LOGGING << "[W] " << msg << std::endl;
                break;
            case Severity::kINFO:
                LOGGING << "[I] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                LOGGING << "[V] " << msg << std::endl;
                break;
            default:
                LOGGING << "[?] " << msg << std::endl;
                break;
        }
    }

    nvinfer1::ILogger& Get() noexcept {
        return *this;
    }

    void SetReportableSeverity(Severity severity) noexcept {
        reportable_severity_ = severity;
    }

private:
    Severity reportable_severity_;
};

inline void Assert(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        LOGGING << Format("Assertion failure %s(%d): %s\n", file, line, expression);
        throw std::runtime_error("TensorRT error");
    }
}

} // namespace trt

#define TRT_ASSERT(condition) trt::Assert((condition), #condition, __FILE__, __LINE__)

#endif
