#pragma once
#include "NvInfer.h"
#include <string>

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity min_severity = Severity::kWARNING);
    void log(Severity severity, const char* msg) noexcept override;

private:
    Severity min_severity_;
    static std::string severity_str(Severity severity);
};