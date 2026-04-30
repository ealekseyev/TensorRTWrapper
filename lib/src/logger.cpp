#include "logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

Logger::Logger(Severity min_severity) : min_severity_(min_severity) {}

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity > min_severity_)
        return;

    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream ts;
    ts << std::put_time(&tm, "%H:%M:%S");

    std::ostream& out = (severity <= Severity::kERROR) ? std::cerr : std::cout;
    out << "[" << ts.str() << "] [TRT " << severity_str(severity) << "] " << msg << "\n";
}

std::string Logger::severity_str(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "FATAL";
        case Severity::kERROR:          return "ERROR";
        case Severity::kWARNING:        return "WARN ";
        case Severity::kINFO:           return "INFO ";
        case Severity::kVERBOSE:        return "DEBUG";
        default:                        return "?????";
    }
}