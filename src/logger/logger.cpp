
#include <iostream>
#include <memory>
#include <chrono>
#include "logger.h"
#include "string_utils.h"

void Logger::log(std::string message) {
    std::cout << std::chrono::system_clock::now() << ": " << message << std::endl;
}
