
#include <iostream>
#include <memory>
#include "logger.h"

void Logger::log(std::string message) {
    std::cout << message << std::endl;
}

void Logger::log(int message) {
    std::cout << message << std::endl;
}