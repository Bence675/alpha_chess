
#include <iostream>
#include <memory>
#include <ctime>
#include "logger.h"
#include "string_utils.h"

void Logger::log(std::string message) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::cout << std::put_time(&tm, "%d-%m-%Y %H-%M-%S.%u") << ": " << message << std::endl;
}
