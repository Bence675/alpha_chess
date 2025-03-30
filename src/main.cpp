
#include <iostream>
#include <memory>

#include "logger/logger.h"

int main() {
    auto logger = std::make_shared<Logger>();
    logger->log("Hello, World!");
    return 0;
}