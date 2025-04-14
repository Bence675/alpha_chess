
#include <iostream>
#include <memory>

#include "logger/logger.h"
#include "trainer/trainer.h"

int main() {
    // Initialize the logger
    auto config = TrainerConfig();
    config.num_simulations = 100;
    config.num_games = 1024;
    config.max_threads = 1024;

    Trainer trainer(config);
    for (int i = 0; i < 100; i++) {
        Logger::log("Iteration " + std::to_string(i));
        memory::clear();
        trainer.self_play();
        trainer.save_dataset("dataset");
        // trainer.load_dataset("dataset");
        trainer.train();
    }
    // trainer.train();


    return 0;
}