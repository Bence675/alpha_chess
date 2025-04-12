
#include <iostream>
#include <memory>

#include "logger/logger.h"
#include "trainer/trainer.h"

int main() {
    // Initialize the logger
    auto config = TrainerConfig();
    config.num_simulations = 100;
    config.num_games = 2048;
    config.max_threads = 1024;

    Trainer trainer(config);
    for (int i = 0; i < 10; i++) {
        trainer.self_play();
        trainer.train();
    }
    // trainer.train();


    return 0;
}