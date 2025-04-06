
#include <iostream>
#include <memory>

#include "logger/logger.h"
#include "trainer/trainer.h"

int main() {
    // Initialize the logger
    auto config = TrainerConfig();
    config.num_simulations = 100;
    config.num_games = 100;

    Trainer trainer(config);
    trainer.self_play();
    // trainer.train();


    return 0;
}