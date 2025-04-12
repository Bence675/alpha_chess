

#include "trainer.h"
#include <gtest/gtest.h>


TEST(TestTrainer, TestPlayGame) {
    TrainerConfig config;
    config.num_simulations = 10;
    Trainer trainer;
    trainer.play_game();
}

TEST(TestTrainer, TestSelfPlay) {
    TrainerConfig config;
    config.num_simulations = 100;

    config.num_games = 100;
    Trainer trainer;
    trainer.self_play();
}