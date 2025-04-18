

#include "trainer.h"
#include <gtest/gtest.h>
#include "config.h"

using namespace config;

TEST(TestTrainer, TestPlayGame) {
    Config config;
    config.mcts_config.num_simulations = 10;
    Trainer trainer(config);
    trainer.play_game(0, 0);
}

TEST(TestTrainer, TestSelfPlay) {
    Config config;
    config.trainer_config.self_play_config.num_games_per_iteration = 100;

    config.mcts_config.num_simulations = 100;
    Trainer trainer(config);
    trainer.self_play(0);
}