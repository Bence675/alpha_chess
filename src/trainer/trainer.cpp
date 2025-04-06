

#include "trainer.h"
#include "node.h"
#include "mcts.h"
#include "model.h"
#include "string_utils.h"


Trainer::Trainer(TrainerConfig config) {
    _config = config;
    Logger::log("Trainer constructor");
    auto model = std::make_shared<ConvModel>(73 * 64);
    Logger::log("Model created");
    _mcts = std::make_shared<MCTS>(model, config.num_simulations, 1.0, 0.03);
}

void Trainer::self_play() {
    for (int i = 0; _config.num_games; i++) {
        Logger::log("Game " + std::to_string(i));
        play_game();
    }
}

void Trainer::play_game() {
    chess::Board board;
    while (true) {
        Logger::log("Cache size: " + std::to_string(node_t::action_probs_map.size()));
        Logger::log("Current Board: " + board.getFen());
        auto root = _mcts->search(board);
        // Logger::log("Search");
        auto action = root->get_action();
        // Logger::log("Action: " + to_string(action));
        board.makeMove(action);
        if (board.isGameOver().second != chess::GameResult::NONE) {
            break;
        }
    }
    auto game_result = board.isGameOver();
    if (game_result.second == chess::GameResult::WIN) {
        Logger::log("Game Over: Win");
    } else if (game_result.second == chess::GameResult::DRAW) {
        Logger::log("Game Over: Draw");
    } else {
        Logger::log("Game Over: Lose");
    }
    Logger::log("Game Over");
}