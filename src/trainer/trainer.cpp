

#include "trainer.h"
#include "node.h"
#include "mcts.h"
#include "model.h"


Trainer::Trainer() {
    auto model = std::make_shared<ConvModel>(73 * 64);
    _mcts = std::make_shared<MCTS>(model, 100, 1.0, 0.03);
}

void Trainer::self_play() {
    chess::Board board;
    while (true) {
        Logger::log("Current Board: " + board.getFen());
        auto root = _mcts->search(board);
        auto action = root->get_action();
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