
#include "mcts.h"
#include <memory>
#include <vector>
#include <torch/torch.h>
#include <chess/chess.hpp>
#include <logger.h>
#include <board_utils.h>


MCTS::MCTS(std::shared_ptr<Model> model, unsigned int num_simulations, float c_puct, float epsilon)
{
    this->model = model;
    this->num_simulations = num_simulations;
    this->c_puct = c_puct;
    this->epsilon = epsilon;
}

std::shared_ptr<node_t> MCTS::search(const chess::Board& board)
{
    chess::Board board_copy = chess::Board(board); 
    auto root = std::make_shared<node_t>(board_copy, this->model);

    for (unsigned int i = 0; i < this->num_simulations; ++i) {
        Logger::log("Simulation " + std::to_string(i));
        root->board = chess::Board(board);
        simulate(root);
        auto game_result = root->board.isGameOver(); // DRAW, LOSE, NONE
        if(game_result.second != chess::GameResult::NONE) {
            Logger::log("Game Over");
            break;
        }
    }
    return root;
}

std::shared_ptr<HistoryObject> MCTS::simulate(std::shared_ptr<node_t> root) {
    auto game_result = root->board.isGameOver(); // DRAW, LOSE, NONE
    if(game_result.second != chess::GameResult::NONE) {
        return std::make_shared<HistoryObject>(utils::board_to_tensor(root->board), torch::zeros({1, 8, 8}), 0.0, game_result.second);
    }
    auto node = root->select_best_leaf();
    auto value = node->expand();
    // Logger::log("Value: " + std::to_string(value));
    node->backpropagate(value);
    
    return std::make_shared<HistoryObject>(utils::board_to_tensor(root->board), torch::zeros({1, 8, 8}), 0.0, chess::GameResult::NONE);
}

MCTS::~MCTS()
{
}