
#include "mcts.h"
#include <memory>
#include <vector>
#include <torch/torch.h>
#include <chess/chess.hpp>
#include <logger.h>
#include <board_utils.h>


MCTS::MCTS(std::shared_ptr<torch::nn::Module> model, const config::Config::MCTSConfig& config){
    this->model = model;
    this->num_simulations = config.num_simulations;
    this->c_puct = config.exploration_constant;
}

std::shared_ptr<node_t> MCTS::search(const chess::Board& board)
{
    chess::Board board_copy = chess::Board(board); 
    auto root = std::make_shared<node_t>(board_copy, this->model);

    for (unsigned int i = 0; i < this->num_simulations; ++i) {
        // Logger::log("Simulation " + std::to_string(i));
        root->board = chess::Board(board);
        auto game_result = root->board.isGameOver(); // DRAW, LOSE, NONE
        if(game_result.second != chess::GameResult::NONE) {
            Logger::log("Game Over");
            break;
        }
        simulate(root);
    }
    return root;
}

void MCTS::simulate(std::shared_ptr<node_t> root) {
    auto game_result = root->board.isGameOver(); // DRAW, LOSE, NONE
    if(game_result.second != chess::GameResult::NONE) {
        return; // std::make_shared<HistoryObject>(utils::board_to_tensor(root->board), torch::zeros({1, 8, 8}), 0.0, game_result.second);
    }
    auto node = root->select_best_leaf();
    auto value = node->expand();
    node->backpropagate(value);
    // return std::make_shared<HistoryObject>(utils::board_to_tensor(root->board), torch::zeros({1, 8, 8}), 0.0, chess::GameResult::NONE);
}

void MCTS::set_model(std::shared_ptr<torch::nn::Module> model)
{
    this->model = model;
}

MCTS::~MCTS()
{
}