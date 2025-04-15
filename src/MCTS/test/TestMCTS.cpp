


#include <gtest/gtest.h>
#include <torch/torch.h>

#include "chess/chess.hpp"
#include <logger.h>
#include "board_utils.h"
#include "node.h"
#include "KotHModel.h"
#include <mcts.h>
#include "string_utils.h"

using namespace config;

TEST(TestSimulate, TestSimulateStart) {
    Config::MCTSConfig mcts_config;
    mcts_config.num_simulations = 1;
    mcts_config.exploration_constant = 1.0;
    auto mcts = MCTS(std::make_shared<KotHModel>(), mcts_config);
    chess::Board board;
    chess::Board board_copy = chess::Board(board); 
    auto root = std::make_shared<node_t>(board_copy, std::make_shared<KotHModel>());
    mcts.simulate(root);

    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 1);
    
    mcts.simulate(root);

    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 2);
    ASSERT_EQ(root->children[17]->visit_count, 1);
    ASSERT_EQ(root->children[17]->value, 0);
    ASSERT_EQ(root->children[17]->children.size(), 20);

}

TEST(TestSimulate, TestSimulateGameOver) {
    
    Config::MCTSConfig mcts_config;
    mcts_config.num_simulations = 1;
    mcts_config.exploration_constant = 1.0;
    auto mcts = MCTS(std::make_shared<KotHModel>(), mcts_config);
    chess::Board board("r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    chess::Board board_copy = chess::Board(board); 
    auto root = std::make_shared<node_t>(board_copy, std::make_shared<KotHModel>());
    mcts.simulate(root);

    // ASSERT_EQ(history->result, chess::GameResult::WIN);
    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 0);

    board = chess::Board("r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    board_copy = chess::Board(board); 
    root = std::make_shared<node_t>(board_copy, std::make_shared<KotHModel>());
    mcts.simulate(root);

    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 0);
}

TEST(TestSearch, TestSearch) {
    Config::MCTSConfig mcts_config;
    mcts_config.num_simulations = 1;
    mcts_config.exploration_constant = 1.0;
    auto mcts = MCTS(std::make_shared<KotHModel>(), mcts_config);
    chess::Board board;
    auto root = mcts.search(board);
    
    Logger::log("Search result");
    Logger::log(to_string(*root));
}