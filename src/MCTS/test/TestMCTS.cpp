


#include <gtest/gtest.h>
#include <torch/torch.h>

#include "chess/chess.hpp"
#include <logger.h>
#include "board_utils.h"
#include "node.h"
#include "KotHModel.h"
#include <mcts.h>
#include "string_utils.h"

TEST(TestSimulate, TestSimulateStart) {
    auto mcts = MCTS(std::make_shared<KotHModel>(), 1, 1.0, 0.03);
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

    Logger::log(root->value);
    Logger::log(root->visit_count);
    for (const auto& child : root->children) {
        Logger::log(child->value);
        Logger::log(child->visit_count);
    }
}

TEST(TestSimulate, TestSimulateGameOver) {
    
    auto mcts = MCTS(std::make_shared<KotHModel>(), 1, 1.0, 0.03);
    chess::Board board("r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    chess::Board board_copy = chess::Board(board); 
    auto root = std::make_shared<node_t>(board_copy, std::make_shared<KotHModel>());
    mcts.simulate(root);

    // ASSERT_EQ(history->result, chess::GameResult::WIN);
    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 0);

    mcts = MCTS(std::make_shared<KotHModel>(), 1, 1.0, 0.03);
    board = chess::Board("r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    board_copy = chess::Board(board); 
    root = std::make_shared<node_t>(board_copy, std::make_shared<KotHModel>());
    mcts.simulate(root);

    ASSERT_EQ(root->value, 0);
    ASSERT_EQ(root->visit_count, 0);
}

TEST(TestSearch, TestSearch) {
    auto mcts = MCTS(std::make_shared<KotHModel>(), 100, 1.0, 0.03);
    chess::Board board;
    auto root = mcts.search(board);
    
    Logger::log("Search result");
    Logger::log(to_string(*root));
}