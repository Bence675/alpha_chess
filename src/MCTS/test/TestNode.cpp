
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "KotHModel.h"
#include "chess/chess.hpp"
#include <logger.h>
#include "board_utils.h"
#include "node.h"

TEST(TestExtract, TestExtractStart) {
    chess::Board board;
    auto tensor = utils::board_to_tensor(board);
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();

    auto node = std::make_shared<node_t>(board, model_ptr);

    Logger::log("Extracting node");

    ASSERT_EQ(node->expand(), 0);
    ASSERT_EQ(node->children.size(), 20);
    ASSERT_EQ(node->children[0]->prior, 1);
    ASSERT_EQ(node->children[0]->move, chess::Move::make(chess::Square("a2"), chess::Square("a3")));
    ASSERT_EQ(node->children[1]->prior, 1);
    ASSERT_EQ(node->children[1]->move, chess::Move::make(chess::Square("b2"), chess::Square("b3")));
    ASSERT_EQ(node->children[8]->prior, 2);
    ASSERT_EQ(node->children[8]->move, chess::Move::make(chess::Square("a2"), chess::Square("a4")));
}

TEST(TestExtract, TestExtractGameOver) {
    chess::Board board("r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    auto tensor = utils::board_to_tensor(board);
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();

    auto node = std::make_shared<node_t>(board, model_ptr);
    node->expand();

    ASSERT_EQ(node->children.size(), 0);
}

TEST(TestExtract, TestExtracAlreadyExpanded) {
    chess::Board board;
    auto tensor = utils::board_to_tensor(board);
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();

    auto node = std::make_shared<node_t>(board, model_ptr);
    node->expand();

    ASSERT_THROW(node->expand(), std::runtime_error);
}

TEST(TestBackPropagate, TestBackPropagateSingleNode) {
    chess::Board board;
    auto tensor = utils::board_to_tensor(board);
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();

    auto node = std::make_shared<node_t>(board, model_ptr);
    node->expand();
    node->backpropagate(1);

    ASSERT_EQ(node->visit_count, 1);
    ASSERT_EQ(node->value, 1);
}

TEST(TestBackPropagate, TestBackPropagateMultipleNodes) {
    chess::Board board;
    auto tensor = utils::board_to_tensor(board);
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();

    auto node = std::make_shared<node_t>(board, model_ptr);
    node->expand();

    
    ASSERT_EQ(node->value, 0);
    ASSERT_EQ(node->visit_count, 0);
    node->children[0]->backpropagate(1);

    ASSERT_EQ(node->children[0]->visit_count, 1);
    ASSERT_EQ(node->children[0]->value, 1);
    ASSERT_EQ(node->value, -1);
    ASSERT_EQ(node->visit_count, 1);
}

TEST(UcbScoreTest, UnvisitedChild) {
    chess::Board board;
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();
    auto node = std::make_shared<node_t>(board, model_ptr);
    node->visit_count = 1;
    auto child = std::make_shared<node_t>(board, model_ptr, node, 0, 0.0);
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.0);
    
    child = std::make_shared<node_t>(board, model_ptr, node, 0, 1.0);
    ASSERT_FLOAT_EQ(node->ucb_score(child), 1.0);

    child = std::make_shared<node_t>(board, model_ptr, node, 0, 0.5);
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.5);

}

TEST(UcbScoreTest, VisitedChild) {
    chess::Board board;
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();
    auto node = std::make_shared<node_t>(board, model_ptr);
    node->visit_count = 1;
    auto child = std::make_shared<node_t>(board, model_ptr, node, 0, 0.0);
    child->visit_count = 1;
    child->value = 1;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.0);

    child->value = 0;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.5);

    child->value = 0.5;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.25);

    child->value = 0.25;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.375);

    node->visit_count = 16;
    child->visit_count = 1;
    child->value = 0;
    child->prior = 0.5;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 1.5);

    child->visit_count = 3;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 1);

    child->visit_count = 7;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.75);

    child->visit_count = 15;
    ASSERT_FLOAT_EQ(node->ucb_score(child), 0.625);
}

TEST(SelectBestChildTest, TestSelectBestChild) {
    chess::Board board;
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();
    auto node = std::make_shared<node_t>(board, model_ptr);
    node->visit_count = 1;
    auto child1 = std::make_shared<node_t>(board, model_ptr, node, 0, 0.0);
    auto child2 = std::make_shared<node_t>(board, model_ptr, node, 0, 1.0);
    auto child3 = std::make_shared<node_t>(board, model_ptr, node, 0, 0.5);
    node->children.push_back(child1);
    node->children.push_back(child2);
    node->children.push_back(child3);

    ASSERT_EQ(node->select_best_child(), child2);
}

TEST(SelectBestLeafTest, TestSelectBestLeafOneLayer) {
    chess::Board board;
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();
    auto node = std::make_shared<node_t>(board, model_ptr);
    node->visit_count = 1;
    auto child1 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("e2"), chess::Square("e3")), 0.0);
    auto child2 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("e2"), chess::Square("e4")), 1.0);
    auto child3 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("d2"), chess::Square("d4")), 0.5);
    node->children.push_back(child1);
    node->children.push_back(child2);
    node->children.push_back(child3);

    auto leaf = node->select_best_leaf();
    ASSERT_EQ(leaf, child2);
    ASSERT_EQ(leaf->board.getFen(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
}

TEST(SelectBestLeafTest, TestSelectBestLeafTwoLayer) {
    chess::Board board;
    std::shared_ptr<Model> model_ptr = std::make_shared<KotHModel>();
    auto node = std::make_shared<node_t>(board, model_ptr);
    node->visit_count = 1;
    auto child1 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("e2"), chess::Square("e3")), 0.0);
    auto child2 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("e2"), chess::Square("e4")), 1.0);
    auto child3 = std::make_shared<node_t>(board, model_ptr, node, chess::Move::make(chess::Square("d2"), chess::Square("d4")), 0.5);
    node->children.push_back(child1);
    node->children.push_back(child2);
    node->children.push_back(child3);

    child1->visit_count = 1;
    child3->visit_count = 1;
    auto child21 = std::make_shared<node_t>(board, model_ptr, child2, chess::Move::make(chess::Square("e7"), chess::Square("e5")), 0.3);
    auto child22 = std::make_shared<node_t>(board, model_ptr, child2, chess::Move::make(chess::Square("e7"), chess::Square("e6")), 0.3);
    auto child23 = std::make_shared<node_t>(board, model_ptr, child2, chess::Move::make(chess::Square("d7"), chess::Square("d5")), 0.3);
    child2->children.push_back(child21);
    child2->children.push_back(child22);
    child2->children.push_back(child23);

    node->visit_count = 15;
    child1->visit_count = 5;
    child2->visit_count = 5;
    child3->visit_count = 5;
    child21->visit_count = 2;
    child22->visit_count = 1;
    child23->visit_count = 2;

    auto leaf = node->select_best_leaf();
    Logger::log(leaf->board.getFen());
    ASSERT_EQ(leaf, child22);
    ASSERT_EQ(leaf->board.getFen(), "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
}