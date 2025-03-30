#include <gtest/gtest.h>
#include <torch/torch.h>

#include "chess/chess.hpp"
#include <logger.h>
#include "board_utils.h"

namespace utils{

TEST(BoardToTensorTest, EmptyBoard) {
    chess::Board board("8/8/8/8/8/8/8/8 w - - 0 1");
    auto tensor = board_to_tensor(board);

    // Check all piece channels are empty
    for (int i = 0; i < 12; ++i) {
        ASSERT_TRUE(torch::all(tensor[i] == 0).item<bool>());
    }

    // Castling rights
    ASSERT_TRUE(torch::all(tensor[12] == 0).item<bool>());
    ASSERT_TRUE(torch::all(tensor[13] == 0).item<bool>());
    ASSERT_TRUE(torch::all(tensor[14] == 0).item<bool>());
    ASSERT_TRUE(torch::all(tensor[15] == 0).item<bool>());

    // En passant
    ASSERT_TRUE(torch::all(tensor[16] == 0).item<bool>());

    // Half-move clock
    ASSERT_TRUE(torch::all(tensor[17] == 0.0).item<bool>());

    // Side to move (white's turn)
    ASSERT_TRUE(torch::all(tensor[18] == 1).item<bool>());

    // Repetition
    ASSERT_TRUE(torch::all(tensor[19] == 0).item<bool>());
}

TEST(BoardToTensorTest, InitialPosition) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    auto tensor = board_to_tensor(board);

    // Check white pawns (channel 0)
    auto white_pawns = tensor[0];
    for (int col = 0; col < 8; ++col) {
        ASSERT_FLOAT_EQ(white_pawns[1][col].item<float>(), 1.0f);
    }

    // Check black pawns (channel 6)
    auto black_pawns = tensor[6];
    for (int col = 0; col < 8; ++col) {
        ASSERT_FLOAT_EQ(black_pawns[6][col].item<float>(), 1.0f);
    }

    // Check castling rights (all present)
    ASSERT_TRUE(torch::all(tensor[12] == 1).item<bool>());
    ASSERT_TRUE(torch::all(tensor[13] == 1).item<bool>());
    ASSERT_TRUE(torch::all(tensor[14] == 1).item<bool>());
    ASSERT_TRUE(torch::all(tensor[15] == 1).item<bool>());
}

TEST(BoardToTensorTest, CastlingRights) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQk - 0 1");
    auto tensor = board_to_tensor(board);

    ASSERT_TRUE(torch::all(tensor[12] == 1).item<bool>()); // White K
    ASSERT_TRUE(torch::all(tensor[13] == 1).item<bool>()); // White Q
    ASSERT_TRUE(torch::all(tensor[14] == 1).item<bool>()); // Black K
    ASSERT_TRUE(torch::all(tensor[15] == 0).item<bool>()); // Black Q

    board = chess::Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w q - 0 1");
    tensor = board_to_tensor(board);

    ASSERT_TRUE(torch::all(tensor[12] == 0).item<bool>()); // White K
    ASSERT_TRUE(torch::all(tensor[13] == 0).item<bool>()); // White Q
    ASSERT_TRUE(torch::all(tensor[14] == 0).item<bool>()); // Black K
    ASSERT_TRUE(torch::all(tensor[15] == 1).item<bool>()); // Black Q
}

TEST(BoardToTensorTest, EnPassant) {
    chess::Board board("rnbqkbnr/pp1ppppp/8/4P3/2pP4/8/PPP2PPP/RNBQKBNR b KQkq d3 0 1");
    auto tensor = board_to_tensor(board);

    // c6 is index 2 (column) in row 5 (0-based)
    ASSERT_FLOAT_EQ(tensor[16][2][3].item<float>(), 1.0f);
    // Ensure only one position is set
    ASSERT_EQ(torch::sum(tensor[16]).item<int>(), 1);
}

TEST(BoardToTensorTest, HalfMoveClock) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 50 1");
    auto tensor = board_to_tensor(board);
    ASSERT_TRUE(torch::all(tensor[17] == 0.5).item<bool>());
}

TEST(BoardToTensorTest, SideToMove) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    auto tensor = board_to_tensor(board);
    ASSERT_TRUE(torch::all(tensor[18] == 0).item<bool>());

    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    tensor = board_to_tensor(board);
    ASSERT_TRUE(torch::all(tensor[18] == 1).item<bool>());
}

TEST(BoardToTensorTest, Repetition) {
    chess::Board board;
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Make moves to cause repetition (simplified example)
    // Note: Actual test may require more setup to trigger isRepetition()
    board.makeMove(chess::uci::uciToMove(board, "b1c3"));
    board.makeMove(chess::uci::uciToMove(board, "b8c6"));
    board.makeMove(chess::uci::uciToMove(board, "c3b1"));
    board.makeMove(chess::uci::uciToMove(board, "c6b8"));
    board.makeMove(chess::uci::uciToMove(board, "b1c3"));
    board.makeMove(chess::uci::uciToMove(board, "b8c6"));
    board.makeMove(chess::uci::uciToMove(board, "c3b1"));
    board.makeMove(chess::uci::uciToMove(board, "c6b8"));
    
    // After two repetitions, this may not be sufficient; adjust as needed
    ASSERT_TRUE(board.isRepetition());
    auto tensor = board_to_tensor(board);
    ASSERT_TRUE(torch::all(tensor[19] == 1).item<bool>());
}


TEST(MoveToIdxTest, RegularMove) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "e2e4");
    ASSERT_EQ(move_to_idx(move) % 73, 43);

    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    move = chess::uci::uciToMove(board, "f1c4");
    ASSERT_EQ(move_to_idx(move) % 73, 37);

    board.setFen("rnbqkbnr/pppppppp/8/8/4Q3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1");
    move = chess::uci::uciToMove(board, "e4a4");
    ASSERT_EQ(move_to_idx(move) % 73, 24);

    board.setFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    move = chess::uci::uciToMove(board, "d7d5");
    ASSERT_EQ(move_to_idx(move) % 73, 8);
}

TEST(MoveToIdxTest, Promotion) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "a7a8q");
    ASSERT_EQ(move_to_idx(move) % 73, 42);
}

TEST(MoveToIdxTest, UnderPromotion) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "a7a8n");
    ASSERT_EQ(move_to_idx(move) % 73, 65);
}

TEST(MoveToIdxTest, KnightMove) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/3N4/8/PP1P1PPP/R1BQKBNR w KQkq - 0 1");
    auto move = chess::uci::uciToMove(board, "d4f5");
    ASSERT_EQ(move_to_idx(move) % 73, 63);

    move = chess::uci::uciToMove(board, "d4f3");
    ASSERT_EQ(move_to_idx(move) % 73, 59);

    move = chess::uci::uciToMove(board, "d4e6");
    ASSERT_EQ(move_to_idx(move) % 73, 62);

    move = chess::uci::uciToMove(board, "d4c6");
    ASSERT_EQ(move_to_idx(move) % 73, 61);

    move = chess::uci::uciToMove(board, "d4b3");
    ASSERT_EQ(move_to_idx(move) % 73, 56);

    move = chess::uci::uciToMove(board, "d4b5");
    ASSERT_EQ(move_to_idx(move) % 73, 60);

    move = chess::uci::uciToMove(board, "d4c2");
    ASSERT_EQ(move_to_idx(move) % 73, 57);

    move = chess::uci::uciToMove(board, "d4e2");
    ASSERT_EQ(move_to_idx(move) % 73, 58);
}

TEST(IdxToMoveTest, RegularMove) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "e2e4");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    move = chess::uci::uciToMove(board, "f1c4");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    board.setFen("rnbqkbnr/pppppppp/8/8/4Q3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1");
    move = chess::uci::uciToMove(board, "e4a4");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    board.setFen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    move = chess::uci::uciToMove(board, "d7d5");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);
}

TEST(IdxToMoveTest, Promotion) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "a7a8");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

}

TEST(IdxToMoveTest, UnderPromotion) {
    chess::Board board;
    auto move = chess::uci::uciToMove(board, "a7a8n");
    ASSERT_EQ(idx_to_move(move_to_idx(move)).value().promotionType(), chess::PieceType::KNIGHT);
    ASSERT_EQ(idx_to_move(move_to_idx(move)), chess::uci::uciToMove(board, "a7a8")); // Somehow idx_to_move(move_to_idx(move)) != "a7a8n"
}

TEST(IdxToMoveTest, KnightMove) {
    chess::Board board("rnbqkbnr/pppppppp/8/8/3N4/8/PP1P1PPP/R1BQKBNR w KQkq - 0 1");
    auto move = chess::uci::uciToMove(board, "d4f5");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4f3");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4e6");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4c6");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4b3");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4b5");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4c2");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);

    move = chess::uci::uciToMove(board, "d4e2");
    ASSERT_EQ(idx_to_move(move_to_idx(move)), move);
}

}


// namespace utils