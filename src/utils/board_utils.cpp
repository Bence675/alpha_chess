

#include "board_utils.h"
#include <cmath>
#include "logger.h"
#include "node.h"
#include "string_utils.h"

namespace utils{

torch::Tensor board_to_tensor(chess::Board& board) {
    torch::Tensor tensor = torch::zeros({19, 8, 8});

    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        auto piece = board.at(i);
        if (piece != chess::Piece::NONE) {
            tensor[piece][row][col] = 1;
        }
    }

    auto castling_rights = board.castlingRights();

    tensor[12] = castling_rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE);
    tensor[13] = castling_rights.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    tensor[14] = castling_rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE);
    tensor[15] = castling_rights.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);

    auto en_passant = board.enpassantSq();
    if (en_passant != chess::Square::NO_SQ) {
        unsigned int index = en_passant.index();
        tensor[16][index / 8][index % 8] = 1;
    }

    tensor[17] = board.sideToMove() == chess::Color::WHITE;
    tensor[18] = board.halfMoveClock();

    return tensor;
}

int move_to_idx(chess::Move move) {
    auto from = move.from();
    auto to = move.to();

    int move_idx = 0;

    // underpromotions
    if (move.typeOf() == chess::Move::PROMOTION && 
        static_cast<int>(move.promotionType()) != static_cast<int>(chess::PieceType::QUEEN)) {

        move_idx = 64 + 3 * (static_cast<int>(move.promotionType()) - 1) - from.file() + to.file() + 1;
    } else if (std::abs(from.file() - to.file()) == 1 && std::abs(from.rank() - to.rank()) == 2 ||  // knight moves
                std::abs(from.file() - to.file()) == 2 && std::abs(from.rank() - to.rank()) == 1) {

        std::vector<std::pair<int, int>> directions = {
            {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
            {1, -2}, {2, -1}, {2, 1}, {1, 2}
        };

        for (int i = 0; i < directions.size(); i++) {
            if (directions[i] == std::make_pair(to.rank() - from.rank(), to.file() - from.file())) {
                move_idx = 56 + i;
                break;
            }
        }
    } else {  // all other moves
        std::vector<std::pair<int, int>> directions = {
            {-1, -1}, {-1, 0}, {-1, 1},
            {0, -1}, {0, 1},
            {1, -1}, {1, 0}, {1, 1}
        };
        std::pair<int, int> direction = {to.rank() - from.rank(), to.file() - from.file()};
        if (direction.first == 0) {
            direction.first = 0;
        } else {
            direction.first /= std::abs(direction.first);
        }
        if (direction.second == 0) {
            direction.second = 0;
        } else {
            direction.second /= std::abs(direction.second);
        }

        for (int i = 0; i < directions.size(); i++) {
            if (directions[i] == direction) {
                move_idx = 7 * i + std::max(std::abs(from.rank() - to.rank()), std::abs(from.file() - to.file())) - 1;
                break;
            }
        }
    }
    return move_idx + 73 * from.index();

}

std::optional<chess::Move> idx_to_move(int idx) {
    int move_idx = idx % 73;
    int from_idx = idx / 73;

    int from_file = from_idx % 8;
    int from_rank = from_idx / 8;

    int to_file, to_rank;
    if (move_idx < 56) {
        std::vector<std::pair<int, int>> directions = {
            {-1, -1}, {-1, 0}, {-1, 1},
            {0, -1}, {0, 1},
            {1, -1}, {1, 0}, {1, 1}
        };
        int direction_idx = move_idx / 7;
        int distance = move_idx % 7 + 1;
        to_rank = from_rank + directions[direction_idx].first * distance;
        to_file = from_file + directions[direction_idx].second * distance;
    } else if (move_idx < 64) {
        std::vector<std::pair<int, int>> directions = {
            {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
            {1, -2}, {2, -1}, {2, 1}, {1, 2}
        };

        to_rank = from_rank + directions[move_idx - 56].first;
        to_file = from_file + directions[move_idx - 56].second;
    } else {
        int promotion_type = (move_idx - 64) / 3;
        to_rank = 7;
        to_file = from_file + (move_idx - 64) % 3 - 1;

        std::vector<chess::PieceType> promotion_types = {chess::PieceType::KNIGHT, chess::PieceType::BISHOP, chess::PieceType::ROOK};

        return chess::Move::make(
            chess::Square(chess::File(from_file), chess::Rank(from_rank)), 
            chess::Square(chess::File(to_file), chess::Rank(to_rank)),
            promotion_types[promotion_type]
        );
    }

    if (to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7) {
        return std::nullopt;
    }

    return chess::Move::make(
        chess::Square(chess::File(from_file), chess::Rank(from_rank)), 
        chess::Square(chess::File(to_file), chess::Rank(to_rank))
    );
}   

} // namespace utils