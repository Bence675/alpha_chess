
#include <torch/torch.h>
#include <memory>
#include <vector>

#include "chess/chess.hpp"


#ifndef BOARD_UTILS_H
#define BOARD_UTILS_H

namespace utils{

torch::Tensor board_to_tensor(chess::Board& board);
int move_to_idx(chess::Move move);
std::optional<chess::Move> idx_to_move(int idx);

} // namespace utils

#endif // BOARD_UTILS_H