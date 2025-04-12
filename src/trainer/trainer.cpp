

#include "trainer.h"
#include "node.h"
#include "mcts.h"
#include "model.h"
#include "string_utils.h"
#include "thread_pool.h"


Trainer::Trainer(TrainerConfig config) : config(config) {
    Logger::log("Trainer constructor");
    _model = std::make_shared<ConvModel>(73 * 64);
    _model->to(torch::kCUDA);
    _model->eval();
    _optimizer = std::make_shared<torch::optim::Adam>(_model->parameters(), torch::optim::AdamOptions(0.001));
    Logger::log("Model created");
    _mcts = std::make_shared<MCTS>(_model, config.num_simulations, 1.0, 0.03);
    

    model_thread = std::thread([this]() {
        while (true) {
            torch::Tensor input_tensor;
            
            {
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                memory::getInstance().processing = memory::getInstance().boards_to_compute;
                input_tensor = torch::zeros({static_cast<long>(memory::getInstance().boards_to_compute.size()), 19, 8, 8});
                for (int i = 0; i < memory::getInstance().boards_to_compute.size(); i++) {
                    auto board = chess::Board(memory::getInstance().boards_to_compute[i]);
                    input_tensor[i] = utils::board_to_tensor(board);
                }
                memory::getInstance().boards_to_compute.clear();
            }
            if (input_tensor.sizes()[0] == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                Logger::log("No boards to compute");
                continue;
            }
            Logger::log("Computing action probabilities for " + to_string(input_tensor.sizes()) + " boards");
            auto output = _model->forward(input_tensor);
            auto policy_tensor = std::get<0>(output).to(torch::kCPU);
            auto value_tensor = std::get<1>(output).to(torch::kCPU);

            Logger::log("Action probabilities computed");

            for (int i = 0; i < memory::getInstance().processing.size(); i++) {
                auto fen_without_fullmove = memory::getInstance().processing[i];
                auto board = chess::Board(fen_without_fullmove + " 0");
                auto action_probs = std::vector<std::pair<chess::Move, float>>();
                auto legal_moves = chess::Movelist();
                chess::movegen::legalmoves(legal_moves, board);
                for (auto move : legal_moves) {
                    auto idx = utils::move_to_idx(move);
                    Logger::log(join_str(" ", "Move", std::string(move.from()), std::string(move.to()), "idx", idx));
                    action_probs.push_back(std::make_pair(move, policy_tensor[i][idx].item<float>()));
                }
                
                
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                
                memory::getInstance().action_probs_map[fen_without_fullmove] = std::make_pair(action_probs, value_tensor[i].item<float>());
            }

            Logger::log("Action probabilities stored");

            /*{
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                Logger::log("Action probs map size: " + std::to_string(memory::getInstance().action_probs_map.size()));
            }*/

            {
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                memory::getInstance().processing.clear();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    model_thread.detach();
}

void Trainer::self_play() {
    ThreadPool pool(config.max_threads);

    torch::Tensor data_memory;
    torch::Tensor target_memory;
    for (int i = 0; i < config.num_games; i++) {
        Logger::log("Game " + std::to_string(i));
        pool.enqueue([this]() {
            play_game();
        });
    }

    return pool.wait();
}

void Trainer::play_game() {
    chess::Board board;
    std::vector<ChessData> history;
    while (true) {
        Logger::log("Cache size: " + std::to_string(memory::getInstance().action_probs_map.size()));
        Logger::log("Current Board: " + board.getFen());
        auto root = _mcts->search(board);
        // Logger::log("Search");
        auto action = root->get_action();
        // Logger::log("Action: " + to_string(action));
        
        history.push_back(ChessData{board.getFen(), root->get_action_probs_tensor(), torch::zeros({1})});
        board.makeMove(action);
        if (board.isGameOver().second != chess::GameResult::NONE) {
            torch::Tensor value_tensor;
            torch::Tensor policy_tensor;
            torch::Tensor target_tensor;
            auto game_result = board.isGameOver();
            int result = 0;
            if (game_result.second == chess::GameResult::WIN) {
                result = 1;
            } else if (game_result.second == chess::GameResult::DRAW) {
                result = 0;
            } else {
                result = -1;
            }

            for (auto& item : history) {
                item.value += result;

                result = -result;
            }
            
            break;
        }
    }

    _dataset.add_data(history);
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

void Trainer::load_model(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    _model->load(archive);

    _mcts->set_model(_model);
}

void Trainer::save_model(const std::string& path) {
    torch::serialize::OutputArchive archive;
    _model->save(archive);
    archive.save_to(path);
    Logger::log("Model saved to " + path);
}

void Trainer::set_model(std::shared_ptr<ConvModel> model) {
    _model = model;
    _mcts->set_model(model);
}

void Trainer::train() {
    _model->train();
    if (!_dataset.size().has_value()) {
        Logger::log("Dataset has no data");
        return;
    }
    int size = _dataset.size().value();
    auto indices = torch::randperm(size);
    int batch_size = 64;

    for (int epoch = 0; epoch < config.num_epochs; epoch++) {
        for (int i = 0; i < size; i += batch_size) {
            auto batch_indices = indices.slice(0, i, std::min(i + batch_size, size));
            auto batch = _dataset.get_batch(batch_indices);
            auto input_tensor = std::get<0>(batch);
            auto target_tensor = std::get<1>(batch);
            auto value_tensor = std::get<2>(batch);
            
            Logger::log("Computing action probabilities for " + to_string(input_tensor.sizes()) + " boards  2");
            auto output = _model->forward(input_tensor);
            auto policy_tensor = std::get<0>(output);
            auto value_tensor_out = std::get<1>(output);
            auto policy_loss = torch::nn::functional::cross_entropy(policy_tensor, target_tensor);
            auto value_loss = torch::nn::functional::mse_loss(value_tensor_out, value_tensor);
            auto loss = policy_loss + value_loss;
            _model->zero_grad();
            loss.backward();
            _optimizer->step();
            Logger::log("Epoch: " + std::to_string(epoch) + " Batch: " + std::to_string(i / batch_size) + " Loss: " + std::to_string(loss.item<float>()));
            Logger::log("Policy Loss: " + std::to_string(policy_loss.item<float>()));
            Logger::log("Value Loss: " + std::to_string(value_loss.item<float>()));
        }
        save_model("model_epoch_" + std::to_string(epoch) + ".pt");
        Logger::log("Model saved");
    }
}

