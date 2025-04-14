

#include "trainer.h"
#include "node.h"
#include "mcts.h"
#include "model.h"
#include "string_utils.h"
#include "thread_pool.h"


Trainer::Trainer(TrainerConfig config) : config(config) {
    Logger::log("Trainer constructor");
    _model = std::make_shared<ConvModel>(73 * 64, 19, 128, 512, 3, 16384);
    _model->to(torch::kCUDA);
    _model->eval();
    _optimizer = std::make_shared<torch::optim::Adam>(_model->parameters(), torch::optim::AdamOptions(0.001));
    Logger::log("Model created");
    _mcts = std::make_shared<MCTS>(_model, config.num_simulations, 1.0, 0.03);
    // auto dataset = ChessDataSet(1000000).map(torch::data::transforms::Stack<>());
    // _dataset = ChessDataSet(1000000).map(torch::data::transforms::Stack<>());
    

    model_thread = std::thread([this]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            // if (!_self_playing) {
            //     Logger::log("Waiting forself play");
             //    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            //     continue;
            // }
            torch::Tensor input_tensor;
            
            {
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                if (memory::getInstance().boards_to_compute.size() == 0) {
                    continue;
                }
                memory::getInstance().processing = memory::getInstance().boards_to_compute;
                // Logger::log("Processing " + std::to_string(memory::getInstance().processing.size()) + " boards");
                input_tensor = torch::zeros({static_cast<long>(memory::getInstance().boards_to_compute.size()), 19, 8, 8});
                for (int i = 0; i < memory::getInstance().boards_to_compute.size(); i++) {
                    auto board = chess::Board(memory::getInstance().boards_to_compute[i]);
                    input_tensor[i] = utils::board_to_tensor(board);
                }
                memory::getInstance().boards_to_compute.clear();
            }
            if (input_tensor.sizes()[0] == 0) {
                Logger::log("No boards to compute");
                continue;
            }
            Logger::log("Computing action probabilities for " + std::to_string(input_tensor.sizes()[0]) + " boards");
            auto output = _model->forward(input_tensor);
            auto policy_tensor = std::get<0>(output).to(torch::kCPU);
            auto value_tensor = std::get<1>(output).to(torch::kCPU);

            for (int i = 0; i < memory::getInstance().processing.size(); i++) {
                auto fen_without_fullmove = memory::getInstance().processing[i];
                auto board = chess::Board(fen_without_fullmove + " 0");
                auto action_probs = std::vector<std::pair<chess::Move, float>>();
                auto legal_moves = chess::Movelist();
                chess::movegen::legalmoves(legal_moves, board);
                for (auto move : legal_moves) {
                    auto idx = utils::move_to_idx(move);
                    action_probs.push_back(std::make_pair(move, policy_tensor[i][idx].item<float>()));
                }
                
                
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                
                memory::getInstance().action_probs_map[fen_without_fullmove] = std::make_pair(action_probs, value_tensor[i].item<float>());
            }

            /*{
                std::unique_lock<std::mutex> lock(memory::getInstance().action_probs_map_mutex);
                Logger::log("Action probs map size: " + std::to_string(memory::getInstance().action_probs_map.size()));
            }*/

            {
                std::unique_lock<std::mutex> lock(memory::getInstance().boards_to_compute_and_processing_mutex);
                memory::getInstance().processing.clear();
            }
        }
    });
    model_thread.detach();
}

void Trainer::self_play() {
    _self_playing = true;
    _dataset.clear();
    ThreadPool pool(config.max_threads);

    torch::Tensor data_memory;
    torch::Tensor target_memory;
    for (int i = 0; i < config.num_games; i++) {
        Logger::log("Game " + std::to_string(i));
        pool.enqueue([this]() {
            play_game();
        });
    }

    pool.wait();

    // save dataset
}

void Trainer::play_game() {
    chess::Board board;
    _self_playing = false;
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
    Logger::log("Training");
    _model->train();
    if (!_dataset.size().has_value()) {
        Logger::log("Dataset has no data");
        return;
    }
    int size = _dataset.size().value();
    auto indices = torch::randperm(size);
    int batch_size = 4096;
    auto data_loader = torch::data::make_data_loader(
        std::move(_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(1)
    );

    int sum_policy_loss = 0;
    int sum_value_loss = 0;

    for (int epoch = 0; epoch < config.num_epochs; epoch++) {
        int batch_count = 0;
        for (auto& batch : *data_loader) {
            batch_count++;
            torch::Tensor input, policy_target, value_target;
            input = batch.input.to(torch::kCUDA);
            policy_target = batch.policy.to(torch::kCUDA);
            // Change 0 in policy_target to -100 for invalid moves
            auto policy_target_mask = policy_target != 0;
            policy_target.masked_fill_(policy_target_mask, -100);

            value_target = batch.value.view({-1}).to(torch::kCUDA);
            // Logger::log("Input shape: " + to_string(input.sizes()));
            // Logger::log("Policy target shape: " + to_string(policy_target.sizes()));
            // Logger::log("Value target shape: " + to_string(value_target.sizes()));
            auto output = _model->forward(input);
            auto policy_output = std::get<0>(output);
            auto value_output = std::get<1>(output).view({-1});
            // Logger::log("Policy output shape: " + to_string(policy_output.sizes()));
            // Logger::log("Value output shape: " + to_string(value_output.sizes()));
            auto policy_loss = torch::nn::functional::cross_entropy(policy_output, policy_target);
            auto value_loss = torch::nn::functional::mse_loss(value_output, value_target);
            auto loss = policy_loss + value_loss;
            _model->zero_grad();
            loss.backward();
            _optimizer->step();
            sum_policy_loss += policy_loss.item<float>();
            sum_value_loss += value_loss.item<float>();
            Logger::log("Epoch: " + std::to_string(epoch) + " Batch: " + std::to_string(batch_count) + " Policy Loss: " + std::to_string(policy_loss.item<float>()) + " Value Loss: " + std::to_string(value_loss.item<float>()) + "Sum Policy Loss: " + std::to_string(sum_policy_loss) + " Sum Value Loss: " + std::to_string(sum_value_loss));
            // Logger::log("Policy Loss: " + std::to_string(policy_loss.item<float>()));
            // Logger::log("Value Loss: " + std::to_string(value_loss.item<float>()));
        }
    }
    save_model("model.pt");
    Logger::log("Model saved");
}

void Trainer::save_dataset(const std::string& path) {
    _dataset.save(path);
}

void Trainer::load_dataset(const std::string& path) {
    _dataset.load(path);
}