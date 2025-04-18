
#include <torch/torch.h>
#include <memory>
#include "model.h"
#include "mcts.h"
#include "memory.h"
#include "dataset.h"

#ifndef TRAINER_H
#define TRAINER_H

class Trainer {
public:
    Trainer(const config::Config& config);

    void train();
    void play_game(int iteration, int game);
    void self_play(int iteration);
    void load_model(const std::string& path);
    void save_model(const std::string& path);
    void set_model(std::shared_ptr<LCZero> model);
    void save_dataset(const std::string& path);
    void load_dataset(const std::string& path);


private:
    std::shared_ptr<LCZero> _model;
    std::shared_ptr<MCTS> _mcts;
    std::thread model_thread;
    config::Config::TrainerConfig config;
    ChessDataSet _dataset;
    std::shared_ptr<torch::optim::Adam> _optimizer;
    bool _self_playing = false;
};

#endif // TRAINER_H