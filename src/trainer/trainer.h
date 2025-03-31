
#include <torch/torch.h>
#include <memory>
#include "model.h"
#include "mcts.h"

class Trainer {
public:
    Trainer();

    void train();
    void self_play();
    void load_model(const std::string& path);
    void save_model(const std::string& path);
    void set_model(std::shared_ptr<Model> model);

private:
    std::shared_ptr<Model> _model;
    std::shared_ptr<MCTS> _mcts;
};