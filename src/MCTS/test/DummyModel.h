

#include "model.h"
#include "logger.h"

class DummyModel : public Model {
public:
    DummyModel() {}

    int tensor_hash(torch::Tensor x) {
        int hash = 0;
        for (int i = 0; i < x.size(0); i++) {
            for (int j = 0; j < x.size(1); j++) {
                for (int k = 0; k < x.size(2); k++) {
                    hash += x[i][j][k].item<int>() * (i + j + k);
                }
            }
        }
        return hash;
    }
            

    std::tuple<torch::Tensor, torch::Tensor> _forward(torch::Tensor x) override {
        if (x.sizes().size() != 4) {
            throw std::runtime_error("Input tensor must have 4 dimensions");
        }
        // Logger::log("Forward pass dummy model");
        int action_space = 73 * 64;
        auto policy = torch::zeros({1, action_space});
        auto hashed = tensor_hash(x);
        for (int i = 0; i < action_space; i++) {
            policy[0][i] = hashed % (69 + i);
        }
        auto value = torch::zeros({1, 1});
        value[0][0] = tensor_hash(x) % 420;
        return std::make_tuple(policy, value);
    }

};