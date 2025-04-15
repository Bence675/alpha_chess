

#include <torch/torch.h>
#include <torch/serialize.h>
#include <vector>
#include "logger.h"
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include "config.h"


#ifndef MODEL_H
#define MODEL_H

class ConvModel : public torch::nn::Module  {
public:
    ConvModel(const config::Config::NetworkConfig& config);
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    void to(torch::Device device) {
        _conv_block->to(device);
        _policy_head->to(device);
        _value_head->to(device);
    }

private:
    
    torch::nn::Sequential _conv_block;
    torch::nn::Sequential _policy_head;
    torch::nn::Sequential _value_head;

};
#endif // MODEL_H