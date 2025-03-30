

#include <torch/torch.h>
#include <vector>
#include "logger.h"

#ifndef MODEL_H
#define MODEL_H

class conv_block_t : torch::nn::Module
{
public:
    conv_block_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding); 
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::ReLU relu1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::ReLU relu2;
};

class policy_head_t : torch::nn::Module
{
public:
    policy_head_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t num_actions);

    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
    torch::nn::ReLU relu;
    torch::nn::Linear fc;
};

class value_head_t : torch::nn::Module
{
public:
    value_head_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, int64_t hidden_dim);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
    torch::nn::ReLU relu;
    torch::nn::Linear fc1;
    torch::nn::ReLU relu1;
    torch::nn::Linear fc2;
    torch::nn::Tanh tanh;
    
};

class Model {
public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x){
        _check_input(x);
        auto res = _forward(x);
        _check_output(res);
        return res;
    }
protected:
    virtual std::tuple<torch::Tensor, torch::Tensor> _forward(torch::Tensor x) = 0;
    virtual void _check_input(torch::Tensor x){};
    virtual void _check_output(std::tuple<torch::Tensor, torch::Tensor> res){};
};


class ConvModel : Model
{
public:
    ConvModel(int64_t num_actions);
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    
private:
    conv_block_t conv_block;
    policy_head_t policy_head;
    value_head_t value_head;

};
#endif // MODEL_H