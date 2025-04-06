

#include <torch/torch.h>
#include <vector>
#include "logger.h"
#include <c10/cuda/CUDAStream.h>
#include <iostream>

#ifndef MODEL_H
#define MODEL_H

class conv_block_t : torch::nn::Module
{
public:
    conv_block_t(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding); 
    torch::Tensor forward(torch::Tensor x);
    void to(torch::Device device) { 
        conv1->to(device);
        bn1->to(device);
        relu1->to(device);
        conv2->to(device);
        bn2->to(device);
        relu2->to(device);
    }
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
    void to(torch::Device device) {
        conv->to(device);
        bn->to(device);
        relu->to(device);
        fc->to(device);
    }
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
    void to(torch::Device device) {
        conv->to(device);
        bn->to(device);
        relu->to(device);
        fc1->to(device);
        relu1->to(device);
        fc2->to(device);
        tanh->to(device);
    }

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


class ConvModel : public Model {
public:
    ConvModel(int num_actions);
    void to(torch::Device device) {
        conv_block.to(device);
        policy_head.to(device);
        value_head.to(device);
    }
    
private:
    virtual std::tuple<torch::Tensor, torch::Tensor> _forward(torch::Tensor x);
    conv_block_t conv_block;
    policy_head_t policy_head;
    value_head_t value_head;

};
#endif // MODEL_H