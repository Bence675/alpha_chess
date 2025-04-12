

#include <torch/torch.h>
#include <torch/serialize.h>
#include <vector>
#include "logger.h"
#include <c10/cuda/CUDAStream.h>
#include <iostream>


#ifndef MODEL_H
#define MODEL_H

/*class conv_block_t : public torch::nn::Module
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
};*/

/*
class policy_head_t : public torch::nn::Module
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

*/

/*
class value_head_t : public torch::nn::Module
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
    
};*/

/*
class Model {
public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x){
        _check_input(x);
        auto res = _forward(x);
        _check_output(res);
        return res;
    }

    void load(const std::string& path) {
        this->load(path);
    }
    void save(const std::string& path) {
        this->save(path);
    }

    virtual void train() = 0;
    virtual void eval() = 0;
    virtual void zero_grad() = 0;
    virtual void step() = 0;
protected:
    virtual std::tuple<torch::Tensor, torch::Tensor> _forward(torch::Tensor x) = 0;
    virtual void _check_input(torch::Tensor x){};
    virtual void _check_output(std::tuple<torch::Tensor, torch::Tensor> res){};
};*/


class ConvModel : public torch::nn::Module  {
public:
    ConvModel(int num_actions, int in_channels=19, int hidden_channels=128, int out_channels=512, int kernel_size=3, int hidden_dim=2048);
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

    /*
    conv_block_t conv_block;
    policy_head_t policy_head;
    value_head_t value_head;*/

};
#endif // MODEL_H