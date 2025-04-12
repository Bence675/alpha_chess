
#include <vector>
#include <utility>
#include <random>

namespace utils{

int random_choose(std::vector<float>& probs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probs.begin(), probs.end());
    return d(gen);
}

}