
#include "random.h"
#include <gtest/gtest.h>

using namespace utils;

TEST(TestRandom, TestRandomChoose) {
    std::vector<float> probs = {0.1, 0.2, 0.3, 0.4};
    int chosen_index = random_choose(probs);
    ASSERT_GE(chosen_index, 0);
    ASSERT_LT(chosen_index, probs.size());
}