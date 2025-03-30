
#include <gtest/gtest.h>
#include "string_utils.h"

TEST(JoinStrTest, JoinStrTest) {
    auto x = join_str(", ", 1, 2, 3);
    ASSERT_EQ(x, "1, 2, 3");
    ASSERT_EQ(join_str("", 1, 2, 3), "123");
    ASSERT_EQ(join_str(" ", 1, "a", 3.14), "1 a 3.140000");
    ASSERT_EQ(join_str("", std::vector<int>{1, 2, 3}), "[1, 2, 3]");
}