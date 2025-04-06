
#include <gtest/gtest.h>
#include "string_utils.h"

TEST(JoinStrTest, JoinStrTest) {
    auto x = join_str(", ", 1, 2, 3);
    ASSERT_EQ(x, "1, 2, 3");
    ASSERT_EQ(join_str("", 1, 2, 3), "123");
    ASSERT_EQ(join_str(" ", 1, "a", 3.14), "1 a 3.140000");
    ASSERT_EQ(join_str("", std::vector<int>{1, 2, 3}), "[1, 2, 3]");
}

TEST(SplitStrTest, SplitStrTest) {
    auto x = split("1,2,3", ",");
    ASSERT_EQ(x.size(), 3);
    ASSERT_EQ(x[0], "1");
    ASSERT_EQ(x[1], "2");
    ASSERT_EQ(x[2], "3");
    x = split("1,2,3", " ");
    auto ref = std::vector<std::string>{"1,2,3"};
    ASSERT_EQ(x, ref);
    x = split(",1,2,3", ",");
    ref = std::vector<std::string>{"", "1", "2", "3"};
    ASSERT_EQ(x, ref);
    x = split("1,2,3,", ",");
    ref = std::vector<std::string>{"1", "2", "3", ""};
    ASSERT_EQ(x, ref);

}