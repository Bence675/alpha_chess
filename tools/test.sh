#! /usr/bin/bash

# Test

cd ../build/debug
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 ctest  --output-on-failure 

mkdir -p coverage
lcov --directory . --capture --output-file ./coverage/code_coverage.info -rc lcov_branch_coverage=1
lcov --directory . --remove ./coverage/code_coverage.info  "*torch/*" "/usr/*" "*include/*" "*src/gtest*" "*test/*" -o ./coverage/code_coverage.info -rc lcov_branch_coverage=1
genhtml ./coverage/code_coverage.info --output-directory ./coverage  --branch-coverage

echo "Performance test"
cd ../release
gprof ./src/alpha_chess/alpha_chess > gprof.txt