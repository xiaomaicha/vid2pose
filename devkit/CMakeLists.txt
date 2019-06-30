cmake_minimum_required(VERSION 2.8.3)
project(eval_odom)
add_compile_options(-std=c++11)

add_executable(evaluate_odometry cpp/evaluate_odometry.cpp cpp/matrix.cpp)