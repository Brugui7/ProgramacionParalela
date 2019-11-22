cmake_minimum_required(VERSION 3.14)
project(ProgramacionParalela C)

find_package(OpenMP)

set(CMAKE_C_STANDARD 99)

add_executable(ProgramacionParalela OpenMPExercises/firstPart/FirstPart.c OpenMPExercises/firstPart/SecondPart.c)