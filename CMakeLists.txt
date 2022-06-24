cmake_minimum_required(VERSION 3.9)

project(libysmm VERSION 1.0.1 DESCRIPTION "libysmm")

add_library(ysmm SHARED
    src/cl.cpp
)

set_target_properties(ysmm PROPERTIES VERSION ${PROJECT_VERSION})

target_compile_features(ysmm PRIVATE cxx_std_17)
target_include_directories(ysmm PRIVATE include)
target_include_directories(ysmm PRIVATE src)

install(TARGETS ysmm
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
