cmake_minimum_required(VERSION 3.11)

# Library version
set(LIBYSMM_VERSION_MAJOR 0)
set(LIBYSMM_VERSION_MINOR 1)
set(LIBYSMM_VERSION_PATCH 0)
set(LIBYSMM_VERSION "${LIBYSMM_VERSION_MAJOR}.${LIBYSMM_VERSION_MINOR}.${LIBYSMM_VERSION_PATCH}")

# Config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/libysmm_cl_config.h.cmake
               ${CMAKE_CURRENT_SOURCE_DIR}/include/libysmm_cl_config.h)

# CMake includes
include(FetchContent)

project(libysmm VERSION ${LIBYSMM_VERSION} DESCRIPTION "libysmm")

# Find OpenCL
find_package(OpenCL REQUIRED)

# Fetch Inja
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_TESTING OFF)
set(BUILD_BENCHMARK OFF)
FetchContent_Declare(inja URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.tar.gz)
FetchContent_MakeAvailable(inja)

add_library(ysmm-cl SHARED
    src/libysmm_cl.cpp
)

set_target_properties(ysmm-cl PROPERTIES VERSION ${PROJECT_VERSION})
target_compile_features(ysmm-cl PRIVATE cxx_std_17)
target_include_directories(ysmm-cl PUBLIC include)
target_include_directories(ysmm-cl PRIVATE src)
target_include_directories(ysmm-cl PRIVATE ${OpenCL_INCLUDE_DIRS})

target_link_directories(ysmm-cl PRIVATE ${OpenCL_LIBRARY})
target_link_libraries(ysmm-cl PRIVATE OpenCL)
target_link_libraries(ysmm-cl PRIVATE pantor::inja)

install(TARGETS ysmm-cl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
