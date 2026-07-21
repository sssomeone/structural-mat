cmake_minimum_required(VERSION 3.20)

project(StructuralMAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP QUIET COMPONENTS CXX)

# AppleClang does not discover Homebrew's libomp automatically.
if(APPLE AND NOT TARGET OpenMP::OpenMP_CXX)
    find_path(OPENMP_INCLUDE_DIR omp.h
        HINTS /opt/homebrew/opt/libomp /usr/local/opt/libomp
        PATH_SUFFIXES include)
    find_library(OPENMP_LIBRARY omp
        HINTS /opt/homebrew/opt/libomp /usr/local/opt/libomp
        PATH_SUFFIXES lib)
    if(OPENMP_INCLUDE_DIR AND OPENMP_LIBRARY)
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xpreprocessor -fopenmp)
        target_include_directories(OpenMP::OpenMP_CXX INTERFACE "${OPENMP_INCLUDE_DIR}")
        target_link_libraries(OpenMP::OpenMP_CXX INTERFACE "${OPENMP_LIBRARY}")
    endif()
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "OpenMP C++ support is required by the RVD implementation.")
endif()

set(STRUCTURAL_MAT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern" CACHE PATH
    "Directory containing polyscope, fcpw, LBFGSpp, portable-file-dialogs and optionally libigl")

set(POLYSCOPE_DIR "${STRUCTURAL_MAT_DEPS_DIR}/polyscope")
set(FCPW_DIR "${STRUCTURAL_MAT_DEPS_DIR}/fcpw")
set(LBFGSPP_DIR "${STRUCTURAL_MAT_DEPS_DIR}/LBFGSpp")
set(DIALOGS_DIR "${STRUCTURAL_MAT_DEPS_DIR}/portable-file-dialogs")

foreach(required_path
        "${POLYSCOPE_DIR}/CMakeLists.txt"
        "${FCPW_DIR}/CMakeLists.txt"
        "${LBFGSPP_DIR}/include/LBFGSB.h"
        "${DIALOGS_DIR}/portable-file-dialogs.h")
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "Missing dependency: ${required_path}\n"
            "Set STRUCTURAL_MAT_DEPS_DIR to the directory containing the required dependencies.")
    endif()
endforeach()

find_path(LIBIGL_INCLUDE_DIR
    NAMES igl/read_triangle_mesh.h
    HINTS "${STRUCTURAL_MAT_DEPS_DIR}/libigl/include"
    DOC "Path containing the libigl include directory")
if(NOT LIBIGL_INCLUDE_DIR)
    message(FATAL_ERROR "libigl was not found. Set LIBIGL_INCLUDE_DIR to its include directory.")
endif()

set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON CACHE BOOL "" FORCE)
add_subdirectory("${POLYSCOPE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/polyscope")

set(FCPW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FCPW_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
add_subdirectory("${FCPW_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/fcpw")

set(SURFACE_VORONOI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/surfaceVoronoi")
set(SPACE_VORONOI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/spaceVoronoi")

add_executable(structural_mat
    src/main.cpp
    src/RobustMeshSimplification.cpp
    "${SPACE_VORONOI_DIR}/src/VoronoiComputation.cpp"
)

target_link_libraries(structural_mat PRIVATE
    CGAL::CGAL
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    polyscope
    fcpw
)

target_include_directories(structural_mat PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${SURFACE_VORONOI_DIR}/src"
    "${SPACE_VORONOI_DIR}/src"
)

target_include_directories(structural_mat SYSTEM PRIVATE
    "${SURFACE_VORONOI_DIR}/extern/tsl"
    "${DIALOGS_DIR}"
    "${LBFGSPP_DIR}/include"
    "${FCPW_DIR}/include"
    "${FCPW_DIR}/deps/enoki/include"
    "${LIBIGL_INCLUDE_DIR}"
)

if(MSVC)
    target_compile_options(structural_mat PRIVATE /W4)
    target_compile_definitions(structural_mat PRIVATE _USE_MATH_DEFINES)
else()
    target_compile_options(structural_mat PRIVATE -Wall -Wextra -Wpedantic)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(structural_mat PRIVATE -Wno-invalid-specialization)
    endif()
endif()
