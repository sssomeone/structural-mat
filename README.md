# Structural MAT

**Clean and Scalable Medial Axis Simplification via Explicit Surface Correspondence**

[![Paper](https://img.shields.io/badge/ACM%20TOG-10.1145%2F3811291-blue)](https://doi.org/10.1145/3811291)
[![SIGGRAPH 2026](https://img.shields.io/badge/SIGGRAPH-2026-blue)](https://s2026.siggraph.org/)

Pengfei Wang, Shuangmin Chen, Dongming Yan, Ying He, Shiqing Xin, Changhe Tu, and Wenping Wang

Structural MAT simplifies a medial axis initialized from a 3D Voronoi diagram of surface samples. During progressive simplification, it explicitly tracks the correspondence between medial-axis vertices and surface regions. This correspondence guides edge-collapse priorities and helps preserve structural alignment, boundary regularity, and triangle quality, even under aggressive simplification.

## Implementation note

The current implementation is generally consistent with the method described in the paper, with minor differences in a few implementation details. The standard version of the code is still being organized and will replace this version in the near future.

## Requirements

- CMake 3.20 or newer
- A C++17 compiler
- CGAL
- Eigen3
- OpenMP
- OpenGL and a working desktop graphics environment
- Polyscope
- FCPW
- LBFGSpp
- portable-file-dialogs
- libigl

The repository already contains the project-specific `surfaceVoronoi` and `spaceVoronoi` sources under `extern/`. The other third-party dependencies should be placed together in a dependency directory with the following layout:

```text
deps/
├── polyscope/
├── fcpw/
├── LBFGSpp/
├── portable-file-dialogs/
└── libigl/
    └── include/igl/
```

Clone dependencies with their submodules where applicable. For example:

```bash
mkdir deps
git clone --recursive https://github.com/nmwsharp/polyscope.git deps/polyscope
git clone --recursive https://github.com/rohan-sawhney/fcpw.git deps/fcpw
git clone https://github.com/yixuan/LBFGSpp.git deps/LBFGSpp
git clone https://github.com/samhocevar/portable-file-dialogs.git deps/portable-file-dialogs
git clone https://github.com/libigl/libigl.git deps/libigl
```

On macOS with Homebrew, the system dependencies can be installed with:

```bash
brew install cmake cgal eigen libomp
```

On Ubuntu/Debian, a typical system setup is:

```bash
sudo apt update
sudo apt install build-essential cmake libcgal-dev libeigen3-dev libomp-dev \
  libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

## Configure and build

If the third-party dependencies are placed directly under the repository's `extern/` directory, configure and build with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target structural_mat -j
```

If they are stored elsewhere, pass the dependency directory explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRUCTURAL_MAT_DEPS_DIR=/absolute/path/to/deps \
  -DLIBIGL_INCLUDE_DIR=/absolute/path/to/deps/libigl/include

cmake --build build --target structural_mat -j
```

`LIBIGL_INCLUDE_DIR` must point to the directory that contains the `igl/` folder. On macOS, CMake also searches the standard Homebrew locations for `libomp`.

The resulting executable is:

```text
build/structural_mat
```

## Run

Launch the application and select a mesh from the UI:

```bash
./build/structural_mat
```

Alternatively, pass an input mesh on the command line:

```bash
./build/structural_mat path/to/model.obj
```

The file dialog accepts common triangle-mesh formats including OBJ, PLY, OFF, and STL.

## Usage workflow

The processing stages depend on one another, so please use the controls in the following order:

1. **Load the input mesh.** Click `Browse Model File...`, unless a mesh was supplied on the command line.
2. **Choose the input type.** Decide whether the input is a CAD model. For CAD input, enable `CAD Input`, adjust the feature-angle threshold if needed, and click `Detect Feature Lines` before sampling.
3. **Set the simplification weights.** The current defaults are `Quality w1 = 0.000005` and `Stability ratio w = 0.025`.
4. **Sample the surface.** Set the number of samples, optionally enable `Blue Noise`, and click `Sampling`.
5. **Compute the surface Voronoi diagram.** Click `2D Voronoi`.
6. **Initialize the medial axis.** Click `3D Voronoi`.
7. **Initialize simplification.** Click `Init Queue` to construct the edge-collapse priority queue.
8. **Set the simplification controls.** `Step Size` is the number of edge collapses performed before the displayed result is refreshed. `Target Vertices` is the desired final number of medial-axis vertices.
9. **Start simplification.** Click `Simplify`. Use `Stop` to pause, or `Simplify One Edge` to inspect a single collapse.
10. **Inspect or save the result.** The current faces and edges are displayed as `Medial Axis (Ours)` and `Medial Axis Edges (Ours)`. Click `Save Model File...` to export the current result as OBJ.

When simplifying to a very small target, the result may contain wire edges that are not incident to any triangle. These edges are displayed separately as a Polyscope curve network so that the complete simplified structure remains visible.

## Paper

- [ACM Digital Library](https://dl.acm.org/doi/10.1145/3811291)
- [DOI: 10.1145/3811291](https://doi.org/10.1145/3811291)

## Citation

If you use this project in academic work, please cite:

```bibtex
@article{10.1145/3811291,
  author = {Wang, Pengfei and Chen, Shuangmin and Yan, Dongming and He, Ying and Xin, Shiqing and Tu, Changhe and Wang, Wenping},
  title = {Structural MAT: Clean and Scalable Medial Axis Simplification via Explicit Surface Correspondence},
  year = {2026},
  issue_date = {July 2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {45},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3811291},
  doi = {10.1145/3811291},
  journal = {ACM Trans. Graph.},
  month = jul,
  articleno = {164},
  numpages = {19}
}
```

## Contact

For questions or discussions, please contact:

- **Pengfei Wang** — pengfei1998@foxmail.com

## Acknowledgments

This work is supported by the National Key R&D Program of China (2022YFB3303200), the National Natural Science Foundation of China (62272277, U23A20312, 62072284, 62172415), the Natural Science Foundation of Shandong Province (ZR2020MF036), the Strategic Priority Research Program of the Chinese Academy of Sciences (XDB0640000 and XDB0640200), and the Beijing Natural Science Foundation (Z240002).
