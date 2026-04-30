# Structural MAT: Clean and Scalable Medial Axis Simplification via Explicit Surface Correspondence

[![SIGGRAPH 2026](https://img.shields.io/badge/SIGGRAPH-2026-blue)](https://s2026.siggraph.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper:

> **Structural MAT: Clean and Scalable Medial Axis Simplification via Explicit Surface Correspondence**  
> Pengfei Wang, Shuangmin Chen, Dongming Yan, Ying He, Shiqing Xin, Changhe Tu, Wenping Wang  
> *ACM Transactions on Graphics (SIGGRAPH 2026)*

## 📖 Overview

This repository will host the official implementation of our SIGGRAPH 2026 paper on clean and scalable medial axis transform (MAT) simplification.

The medial axis transform is a powerful shape representation, but raw MATs are often noisy, redundant, and structurally cluttered. We propose a simplification framework based on **explicit surface correspondence** that produces clean, structurally meaningful medial axes while preserving geometric fidelity. Our method scales gracefully to large CAD models and complex geometries.

### Key Features

- 🧹 **Clean MAT output** — removes spurious branches, artifacts, and topological clutter
- 📐 **Explicit surface correspondence** — every medial element is anchored to surface regions
- ⚡ **Scalable** — efficient on large and complex CAD models
- 🎯 **Structure-preserving** — retains semantic and geometric features of the input shape


## 🚧 Code Release

> **Code is coming soon.** We are currently cleaning up the implementation and preparing documentation. Expected release: shortly after the SIGGRAPH 2026 camera-ready deadline.

Please **⭐ star** this repository to receive updates when the code is released.

## 📄 Paper

- 📑 [Paper PDF (coming soon)](#)
- 🎬 [Project page (coming soon)](#)
- 📦 arXiv preprint: *coming soon*

## 🖼️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{wang2026structuralmat,
  title   = {Structural {MAT}: Clean and Scalable Medial Axis Simplification via Explicit Surface Correspondence},
  author  = {Wang, Pengfei and Chen, Shuangmin and Yan, Dongming and He, Ying and Xin, Shiqing and Tu, Changhe and Wang, Wenping},
  journal = {ACM Transactions on Graphics (SIGGRAPH)},
  year    = {2026},
  publisher = {ACM}
}
```

## 🔗 Related Work

- [Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds](https://github.com/sssomeone/structural-mat) (SIGGRAPH 2026)
- [SurfaceVoronoi (SIGGRAPH Asia 2022)](#)
- [P2M (TOG 2023)](#)

## 📬 Contact

For questions or discussions, feel free to reach out:

- **Pengfei Wang** — pengfei1998@foxmail.com

## Acknowledgments

This work is supported by the National Key R&D Program of China (2022YFB3303200), the National Natural Science Foundation of China (62272277, U23A20312, 62072284, 62172415), the Natural Science Foundation of Shandong Province (ZR2020MF036), the Strategic Priority Research Program of the Chinese Academy of Sciences (XDB0640000 and XDB0640200), and the Beijing Natural Science Foundation (Z240002).

## 📜 License

Code will be released under the MIT License (see [LICENSE](LICENSE) once available).
