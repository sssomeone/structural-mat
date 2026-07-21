#ifndef VORONOI_COMPUTATION_H
#define VORONOI_COMPUTATION_H

#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <map>
#include <array>
#include <mutex>

// libigl for fast winding number
#include <igl/fast_winding_number.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>

namespace VoronoiComputation {

// CGAL类型定义
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;
typedef Delaunay::Point Point_3;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Cell_handle Cell_handle;
typedef Delaunay::Edge Edge;

// 结果结构体
struct VoronoiResult {
    Eigen::MatrixXd vertices;           // Voronoi顶点
    Eigen::MatrixXi faces;              // Voronoi面片（三角化后）
    std::vector<std::pair<int,int>> face_sites;
    std::vector<Eigen::Vector4i> vertex_sites;  // 每个顶点对应的四个站点

    VoronoiResult() = default;
};

// Voronoi顶点详细信息结构体
struct VoronoiVertexInfo {
    Eigen::Vector3d position;
    Eigen::Vector4i sites;
    double radius;  // 到四个站点的距离（外接球半径）
};

// 使用libigl fast winding number的内外点判断类
class FastMeshInside {
private:
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::FastWindingNumberBVH fwn_bvh;
    mutable std::mutex query_mutex;  // 保证线程安全

public:
    FastMeshInside(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);

    // 单点查询
    bool isInside(const Point_3& point) const;
    bool isInside(const Eigen::Vector3d& point) const;

    // 批量查询（更高效）
    std::vector<bool> isInside(const std::vector<Eigen::Vector3d>& points) const;
    std::vector<bool> isInside(const std::vector<Point_3>& points) const;
};

// 函数声明

// 辅助函数：Eigen向量转CGAL点
Point_3 eigenToPoint(const Eigen::Vector3d& v);

// 辅助函数：CGAL点转Eigen向量
Eigen::Vector3d pointToEigen(const Point_3& p);

// 计算四面体的外心（Voronoi顶点）
Point_3 computeCircumcenter(Cell_handle cell);

// 多边形三角化
std::vector<std::array<int, 3>> triangulatePolygon(const std::vector<int>& polygon);

// 主要计算函数
VoronoiResult computeVoronoi(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const std::vector<Eigen::Vector3d>& sites);

// 便利的调用接口
VoronoiResult computeVoronoi3D(const Eigen::MatrixXd& vertices,
                             const Eigen::MatrixXi& faces,
                             const std::vector<Eigen::Vector3d>& sites);

// 额外的辅助函数：获取Voronoi顶点的详细信息
std::vector<VoronoiVertexInfo> getVoronoiVertexDetails(const VoronoiResult& result,
                                                      const std::vector<Eigen::Vector3d>& sites);

} // namespace VoronoiComputation

#endif // VORONOI_COMPUTATION_H