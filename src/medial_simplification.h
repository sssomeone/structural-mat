
//
// Created by pengfei on 2025/9/15.
//

#ifndef SKELETON_SIMPLIFICATION_MEDIAL_SIMPLIFICATION_H
#define SKELETON_SIMPLIFICATION_MEDIAL_SIMPLIFICATION_H
#include <tuple> // 确保包含了 tuple 头文件

#include "featureExtraction.h"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <queue>
#include <set>
#include <Eigen/Dense>
#include <fcpw/fcpw.h>
struct OptimizationResult {
    Eigen::Vector3d optimal_point;  // q*
    double optimal_radius;          // r*
    double min_energy_total;        // 总最小能量
    double min_energy_triangle_quality;           // E1的能量值
    double min_energy_qem;           // E2的能量值
};
using QuadricTuple = std::tuple<Eigen::Matrix4d, Eigen::Vector4d, double>;
#include "MATOptimization.cpp"
class TriangleEnergyOptimizer {
public:
    struct TriangleInfo {
        std::vector<double> areas;              // S_i
        std::vector<Eigen::Vector4d> planes;    // (a_i, b_i, c_i, d_i)
        double total_area;
    };
    double error_ration=0.05;
    TriangleInfo cached_info_;  // 缓存的三角形信息

    // ============ 预计算三角形信息（内部使用） ============
    void precomputeTriangleInfo(
        const std::vector<std::vector<Eigen::Vector3d>>& triangles)
    {
        int n = triangles.size();

        cached_info_.areas.resize(n);

        cached_info_.planes.resize(n);
        cached_info_.total_area = 0.0;

        for (int i = 0; i < n; ++i) {
            const auto& tri = triangles[i];
            // 计算面积
            cached_info_.areas[i] = computeTriangleArea(tri);
            cached_info_.total_area += cached_info_.areas[i];
            // 计算平面参数
            cached_info_.planes[i] = computePlaneEquation(tri);
        }
    }
    // ============ 根据给定点和半径计算能量（使用缓存的info） ============
    struct EnergyResult {
        double total_energy;
        double min_energy_triangle_quality;
        double min_energy_qem;
    };


    // ============ 工具函数：计算能量 ============
    // 现在利用 QEM 矩阵快速计算 E_Sqem，利用点集计算 E_Euclidean
    EnergyResult computeEnergy(
        const QuadricTuple& sum_quadric,    // <--- 聚合后的 QEM 矩阵 (A, b, c)
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Vector3d& q,
        double r,
        double w_Euclidean)
    {
        // --- 1. 计算 E_Euclidean (正则项) ---
        // E_reg = (1/N) * sum ||p_i - q||^2
        double E_euclidean = 0.0;
        if (!points.empty()) {
            for (const auto& p : points) {
                E_euclidean += (p - q).squaredNorm();
            }
            E_euclidean /= points.size();
        }

        // --- 2. 计算 E_Sqem (表面误差) ---
        // 利用 Quadric 公式: E = x^T * A * x + 2 * b^T * x + c
        // 其中 x = [q_x, q_y, q_z, r]^T
        Eigen::Vector4d x;
        x << q, r;

        const auto& [A, b_vec, c_val] = sum_quadric;

        // 注意公式：E = x.T * A * x + 2 * b.T * x + c
        double E_Sqem = x.dot(A * x) + 2.0 * b_vec.dot(x) + c_val;

        // --- 3. 总能量 ---
        EnergyResult result;
        result.min_energy_triangle_quality = E_euclidean;
        result.min_energy_qem = E_Sqem;
        result.total_energy = w_Euclidean * E_euclidean + E_Sqem;

        return result;
    }


    // ============ 核心优化函数 ============
    OptimizationResult optimize(
        const QuadricTuple& sum_quadric,            // 输入：累加好的 QEM 信息 (替代了 triangles 和 cached_info_)
        const std::vector<Eigen::Vector3d>& points, // 输入：用于正则项的邻居点
        double w_Euclidean)
    {
        // 解包 QEM 矩阵
        const auto& [A_qem, b_qem, c_qem] = sum_quadric;

        // ============ 1. 构建正则项 (Euclidean) 的 QEM 形式 ============
        // 目标：将 E_reg = w * (1/N) * sum ||q - p_i||^2 写成 4x4 矩阵形式
        // 展开：E_reg = w * ( q.dot(q) - 2 * q.dot(p_bar) + const )
        // 对应的变量是 x = [q, r]

        int n = points.size();
        Eigen::Matrix4d A_reg = Eigen::Matrix4d::Zero();
        Eigen::Vector4d b_reg = Eigen::Vector4d::Zero();

        if (n > 0) {
            // 计算重心 p_bar
            Eigen::Vector3d p_bar = Eigen::Vector3d::Zero();
            for (const auto& p : points) {
                p_bar += p;
            }
            p_bar /= n;

            // 构造 A_reg: 对应 q.dot(q) 部分，左上角 3x3 为单位阵，r 的系数为 0
            A_reg.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

            // 构造 b_reg: 对应 -2 * q.dot(p_bar) 部分
            // 二次型一次项系数通常为 2*b^T*x，所以 b 应该是 -p_bar
            b_reg.head<3>() = -p_bar;

            // 注意：我们这里不需要计算常数项 c_reg，因为它不影响导数求解最优值
        }

        // ============ 2. 融合总矩阵 ============
        // 目标函数 E_total = E_sqem + w * E_reg
        // 对应矩阵叠加：
        Eigen::Matrix4d A_total = A_qem + w_Euclidean * A_reg;
        Eigen::Vector4d b_total = b_qem + w_Euclidean * b_reg;

        // ============ 3. 求解线性方程组 ============
        // 求导为 0 => 2 * A * x + 2 * b = 0  =>  A * x = -b
        Eigen::Vector4d x_star = A_total.ldlt().solve(-b_total);

        // ============ 4. 提取结果 ============
        Eigen::Vector3d q_star = x_star.head<3>();
        double r_star = x_star(3);

        // ============ 5. 计算最终能量并返回 ============
        EnergyResult energy = computeEnergy(sum_quadric, points, q_star, r_star, w_Euclidean);

        return transEnergyResultToOptimizationResult(energy, q_star, r_star);
    }
    OptimizationResult transEnergyResultToOptimizationResult(const EnergyResult&energy,const Eigen::Vector3d& q_star, const double r_star)const {
        OptimizationResult result;
        result.optimal_point = q_star;
        result.optimal_radius = r_star;
        result.min_energy_total = energy.total_energy;
        result.min_energy_triangle_quality = energy.min_energy_triangle_quality;
        result.min_energy_qem = energy.min_energy_qem;
        return result;
    }


public:
    // 计算三角形面积（使用叉积）
    double computeTriangleArea(const std::vector<Eigen::Vector3d>& vertices) const {
        Eigen::Vector3d v0 = vertices[0];
        Eigen::Vector3d v1 = vertices[1];
        Eigen::Vector3d v2 = vertices[2];

        Eigen::Vector3d edge1 = v1 - v0;
        Eigen::Vector3d edge2 = v2 - v0;

        return 0.5 * edge1.cross(edge2).norm();
    }
    Eigen::Vector4d computePlaneEquation(const std::vector<Eigen::Vector3d>& vertices)const {
        // 至少需要3个点来确定一个平面
        if (vertices.size() < 3) {
            throw std::invalid_argument("At least 3 vertices are required to define a plane");
        }

        // 使用第一个点作为基准点
        const Eigen::Vector3d& p0 = vertices[0];
        Eigen::Vector3d normal(0, 0, 0);

        Eigen::Vector3d v1 = vertices[1] - p0;
        Eigen::Vector3d v2 = vertices[2] - p0;

        normal=v1.cross(v2);
        // 检查是否找到了有效的法向量
        if (normal.norm() < 1e-12) {
            // throw std::runtime_error("All vertices are collinear, cannot define a plane");
        }

        // 归一化法向量
        normal.normalize();

        // 计算d: 平面方程 ax + by + cz + d = 0
        // 将p0代入方程: a*p0.x + b*p0.y + c*p0.z + d = 0
        // 因此: d = -(a*p0.x + b*p0.y + c*p0.z) = -normal.dot(p0)
        double d = -normal.dot(p0);


        // 返回平面方程参数 [a, b, c, d]
        return -Eigen::Vector4d(normal.x(), normal.y(), normal.z(), d);
    }


public:
    // ============ 带半径正则化的能量计算 ============
    struct EnergyResultWithRegularization {
        double total_energy;
        double E1;  // 点到点能量
        double E2;  // 点到平面能量
        double E3;  // 半径正则化能量
    };

    EnergyResultWithRegularization computeEnergyWithRegularization(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Vector3d& q,
        double r,
        double w_Euclidean,
        double w_feature) const
    {
        int n_tris = cached_info_.areas.size();
        int n_points = points.size();

        // ============ 计算 E1 能量 ============
        double E_Euclidean = 0.0;
        for (const auto& p : points) {
            E_Euclidean += (p - q).squaredNorm();
        }
        E_Euclidean /= n_points;

        // ============ 计算 E2 能量 ============
        double E_Sqem = 0.0;
        for (int i = 0; i < n_tris; ++i) {
            Eigen::Vector3d n_i = cached_info_.planes[i].head<3>();
            double d_i = cached_info_.planes[i](3);

            // dist_i = n_i · q + d_i
            double dist_i = n_i.dot(q) + d_i;

            // E2 += S_i * (dist_i - r)^2
            E_Sqem += cached_info_.areas[i] * std::pow(dist_i - r, 2);
        }

        // ============ 计算 E3 能量（半径正则化） ============
        double E_feature = r * r;

        // ============ 计算加权总能量 ============
        EnergyResultWithRegularization result;
        result.E1 = E_Euclidean;
        result.E2 = E_Sqem;
        result.E3 = E_feature;
        result.total_energy = w_Euclidean * E_Euclidean + E_Sqem + w_feature * E_feature;

        return result;
    }


    // ============ 带边界约束和半径正则化的优化 ============
    OptimizationResult optimize_boundary_points(
        const std::vector<std::vector<Eigen::Vector3d>>& triangles,
        const std::vector<Eigen::Vector3d>& points,  // E1能量的点集
        double w_Euclidean,
        double w_feature)  // 新增：半径正则化权重
    {
        // ============ 第一步：基本参数 ============
        int m = triangles.size();  // 三角形数量
        int n = points.size();     // 点的数量

        // ============ 第二步：计算第一能量E1相关的量 ============
        // 计算点集的重心 p_bar
        Eigen::Vector3d p_bar = Eigen::Vector3d::Zero();
        for (int i = 0; i < n; ++i) {
            p_bar += points[i];
        }
        p_bar /= n;

        // ============ 第三步：计算第二能量E2相关的量 ============
        Eigen::Vector3d n_bar = Eigen::Vector3d::Zero();
        double d_bar = 0.0;

        for (int i = 0; i < m; ++i) {
            n_bar += cached_info_.areas[i] * cached_info_.planes[i].head<3>();
            d_bar += cached_info_.areas[i] * cached_info_.planes[i](3);
        }
        n_bar /= cached_info_.total_area;
        d_bar /= cached_info_.total_area;

        // ============ 第四步：计算自适应权重系数 α ============
        // α = (w2 * S_total) / (w2 * S_total + w3)
        double alpha = (cached_info_.total_area) / (cached_info_.total_area + w_feature);

        // ============ 第五步：计算修正的协方差矩阵 M' 和向量 v' ============
        // 使用修正的法向量: n'_j = n_j - α*n_bar
        // 使用修正的距离: d'_j = d_j - α*d_bar
        Eigen::Matrix3d M_prime = Eigen::Matrix3d::Zero();
        Eigen::Vector3d v_prime = Eigen::Vector3d::Zero();

        for (int i = 0; i < m; ++i) {
            Eigen::Vector3d n_i = cached_info_.planes[i].head<3>();
            double d_i = cached_info_.planes[i](3);

            // 计算修正后的法向量和距离
            Eigen::Vector3d n_prime_i = n_i - alpha * n_bar;
            double d_prime_i = d_i - alpha * d_bar;

            // M' = Σ S_i * n'_i * n'_i^T
            M_prime += cached_info_.areas[i] * n_prime_i * n_prime_i.transpose();

            // v' = Σ S_i * d'_i * n'_i
            v_prime += cached_info_.areas[i] * d_prime_i * n_prime_i;
        }

        // ============ 第六步：构造总系统矩阵 ============
        // M_total = w1*n*I + w2*M' + w3*α²*n_bar*n_bar^T
        Eigen::Matrix3d M_total = w_Euclidean * n * Eigen::Matrix3d::Identity()
                                + M_prime
                                + w_feature * alpha * alpha * (n_bar * n_bar.transpose());

        // ============ 第七步：构造右端向量 ============
        // b = w1*n*p_bar - w2*v' - w3*α²*d_bar*n_bar
        Eigen::Vector3d b = w_Euclidean * n * p_bar
                          - v_prime
                          - w_feature * alpha * alpha * d_bar * n_bar;

        // ============ 第八步：求解线性方程组 ============
        Eigen::Vector3d q_star = M_total.ldlt().solve(b);

        // ============ 第九步：计算最优半径 ============
        // r* = α * (n_bar · q* + d_bar)
        double r_star = alpha * (n_bar.dot(q_star) + d_bar);

        // ============ 第十步：计算最优点和半径下的能量 ============
        EnergyResultWithRegularization energy = computeEnergyWithRegularization(
            points, q_star, r_star, w_Euclidean, w_feature);

        // ============ 返回结果 ============
        OptimizationResult result;
        result.optimal_point = q_star;
        result.optimal_radius = r_star;
        result.min_energy_total = energy.total_energy;
        result.min_energy_triangle_quality = energy.E1;
        result.min_energy_qem = energy.E2;
        // 注意：这里复用了 min_energy_qem 字段来存储 E2，
        // E3 的值包含在 total_energy 中

        return result;
    }

};

// ========== 自定义简化器：重载边代价计算和折叠回调 ==========
class CustomMeshSimplifier : public RobustMeshSimplification {
private:
    std::vector<double> cached_signed_distance;  // 顶点到表面的距离缓存

    bool boundry_feature_preserve=false;


    vector<Eigen::Vector3d> voronoi_sites;

    vector<bool> boundry_vertex_isFixed;
    vector<bool> boundry_vertex_concave;


public:
    double k=100;

    double stable_radio_threshold=0.1;
    double w_triangle=0;

    const fcpw::Scene<3>& scene;

    vector<QuadricTuple>  MedialSphere_exclude_concave;
    vector<map<int,double>>   concave_sites;
    /**
     * @brief 根据平面参数和面积构建 QEM 矩阵信息
     * * @param plane_params 平面参数 (a, b, c, d)，其中 ax + by + cz + d = 0，且 a^2+b^2+c^2=1
     * @param area 三角形的面积 S_i
     * @return std::tuple<Matrix4d, Vector4d, double> 对应的 (A, b, c)
     */
    QuadricTuple computePlaneQuadric(const Eigen::Vector4d& plane_params, double area) {
        // 1. 提取法向 n = (a, b, c) 和截距 d
        // plane_params.head<3>() 就是法向量，题目保证了它是归一化的
        Eigen::Vector3d n = plane_params.head<3>();
        double d = plane_params(3);

        // 2. 构造辅助向量 u = [a, b, c, -1]^T
        // 对应公式中的 vector u = [n; -1]
        Eigen::Vector4d u;
        u << n, -1.0;

        // 3. 计算 A = S * (u * u^T)
        // 利用外积计算 4x4 矩阵
        Eigen::Matrix4d A = area * (u * u.transpose());

        // 4. 计算 b = S * d * u
        // 注意：这里计算的是用于 2*b^T*x 中的 b 向量
        Eigen::Vector4d b_vec = (area * d) * u;

        // 5. 计算 c = S * d^2
        double c_scalar = area * d * d;

        return std::make_tuple(A, b_vec, c_scalar);
    }
    // 伪代码：合并两个点的 Quadric
    QuadricTuple mergeQuadrics(const QuadricTuple& q1, const QuadricTuple& q2) {
        return std::make_tuple(
            std::get<0>(q1) + std::get<0>(q2), // A_new = A1 + A2
            std::get<1>(q1) + std::get<1>(q2), // b_new = b1 + b2
            std::get<2>(q1) + std::get<2>(q2)  // c_new = c1 + c2
        );
    }

    vector<vector<vector<Eigen::Vector3d>>> regions_of_each_site;
    vector<map<int,double>> sites_of_each_vertex;
    vector<Eigen::Vector3d> optimal_vertex_position;


    double distancePointToTriangle(
    const std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>& triangle,
    const Eigen::Vector3d& q)
    {
        const Eigen::Vector3d& a = std::get<0>(triangle);
        const Eigen::Vector3d& b = std::get<1>(triangle);
        const Eigen::Vector3d& c = std::get<2>(triangle);

        const Eigen::Vector3d ab = b - a;
        const Eigen::Vector3d ac = c - a;
        const Eigen::Vector3d ap = q - a;

        const double d1 = ab.dot(ap);
        const double d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) {
            return (q - a).norm(); // 最近点是顶点 a
        }

        const Eigen::Vector3d bp = q - b;
        const double d3 = ab.dot(bp);
        const double d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) {
            return (q - b).norm(); // 最近点是顶点 b
        }

        const Eigen::Vector3d cp = q - c;
        const double d5 = ab.dot(cp);
        const double d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) {
            return (q - c).norm(); // 最近点是顶点 c
        }

        const double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            const double v = d1 / (d1 - d3);
            const Eigen::Vector3d closest = a + v * ab; // 最近点在边 ab 上
            return (q - closest).norm();
        }

        const double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            const double v = d2 / (d2 - d6);
            const Eigen::Vector3d closest = a + v * ac; // 最近点在边 ac 上
            return (q - closest).norm();
        }

        const double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            const double v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            const Eigen::Vector3d closest = b + v * (c - b); // 最近点在边 bc 上
            return (q - closest).norm();
        }

        // 最近点在三角形内部
        const double denom = 1.0 / (va + vb + vc);
        const double v = vb * denom;
        const double w = vc * denom;
        const Eigen::Vector3d closest = a + v * ab + w * ac;
        return (q - closest).norm();
    }

    double computePolygonArea(const std::vector<Eigen::Vector3d>& vertices)const {
        // 至少需要3个点来形成一个多边形
        if (vertices.size() < 3) {
            throw std::invalid_argument("At least 3 vertices are required to define a polygon");
        }

        // 使用扇形三角剖分法（Fan Triangulation）
        // 以第一个顶点为中心，将多边形分解为多个三角形
        const Eigen::Vector3d& v0 = vertices[0];
        double totalArea = 0.0;

        // 遍历所有相邻的顶点对，形成三角形
        for (size_t i = 1; i < vertices.size() - 1; ++i) {
            const Eigen::Vector3d& v1 = vertices[i];
            const Eigen::Vector3d& v2 = vertices[i + 1];

            // 计算从v0出发的两条边向量
            Eigen::Vector3d edge1 = v1 - v0;
            Eigen::Vector3d edge2 = v2 - v0;

            // 三角形面积 = ||edge1 × edge2|| / 2
            Eigen::Vector3d crossProduct = edge1.cross(edge2);
            double triangleArea = 0.5 * crossProduct.norm();

            totalArea += triangleArea;
        }
        return totalArea;
    }

    double getDiff(double a, double b) {
        double abs_a = std::abs(a);
        double abs_b = std::abs(b);

        // 如果两个都是0，差异为0
        if (abs_a + abs_b < 1e-7) return 0.0;

        return std::abs(a - b) / (abs_a + abs_b);
    }

    vector<FeatureType> _sites_feature_crossing_flags;
    CustomMeshSimplifier(const fcpw::Scene<3>& _scene, const Eigen::MatrixXd& _vertices, const Eigen::MatrixXi& _faces,
        const vector<vector<vector<Eigen::Vector3d>>>& _regions_of_each_vert,
        const std::vector<Eigen::Vector4i>& vertex_sites, vector<Eigen::Vector3d>& samples_sites, bool _boundry_feature_preserve, vector<FeatureType>&sites_feature_crossing_flags,
        double _w_euclidean=0.0001,double _w_stability=0.1)
        : RobustMeshSimplification(_vertices, _faces),
          boundry_feature_preserve(_boundry_feature_preserve),
          voronoi_sites(samples_sites),
          stable_radio_threshold(_w_stability),
          w_triangle(_w_euclidean),
          scene(_scene),
          regions_of_each_site(_regions_of_each_vert),
          _sites_feature_crossing_flags(sites_feature_crossing_flags) {

        sites_of_each_vertex.resize(_vertices.rows());

        initializeMesh();

        boundry_vertex_isFixed = vector<bool>(vertices.size(),false);
        boundry_vertex_concave = vector<bool>(vertices.size(),false);


        if (boundry_feature_preserve){
            for (int i=0;i<vertices.size();++i) {
                boundry_vertex_concave[i] = sites_feature_crossing_flags[vertex_sites[i](0)]==FeatureType::YIN_CONCAVE||sites_feature_crossing_flags[vertex_sites[i](1)]==FeatureType::YIN_CONCAVE ||
                                sites_feature_crossing_flags[vertex_sites[i](2)]==FeatureType::YIN_CONCAVE ||sites_feature_crossing_flags[vertex_sites[i](3)]==FeatureType::YIN_CONCAVE;

                if (!vertices[i].is_boundary) {
                    continue;
                }
                boundry_vertex_isFixed[i] = sites_feature_crossing_flags[vertex_sites[i](0)]==FeatureType::YANG_CONVEX||sites_feature_crossing_flags[vertex_sites[i](1)]==FeatureType::YANG_CONVEX ||
                                sites_feature_crossing_flags[vertex_sites[i](2)]==FeatureType::YANG_CONVEX ||sites_feature_crossing_flags[vertex_sites[i](3)]==FeatureType::YANG_CONVEX;

                if (!boundry_vertex_isFixed[i]) {
                    continue;
                }

                TriangleEnergyOptimizer opt;
                vector<vector<Eigen::Vector3d>> tris;
                for (auto s:vertex_sites[i]) {
                    for (auto& region:regions_of_each_site[s]) {
                        for (int i=2;i<region.size();++i) {
                            tris.push_back(vector<Eigen::Vector3d>{region[0],region[1],region[i]});
                        }
                    }
                }
                opt.precomputeTriangleInfo(tris);

                vector<Eigen::Vector3d> neighbor_vertices({vertices[i].position});
                OptimizationResult res;
                res = opt.optimize_boundary_points(tris, neighbor_vertices, 0.1 * w_triangle, 100000);



                fcpw::Interaction<3> interaction;
                fcpw::Vector3 q(res.optimal_point.x(), res.optimal_point.y(), res.optimal_point.z());

                scene.findClosestPoint(q, interaction);


                double dis=interaction.d;




                for (auto v:opt.cached_info_.planes) {
                    dis=max(dis,abs(v.dot(Eigen::Vector4d(res.optimal_point(0),res.optimal_point(1),res.optimal_point(2),1))));
                }
                for (auto s:vertex_sites[i]) {
                    if (boundry_vertex_concave[i]) {
                        double concave_dis=1e9;
                        for (auto& region:regions_of_each_site[s]) {
                            for (int i=2;i<region.size();++i) {
                                concave_dis=min(concave_dis,distancePointToTriangle(tuple(region[0],region[i-1],region[i]),res.optimal_point));
                            }
                        }
                        dis=max(dis,concave_dis);
                    }
                }
                if (dis>sqrt(3.0)*0.005) {
                    continue;
                }
                vertices[i].position=res.optimal_point;
            }
        }

        optimal_vertex_position.resize(edges.size());

        MedialSphere_exclude_concave.resize(vertices.size());
        concave_sites.resize(vertices.size());

        cached_signed_distance.resize(vertices.size());

        for (int i=0;i<vertices.size();++i) {
            QuadricTuple  current_quadric = QuadricTuple(Eigen::Matrix4d::Zero(),Eigen::Vector4d::Zero(),0);
            TriangleEnergyOptimizer opt;

            fcpw::Interaction<3> interaction;
            fcpw::Vector3 q(vertices[i].position.x(), vertices[i].position.y(), vertices[i].position.z());
            scene.findClosestPoint(q, interaction);
            cached_signed_distance[i] = interaction.d;


            for (auto s:vertex_sites[i]) {

                double Area=0;
                for (auto& region:regions_of_each_site[s]) {
                    for (int i=2;i<region.size();++i) {
                        auto tri = vector<Eigen::Vector3d>{region[0],region[i-1],region[i]};
                        double s = opt.computeTriangleArea(tri);
                        Area+=opt.computeTriangleArea(tri);
                        if (sites_feature_crossing_flags[s]!=FeatureType::YIN_CONCAVE) {
                            current_quadric = mergeQuadrics(current_quadric,computePlaneQuadric(opt.computePlaneEquation(tri),s));
                        }
                    }
                }
                sites_of_each_vertex[i][s]=Area;
                if (sites_feature_crossing_flags[s]==FeatureType::YIN_CONCAVE) {
                    concave_sites[i][s] = Area;
                    continue;
                }
            }
            MedialSphere_exclude_concave[i]=current_quadric;
        }

    }


    double calculateEdgeCollapseEnergy(int edge_id) override {
        const Edge& edge = edges[edge_id];
        int v1=edge.v1, v2 = edge.v2;

        vector<Eigen::Vector3d> neighbor_vertices;

        // if (isBoundaryEdge(edge_id)) {
        if (false) {
            neighbor_vertices.push_back(vertices[v1].position);
            neighbor_vertices.push_back(vertices[v2].position);
        }
        else {
            for (auto s:vertices[v1].neighbors) {
                if (s!=v2)
                    neighbor_vertices.push_back(vertices[s].position);
            }
            for (auto s:vertices[v2].neighbors) {
                if (s!=v1)
                    neighbor_vertices.push_back(vertices[s].position);
            }
        }
        TriangleEnergyOptimizer opt;

        double triangle_quality_maxmal_error=0;

        double min_stability_ratio = stability_ratio(edge_id);

        OptimizationResult optimal_result;

        auto cal_concave_cost=[&](const Eigen::Vector3d& position,double r) {
            double concave_cost=0;
            for (auto v:concave_sites[v1]) {
                concave_cost+=v.second*pow(abs((voronoi_sites[v.first]-position).norm()-r),2);
            }


            for (auto v:concave_sites[v2]) {
                concave_cost+=v.second*pow(abs((voronoi_sites[v.first]-position).norm()-r),2);
            }
            return concave_cost;
        };
        auto  edge_Quadrics=mergeQuadrics(MedialSphere_exclude_concave[v1],MedialSphere_exclude_concave[v2]);

        if (boundry_vertex_concave[v1]||boundry_vertex_concave[v2]) {
        // if (false){
            std::vector<SurfacePoint> points;
            std::vector<Plane> planes;// 20个面


            for (auto s:sites_of_each_vertex[v1]) {
                points.push_back({voronoi_sites[s.first],s.second});
            }
            for (auto s:sites_of_each_vertex[v2]) {
                points.push_back({voronoi_sites[s.first],s.second});
            }


            // for (auto s:sites_of_each_vertex[v1]) {
            //     if (_sites_feature_crossing_flags[s.first]==FeatureType::YIN_CONCAVE)
            //         points.push_back({voronoi_sites[s.first],s.second});
            // }
            // for (auto s:sites_of_each_vertex[v2]) {
            //     if (_sites_feature_crossing_flags[s.first]==FeatureType::YIN_CONCAVE)
            //         points.push_back({voronoi_sites[s.first],s.second});
            // }
            // map<int,double> current_sites;
            // for (auto s:sites_of_each_vertex[v1]) {
            //     if (_sites_feature_crossing_flags[s.first]==FeatureType::YIN_CONCAVE)
            //         continue;
            //     current_sites[s.first]+=s.second;
            // }
            // for (auto s:sites_of_each_vertex[v2]) {
            //     if (_sites_feature_crossing_flags[s.first]==FeatureType::YIN_CONCAVE)
            //         continue;
            //     current_sites[s.first]+=s.second;
            // }
            // vector<vector<Eigen::Vector3d>> tris;
            // vector<double> sites;
            // for (auto s:current_sites) {
            //     for (auto& region:regions_of_each_site[s.first]) {
            //         tris.push_back(vector<Eigen::Vector3d>{region[0],region[1],region[2]});
            //         sites.push_back(s.second);
            //     }
            // }
            // TriangleEnergyOptimizer opt;
            // opt.precomputeTriangleInfo(tris);
            // for (int i=0;i<opt.cached_info_.planes.size();++i) {
            //     Plane p;
            //     p.normal=Eigen::Vector3d(opt.cached_info_.planes[i](0),opt.cached_info_.planes[i](1),opt.cached_info_.planes[i](2));
            //     p.d=opt.cached_info_.planes[i](3);
            //     p.weight=sites[i];
            //     planes.push_back(p);
            // }




            auto getMaxDist = [](const auto& points, const Eigen::Vector3d& q) -> double {
                double max_sq_dist = 0.0; // 初始化最大平方距离

                for (const auto& point : points) {
                    // point.p 是你的结构体中的 Eigen 向量
                    // 使用 squaredNorm() 避免在循环中频繁开根号
                    double sq_d = (q - point.p).squaredNorm();
                    if (sq_d > max_sq_dist) {
                        max_sq_dist = sq_d;
                    }
                }
                return std::sqrt(max_sq_dist); // 最后开根号返回实际距离
            };

            // 初始猜测 (可以用邻居的重心作为 x，平均距离作为 r)
            Eigen::Vector3d x0;
            double r0 = 0;


            auto v1_max=getMaxDist(points, vertices[v1].position);
            auto v2_max=getMaxDist(points, vertices[v2].position);
            auto mid_max=getMaxDist(points, (vertices[v1].position+vertices[v2].position)/2);
            if (mid_max<v1_max&&mid_max<v2_max) {
                x0=(vertices[v1].position+vertices[v2].position)/2;
                r0=mid_max;
            }
            else if (v1_max<v2_max) {
                x0=vertices[v1].position;
                r0=v1_max;
            }
            else {
                x0=vertices[v2].position;
                r0=v2_max;
            }

            OptimizationConfig config;
            config.lambda_sqem = 1;
            config.lambda_triangle = w_triangle;

            // 调用优化
            auto res = optimize_medial_sphere(points, planes, neighbor_vertices, x0, r0, config);

            if(res.success) {
                optimal_result.optimal_point=res.opt_x;
                optimal_result.optimal_radius=res.opt_r;
                optimal_result.min_energy_qem=res.energy_fitting;
                optimal_result.min_energy_triangle_quality=res.energy_uniform;
                optimal_result.min_energy_total=res.final_energy;
            }
            else {
                optimal_result.optimal_point=(vertices[v1].position+vertices[v2].position)/2;
                optimal_result.optimal_radius=1e9;
                optimal_result.min_energy_qem=1e9;
                optimal_result.min_energy_triangle_quality=1e9;
            }
        }
        else if (boundry_vertex_isFixed[v1]||boundry_vertex_isFixed[v2]) {

            if (boundry_vertex_isFixed[v1]&&!boundry_vertex_isFixed[v2]) {
                auto r = opt.computeEnergy(edge_Quadrics,neighbor_vertices, vertices[v1].position, 0, w_triangle);
                optimal_result = opt.transEnergyResultToOptimizationResult(r,vertices[v1].position, 0);
                optimal_result.min_energy_qem+=cal_concave_cost(optimal_result.optimal_point,optimal_result.optimal_radius);
            }
            else if (boundry_vertex_isFixed[v2]&&!boundry_vertex_isFixed[v1]) {
                auto r = opt.computeEnergy(edge_Quadrics,neighbor_vertices, vertices[v2].position, 0, w_triangle);
                optimal_result = opt.transEnergyResultToOptimizationResult(r,vertices[v2].position, 0);
                optimal_result.min_energy_qem+=cal_concave_cost(optimal_result.optimal_point,optimal_result.optimal_radius);
            }
            else {
                auto energy_v1 = opt.computeEnergy(edge_Quadrics,neighbor_vertices, vertices[v1].position, 0, w_triangle);
                energy_v1.min_energy_qem+=cal_concave_cost(vertices[v1].position,0);

                auto energy_v2 = opt.computeEnergy(edge_Quadrics,neighbor_vertices, vertices[v2].position, 0, w_triangle);
                energy_v2.min_energy_qem+=cal_concave_cost(vertices[v2].position,0);

                auto energy_mid = opt.computeEnergy(edge_Quadrics,neighbor_vertices, (vertices[v1].position+vertices[v2].position)/2, 0, w_triangle);
                energy_mid.min_energy_qem+=cal_concave_cost((vertices[v1].position+vertices[v2].position)/2,0);

                double cost_v1 = w_triangle*(1+triangle_quality_maxmal_error) * energy_v1.min_energy_triangle_quality + energy_v1.min_energy_qem;
                double cost_v2 = w_triangle*(1+triangle_quality_maxmal_error) * energy_v2.min_energy_triangle_quality + energy_v2.min_energy_qem;
                double cost_mid = w_triangle*(1+triangle_quality_maxmal_error) * energy_mid.min_energy_triangle_quality + energy_mid.min_energy_qem;
                if (cost_mid<cost_v1&&cost_mid<cost_v2) {
                    optimal_result = opt.transEnergyResultToOptimizationResult(energy_mid,(vertices[v1].position+vertices[v2].position)/2, 0);
                }
                else if (cost_v1<cost_v2) {
                    optimal_result = opt.transEnergyResultToOptimizationResult(energy_v1,vertices[v1].position, 0);
                }
                else {
                    optimal_result = opt.transEnergyResultToOptimizationResult(energy_v2,vertices[v2].position, 0);
                }
            }
        }
        else {
            optimal_result = opt.optimize(edge_Quadrics,neighbor_vertices,w_triangle);
        }


        //
        // auto real_dis = abs(ptm->signed_distance(optimal_result.optimal_point).signedDistance);
        // double r = optimal_result.optimal_radius;
        //
        // if (abs(real_dis-r)>sqrt(3.0)*0.05&&(r>3*real_dis||real_dis>3*r)) {
        //     cerr<<"An error occurred during the optimization process; the midpoint was used as the target point."<<endl;
        //     auto energy_mid = opt.computeEnergy(edge_Quadrics,neighbor_vertices, (vertices[v1].position+vertices[v2].position)/2, 0, w_triangle);
        //     energy_mid.min_energy_qem+=cal_concave_cost((vertices[v1].position+vertices[v2].position)/2,0);
        //     double cost_mid = w_triangle*(1+w_omega*triangle_quality_maxmal_error) * energy_mid.min_energy_triangle_quality + energy_mid.min_energy_qem;
        //     optimal_result = opt.transEnergyResultToOptimizationResult(energy_mid,(vertices[v1].position+vertices[v2].position)/2, 0);
        // }

        // 在 calculateEdgeCollapseEnergy 函数中，获取到 optimal_result 之后




        // // 1. 获取优化结果
        // Eigen::Vector3d opt_p = optimal_result.optimal_point;
        // double r = optimal_result.optimal_radius;
        //
        // // 2. 使用 FCPW 计算该点到表面的【真实几何距离】
        // fcpw::Interaction<3> interaction_check;
        // fcpw::Vector3 q_check(opt_p.x(), opt_p.y(), opt_p.z());
        // scene.findClosestPoint(q_check, interaction_check);
        //
        // // 【修正点】FCPW 返回的是无符号距离，直接使用即可
        // double real_dis = interaction_check.d;
        //
        // // 3. 【恢复安全性检查】(核心修复)
        // // 如果 优化半径 和 几何距离 偏差太大，说明 MAT 优化发散了，必须回退！
        // if (abs(real_dis - r) > sqrt(3.0) * 0.05 && (r > 3 * real_dis || real_dis > 3 * r)) {
        //     // std::cerr << "Optimization diverged, falling back to midpoint." << std::endl;
        //
        //     // --- 回退策略：强制使用中点 ---
        //     Eigen::Vector3d mid_pos = (vertices[v1].position + vertices[v2].position) / 2.0;
        //
        //     // 计算中点的 Energy (纯几何误差)
        //     auto energy_mid = opt.computeEnergy(edge_Quadrics, neighbor_vertices, mid_pos, 0, w_triangle);
        //     // 加上凹点惩罚
        //     energy_mid.min_energy_qem += cal_concave_cost(mid_pos, 0);
        //
        //     // 重新打包结果
        //     optimal_result = opt.transEnergyResultToOptimizationResult(energy_mid, mid_pos, 0);
        //
        //     // 更新这一步的 optimal_vertex_position 缓存
        //     optimal_vertex_position[edge_id] = mid_pos;
        // }
        //
        //
        //


        optimal_vertex_position[edge_id]=optimal_result.optimal_point;

        double cost_triangle = w_triangle * optimal_result.min_energy_triangle_quality;
        double cost_qem = optimal_result.min_energy_qem;
        double cost = cost_qem + cost_triangle;

        // cout<<"cost_qem="<<cost_qem<<endl;
        // cout<<"cost_triangle="<<cost_triangle<<endl;

        return cost*(1.0/(1.0+exp(-k*(min_stability_ratio-stable_radio_threshold))));
    }
    Eigen::Vector3d getOptimalVertexPosition(int edge_id) const override {
        return optimal_vertex_position[edge_id];
    }

    double stability_ratio(int edge_id)const {
        const Edge& edge = edges[edge_id];
        Eigen::Vector3d v1=vertices[edge.v1].position;
        Eigen::Vector3d v2=vertices[edge.v2].position;

        auto res_1 = cached_signed_distance[edge.v1] ;
        auto res_2 = cached_signed_distance[edge.v2] ;

        double dh = max(0.0,(v1-v2).norm()-abs(abs(res_1)-abs(res_2)));

        return dh / (v1-v2).norm();
    }
    double stability_ratio_a(int edge_id)const {
        const Edge& edge = edges[edge_id];
        Eigen::Vector3d v1=vertices[edge.v1].position;
        Eigen::Vector3d v2=vertices[edge.v2].position;
        Eigen::Vector3d v_mid=(v1+v2)/2;

        auto res_1 = cached_signed_distance[edge.v1] ;
        auto res_2 = cached_signed_distance[edge.v2] ;


        fcpw::Interaction<3> interaction;
        fcpw::Vector3 q(v_mid.x(), v_mid.y(),v_mid.z());
        scene.findClosestPoint(q, interaction);


        double res_mid=interaction.d;

        double dh_whole = max(0.0,(v1-v2).norm()-abs(abs(res_1)-abs(res_2))) / (v1-v2).norm();
        double dh_l=max(0.0,(v1-v_mid).norm()-abs(abs(res_1)-abs(res_mid))) / (v1-v_mid).norm();
        double dh_r=max(0.0,(v_mid-v2).norm()-abs(abs(res_mid)-abs(res_2))) / (v_mid-v2).norm();

        return min(dh_l,min(dh_r,dh_whole));
    }


    // *** 新增：重载边折叠完成后的回调函数 ***
    void onEdgeCollapseCompleted(int v_removed, int v_kept) override {

        fcpw::Interaction<3> interaction;
        fcpw::Vector3 q(vertices[v_kept].position.x(), vertices[v_kept].position.y(),vertices[v_kept].position.z());
        scene.findClosestPoint(q, interaction);


        cached_signed_distance[v_kept] = interaction.d;

        for (auto s:sites_of_each_vertex[v_removed]) {
            sites_of_each_vertex[v_kept][s.first]+=s.second;
        }
        sites_of_each_vertex[v_removed].clear(); // 清空原vector



        boundry_vertex_concave[v_kept] = (boundry_vertex_concave[v_kept] || boundry_vertex_concave[v_removed]);
        boundry_vertex_isFixed[v_kept] = (boundry_vertex_isFixed[v_kept] || boundry_vertex_isFixed[v_removed]);
        MedialSphere_exclude_concave[v_kept]=mergeQuadrics(MedialSphere_exclude_concave[v_kept], MedialSphere_exclude_concave[v_removed]);


        for (auto v:concave_sites[v_removed]) {
            concave_sites[v_kept][v.first]+=v.second;
        }
        concave_sites[v_removed].clear();
    }
};

#endif //SKELETON_SIMPLIFICATION_MEDIAL_SIMPLIFICATION_H
