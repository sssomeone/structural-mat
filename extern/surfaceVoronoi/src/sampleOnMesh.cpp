//
// Created by pengfei on 2025/9/8.
//
#include <random>
#include <Eigen/dense>
#include <iostream>
#include <sstream>
#include <fstream>
#include "robin_set.h"
#include "robin_map.h"
#include <vector>
#include <set>
#include <omp.h>
#include <queue>
#include<math.h>
#include <map>
#include <algorithm>
using namespace std;
#include <Eigen/Dense>
#include <igl/random_points_on_mesh.h>
#include <igl/blue_noise.h>
#include <igl/doublearea.h>
#include <igl/random_points_on_mesh.h>
using namespace std;

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd> blue_noise_sample(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const int n_desired)
{
    // 启发式计算半径
    const double r = [&V, &F](const int n) {
        Eigen::VectorXd A;
        igl::doublearea(V, F, A);
        return sqrt(((A.sum() * 0.5 / (n * 0.6162910373)) / M_PI));
    }(n_desired);

    printf("blue noise radius: %g\n", r);

    Eigen::MatrixXd P_blue;
    Eigen::MatrixXd B;
    Eigen::VectorXi I;

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 urbg(rd());

    igl::blue_noise(V, F, r, B, I, P_blue, urbg);

    return { P_blue, I, B };
}

/**
 * 白噪声采样函数
 * @param V 顶点矩阵 (n x 3)
 * @param F 面矩阵 (m x 3)
 * @param n 采样点数目
 * @return tuple<采样点, 面索引, 重心坐标>
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd> white_noise_sample(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const int n)
{
    Eigen::MatrixXd P;
    Eigen::MatrixXd B;
    Eigen::VectorXi FI;

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 urbg(rd());

    igl::random_points_on_mesh(n, V, F, B, FI, P, urbg);

    return { P, FI, B };
}

/**
 * 统一的采样函数接口
 * @param V 顶点矩阵 (n x 3)
 * @param F 面矩阵 (m x 3)
 * @param n 采样点数目
 * @param use_blue_noise true为蓝噪声，false为白噪声
 * @return tuple<采样点, 面索引, 重心坐标>
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd> sample_mesh_surface_complete(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int n,
    bool use_blue_noise = false)
{
    if (use_blue_noise) {
        return blue_noise_sample(V, F, n);
    } else {
        return white_noise_sample(V, F, n);
    }
}

/**
 * 简化版本：只返回采样点坐标
 * @param V 顶点矩阵 (n x 3)
 * @param F 面矩阵 (m x 3)
 * @param n 采样点数目
 * @param use_blue_noise true为蓝噪声，false为白噪声
 * @return 采样点的向量
 */
std::vector<Eigen::Vector3d> sample_mesh_surface(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int n,
    bool use_blue_noise = false)
{
    auto [P, FI, B] = sample_mesh_surface_complete(V, F, n, use_blue_noise);

    std::vector<Eigen::Vector3d> sample_points;
    sample_points.reserve(P.rows());

    for (int i = 0; i < P.rows(); ++i) {
        sample_points.push_back(P.row(i));
    }
    return sample_points;
}


/**
 * 封装版本: 返回采样点和采样点所在面片的索引
 * @param V 顶点矩阵 (n x 3)
 * @param F 面矩阵 (m x 3)
 * @param n 采样点数目
 * @param use_blue_noise true为蓝噪声,false为白噪声
 * @return pair<采样点向量, 面索引向量>
 */
std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> sample_mesh_surface_with_faces(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int n,
    bool use_blue_noise = false)
{
    // 1. 调用您已有的完整函数来获取所有数据
    // P: 采样点矩阵 (n_samples x 3)
    // FI: 面索引矩阵 (n_samples x 1)
    // B: 重心坐标 (n_samples x 3)
    auto [P, FI, B] = sample_mesh_surface_complete(V, F, n, use_blue_noise);

    // 2. 创建用于返回的 vector
    std::vector<Eigen::Vector3d> sample_points;
    std::vector<int> face_indices;

    // 3. 确定实际采样到的点的数量 (蓝噪声可能不精确等于n)
    const int num_samples = P.rows();

    // 4. 预分配内存以提高效率
    sample_points.reserve(num_samples);
    face_indices.reserve(num_samples);

    // 5. 遍历 Eigen 矩阵，将其转换为 std::vector
    for (int i = 0; i < num_samples; ++i) {
        // P.row(i) 返回一个 Eigen::RowVector3d，可以隐式转换为 Eigen::Vector3d
        sample_points.push_back(P.row(i));

        // FI 在您的元组定义中是 Eigen::MatrixXi
        // 它是一个 n_samples x 1 的矩阵，所以我们用 (i, 0) 访问
        face_indices.push_back(FI(i, 0));
    }

    // 6. 使用 C++17 的特性返回 pair
    return {sample_points, face_indices};

    // 或者使用传统方式:
    // return std::make_pair(sample_points, face_indices);
}


#include <random>
#include <vector>
#include <tuple>
#include <Eigen/Dense>


#include <random>
#include <vector>
#include <tuple>
#include <numeric> // for std::partial_sum
#include <algorithm> // for std::upper_bound
#include <Eigen/Dense>

/**
 * 辅助函数: 在单个三角面片内部进行 *一次* 均匀随机采样
 * (这是您之前的方法，它是正确的)
 */
Eigen::Vector3d sample_triangle_once(
    const std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>& triangle,
    std::mt19937& urbg,
    std::uniform_real_distribution<double>& dist)
{
    const auto& [v0, v1, v2] = triangle;
    double r1 = dist(urbg);
    double r2 = dist(urbg);

    if (r1 + r2 > 1.0) {
        r1 = 1.0 - r1;
        r2 = 1.0 - r2;
    }
    return (1.0 - r1 - r2) * v0 + r1 * v1 + r2 * v2;
}


/**
 * 在一个 3D 凸多边形 内部进行均匀随机采样
 * (使用 累积面积 + 二分查找)
 *
 * @param polygon 顶点列表 (v0, v1, v2, ...)，必须按顺序
 * @param n 采样点数目
 * @return n个采样点的向量 (保证均匀分布)
 */
std::vector<Eigen::Vector3d> sample_convex_polygon_uniform(
    const std::vector<Eigen::Vector3d>& polygon,
    int n)
{
    std::vector<Eigen::Vector3d> sampled_points;
    if (polygon.size() < 3) {
        return sampled_points;
    }

    // --- 1. 三角化并计算每个三角形的面积 ---
    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> triangles;
    std::vector<double> areas;

    const Eigen::Vector3d& v0 = polygon[0]; // 锚点

    for (size_t i = 1; i < polygon.size() - 1; ++i) {
        const Eigen::Vector3d& v1 = polygon[i];
        const Eigen::Vector3d& v2 = polygon[i+1];

        triangles.emplace_back(v0, v1, v2);

        // 面积 = 0.5 * || (v1 - v0) x (v2 - v0) ||
        double area = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();
        areas.push_back(area);
    }

    // --- 2. 创建累积面积分布 ---
    // (这是替代 std::discrete_distribution 的 "更简单" 的方法)
    std::vector<double> cumulative_areas(areas.size());
    std::partial_sum(areas.begin(), areas.end(), cumulative_areas.begin());

    double totalArea = cumulative_areas.empty() ? 0 : cumulative_areas.back();

    if (totalArea == 0.0) {
        // 退化多边形 (所有点共线)
        return sampled_points;
    }

    // --- 3. 初始化随机数生成器 ---
    std::random_device rd;
    std::mt19937 urbg(rd());
    // 我们需要一个在 [0, totalArea] 范围内的随机数
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> area_dist(0.0, totalArea);

    // --- 4. 采样 ---
    sampled_points.reserve(n);
    for (int i = 0; i < n; ++i) {
        // a. 随机选择一个 "面积值"
        double random_area = area_dist(urbg);

        // b. 找到这个 "面积值" 对应的三角形索引
        // (在累积数组中查找第一个 *大于* random_area 的元素)
        auto it = std::upper_bound(cumulative_areas.begin(), cumulative_areas.end(), random_area);
        int triangle_index = std::distance(cumulative_areas.begin(), it);

        // (安全检查，理论上不应该在 totalArea > 0 时发生)
        if (triangle_index >= triangles.size()) {
            continue;
        }

        // c. 在选中的三角形内采样一个点
        sampled_points.push_back(
            sample_triangle_once(triangles[triangle_index], urbg, dist)
        );
    }

    return sampled_points;
}


/**
 * 在 多个 3D 凸多边形 上进行均匀随机采样
 *
 * @param polygons 包含多个多边形的向量 (每个多边形是顶点的向量)
 * @param n 采样点总数
 * @return n个采样点的向量 (保证在所有多边形的总面积上均匀分布)
 */
std::vector<Eigen::Vector3d> sample_multiple_polygons(
    const std::vector<std::vector<Eigen::Vector3d>>& polygons,
    int n)
{
    std::vector<Eigen::Vector3d> sampled_points;
    sampled_points.reserve(n);

    // --- 1. 预处理: 将所有多边形三角化 (Fan Triangulation) ---
    //    并收集 *所有* 三角形及其面积到一个 "全局" 列表中

    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> all_triangles;
    std::vector<double> all_areas;

    for (const auto& polygon : polygons) {
        if (polygon.size() < 3) {
            continue; // 跳过无效多边形 (点或线)
        }

        const Eigen::Vector3d& v0 = polygon[0]; // 扇形锚点

        for (size_t i = 1; i < polygon.size() - 1; ++i) {
            const Eigen::Vector3d& v1 = polygon[i];
            const Eigen::Vector3d& v2 = polygon[i+1];

            // 存储这个三角形
            all_triangles.emplace_back(v0, v1, v2);

            // 计算并存储其面积
            double area = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();
            all_areas.push_back(area);
        }
    }

    // --- 2. 创建全局累积面积分布 ---
    std::vector<double> cumulative_areas(all_areas.size());
    std::partial_sum(all_areas.begin(), all_areas.end(), cumulative_areas.begin());

    double totalArea = cumulative_areas.empty() ? 0 : cumulative_areas.back();

    if (totalArea == 0.0 || all_triangles.empty()) {
        // 没有可供采样的有效面积
        return sampled_points; // 返回空向量
    }

    // --- 3. 初始化随机数生成器 ---
    std::random_device rd;
    std::mt19937 urbg(rd());
    // 用于重心坐标
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // 用于选择三角形
    std::uniform_real_distribution<double> area_dist(0.0, totalArea);

    // --- 4. 采样 n 个点 ---
    for (int i = 0; i < n; ++i) {
        // a. 随机选择一个 "面积值"
        double random_area = area_dist(urbg);

        // b. 找到这个 "面积值" 对应的 *全局* 三角形索引
        //    (在累积数组中查找第一个 *大于* random_area 的元素)
        auto it = std::upper_bound(cumulative_areas.begin(), cumulative_areas.end(), random_area);
        int triangle_index = std::distance(cumulative_areas.begin(), it);

        // c. 在选中的三角形内采样一个点
        sampled_points.push_back(
            sample_triangle_once(all_triangles[triangle_index], urbg, dist)
        );
    }

    return sampled_points;
}
