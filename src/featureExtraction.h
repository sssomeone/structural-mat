//
// Created by pengfei on 2025/11/28.
//

#ifndef SKELETON_SIMPLIFICATION_FATUREEXTRACTION_H
#define SKELETON_SIMPLIFICATION_FATUREEXTRACTION_H


#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm> // for std::max, std::min
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 定义特征类型
enum FeatureType {
    NON_FEATURE = 0,
    YANG_CONVEX = 1, // 阳角：凸棱 (Convex)
    YIN_CONCAVE = 2  // 阴角：凹槽 (Concave)
};

// 升级后的结构体：增加了 f1, f2
struct FeatureEdge {
    int v1;         // 边的端点1索引 (Vertex Index)
    int v2;         // 边的端点2索引 (Vertex Index)

    int f1;         // 相邻三角形1索引 (Face Index)
    int f2;         // 相邻三角形2索引 (Face Index)

    double angle;   // 法线夹角 (弧度)
    FeatureType type;
};

/**
 * @brief 提取特征线
 * @param V 顶点矩阵 (N x 3)
 * @param F 面索引矩阵 (M x 3)
 * @param angleThresholdDeg 角度阈值（度）。例如输入 30.0，表示法线夹角大于30度才算特征。
 * @return std::vector<FeatureEdge>
 */
std::vector<FeatureEdge> extractFeatureLines(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double angleThresholdDeg)
{
    std::vector<FeatureEdge> features;

    // 将角度阈值转换为弧度
    // 如果两个面的法线夹角小于此值，视为平滑区域，忽略之
    double thresholdRad = angleThresholdDeg * EIGEN_PI / 180.0;

    // 1. 构建边到面的拓扑映射 (Edge -> Faces)
    // Key: pair<min_v_idx, max_v_idx>, Value: vector<face_idx>
    // 使用 std::map 方便理解，生产环境大模型可换成 unordered_map 或半边结构
    std::map<std::pair<int, int>, std::vector<int>> edgeToFaces;

    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v1 = F(i, j);
            int v2 = F(i, (j + 1) % 3);

            // 保证 Key 的顺序一致 (v1 < v2)
            if (v1 > v2) std::swap(v1, v2);
            edgeToFaces[{v1, v2}].push_back(i);
        }
    }

    // 2. 预计算所有面的法线 (Face Normals)
    Eigen::MatrixXd faceNormals(F.rows(), 3);
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3d p0 = V.row(F(i, 0));
        Eigen::Vector3d p1 = V.row(F(i, 1));
        Eigen::Vector3d p2 = V.row(F(i, 2));

        // 注意：这里依赖顶点绕序 (CCW) 来确定法线方向
        // cross product 顺序：(1-0) x (2-0)
        Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0).normalized();
        faceNormals.row(i) = normal;
    }

    // 3. 遍历每一条边进行检测
    for (auto const& [edge, faces] : edgeToFaces) {
        // 特征线只定义在流形(Manifold)内部边上，必须恰好有两个相邻面
        if (faces.size() != 2) {
            continue;
        }

        int f1_idx = faces[0];
        int f2_idx = faces[1];

        Eigen::Vector3d n1 = faceNormals.row(f1_idx);
        Eigen::Vector3d n2 = faceNormals.row(f2_idx);

        // 计算点积
        double dot = n1.dot(n2);
        // 数值稳定性截断
        dot = std::max(-1.0, std::min(1.0, dot));

        // 计算夹角 (Angle between normals)
        // 0 表示平坦，PI 表示对折
        double angle = std::acos(dot);

        // --- 阈值判定 ---
        if (angle < thresholdRad) {
            // 夹角太小，视为平坦表面，不是特征线
            continue;
        }

        // --- 阴阳判定 (Convex/Concave Check) ---
        // 寻找 Face2 上不属于共享边的“对角点” (Opposite Vertex)
        int v_op_in_f2 = -1;
        for (int k = 0; k < 3; ++k) {
            int vid = F(f2_idx, k);
            if (vid != edge.first && vid != edge.second) {
                v_op_in_f2 = vid;
                break;
            }
        }

        // 取 Face1 上的任意一点（作为平面基准点）
        Eigen::Vector3d p_on_f1 = V.row(edge.first);
        Eigen::Vector3d p_op    = V.row(v_op_in_f2);

        // Signed Distance = n1 dot (p_op - p_on_f1)
        double dist = n1.dot(p_op - p_on_f1);

        FeatureType type = NON_FEATURE;
        double epsilon = 1e-6; // 浮点容差

        if (dist < -epsilon) {
            // 对角点在 Face1 平面“背面/下方” -> 面向下弯折 -> 凸 (Yang)
            type = YANG_CONVEX;
        } else if (dist > epsilon) {
            // 对角点在 Face1 平面“正面/上方” -> 面向上折起 -> 凹 (Yin)
            type = YIN_CONCAVE;
        } else {
            // 几乎共面但角度计算有微小偏差，或模型退化
            continue;
        }

        // 保存结果：包含边端点、相邻面索引、角度、类型
        features.push_back({
            edge.first,  // v1
            edge.second, // v2
            f1_idx,      // f1 (新增)
            f2_idx,      // f2 (新增)
            angle,
            type
        });
    }

    return features;
}



/**
 * @brief 提取特征线（区分阴阳角阈值）
 * @param V 顶点矩阵 (N x 3)
 * @param F 面索引矩阵 (M x 3)
 * @param convexAngleThresholdDeg 阳角（凸）阈值（度）
 * @param concaveAngleThresholdDeg 阴角（凹）阈值（度）
 * @return std::vector<FeatureEdge>
 */
std::vector<FeatureEdge> extractFeatureLines(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    double convexAngleThresholdDeg,  // 修改点：参数拆分
    double concaveAngleThresholdDeg  // 修改点：参数拆分
)
{
    std::vector<FeatureEdge> features;

    // 将两个角度阈值分别转换为弧度
    double convexThresholdRad = convexAngleThresholdDeg * EIGEN_PI / 180.0;
    double concaveThresholdRad = concaveAngleThresholdDeg * EIGEN_PI / 180.0;

    // 1. 构建边到面的拓扑映射 (Edge -> Faces)
    std::map<std::pair<int, int>, std::vector<int>> edgeToFaces;

    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v1 = F(i, j);
            int v2 = F(i, (j + 1) % 3);
            if (v1 > v2) std::swap(v1, v2);
            edgeToFaces[{v1, v2}].push_back(i);
        }
    }

    // 2. 预计算所有面的法线 (Face Normals)
    Eigen::MatrixXd faceNormals(F.rows(), 3);
    for (int i = 0; i < F.rows(); ++i) {
        Eigen::Vector3d p0 = V.row(F(i, 0));
        Eigen::Vector3d p1 = V.row(F(i, 1));
        Eigen::Vector3d p2 = V.row(F(i, 2));
        Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0).normalized();
        faceNormals.row(i) = normal;
    }

    // 3. 遍历每一条边进行检测
    for (auto const& [edge, faces] : edgeToFaces) {
        // 只处理流形内部边
        if (faces.size() != 2) {
            continue;
        }

        int f1_idx = faces[0];
        int f2_idx = faces[1];

        Eigen::Vector3d n1 = faceNormals.row(f1_idx);
        Eigen::Vector3d n2 = faceNormals.row(f2_idx);

        double dot = n1.dot(n2);
        dot = std::max(-1.0, std::min(1.0, dot));
        double angle = std::acos(dot); // 当前的二面角补角

        // 【优化】如果角度比两个阈值中最小的还要小，直接跳过，省去后面的几何计算
        if (angle < std::min(convexThresholdRad, concaveThresholdRad)) {
            continue;
        }

        // --- 阴阳判定 (Convex/Concave Check) ---
        // 寻找 Face2 上不属于共享边的“对角点”
        int v_op_in_f2 = -1;
        for (int k = 0; k < 3; ++k) {
            int vid = F(f2_idx, k);
            if (vid != edge.first && vid != edge.second) {
                v_op_in_f2 = vid;
                break;
            }
        }

        Eigen::Vector3d p_on_f1 = V.row(edge.first);
        Eigen::Vector3d p_op    = V.row(v_op_in_f2);

        // Signed Distance
        double dist = n1.dot(p_op - p_on_f1);

        FeatureType type = NON_FEATURE;
        double epsilon = 1e-6;

        // --- 核心修改逻辑 ---
        // 先判断类型，再应用对应的阈值
        if (dist < -epsilon) {
            // 对角点在背面 -> 凸 (Yang)
            if (angle < convexThresholdRad) continue; // 未达到阳角阈值，跳过
            type = YANG_CONVEX;
        } else if (dist > epsilon) {
            // 对角点在正面 -> 凹 (Yin)
            if (angle < concaveThresholdRad) continue; // 未达到阴角阈值，跳过
            type = YIN_CONCAVE;
        } else {
            // 共面，跳过
            continue;
        }

        features.push_back({
            edge.first,
            edge.second,
            f1_idx,
            f2_idx,
            angle,
            type
        });
    }

    return features;
}

#endif //SKELETON_SIMPLIFICATION_FATUREEXTRACTION_H