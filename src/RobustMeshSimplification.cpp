//
// Rewritten to support boundary and non-manifold edge collapse
// 重写以支持边界边和非流形边的折叠
//

#include "RobustMeshSimplification.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

RobustMeshSimplification::RobustMeshSimplification(
    const Eigen::MatrixXd& vertices_input,
    const Eigen::MatrixXi& faces_input)
    : V_original(vertices_input), F_original(faces_input) {
    // initializeMesh();
}

void RobustMeshSimplification::initializeMesh() {
    // 初始化顶点
    vertices.clear();
    vertices.reserve(V_original.rows());
    for (int i = 0; i < V_original.rows(); ++i) {
        Vertex v(V_original.row(i));
        v.id = i;
        vertices.push_back(v);
    }
    num_vertices = V_original.rows();

    // 初始化面
    faces.clear();
    faces.reserve(F_original.rows());
    for (int i = 0; i < F_original.rows(); ++i) {
        std::vector<int> face_verts = {F_original(i, 0), F_original(i, 1), F_original(i, 2)};
        Face f(face_verts);
        f.id = i;
        faces.push_back(f);
    }
    num_faces = F_original.rows();

    // 构建拓扑关系
    buildTopology();

    // 计算顶点的二次误差矩阵
    computeVertexQuadrics();

    // 分类边和顶点
    classifyEdges();
    classifyVertices();

    // 初始化边队列
    // updateEdgeQueue();
}

void RobustMeshSimplification::buildTopology() {
    edges.clear();
    vertex_pair_to_edge.clear();

    // 为每个面构建边
    for (int f_id = 0; f_id < faces.size(); ++f_id) {
        if (!faces[f_id].valid) continue;

        Face& face = faces[f_id];
        for (int i = 0; i < 3; ++i) {
            int v1 = face.vertices[i];
            int v2 = face.vertices[(i + 1) % 3];

            int edge_id = findOrCreateEdge(v1, v2);

            // 更新面-边关系
            face.edges.insert(edge_id);
            edges[edge_id].faces.insert(f_id);

            // 更新顶点-面关系
            vertices[v1].faces.insert(f_id);
            vertices[v2].faces.insert(f_id);
        }

        // 计算面法向量
        face.normal = computeFaceNormal(f_id);
    }

    std::cout << "Built topology: " << num_vertices << " vertices, "
              << num_edges << " edges, " << num_faces << " faces" << std::endl;
}

int RobustMeshSimplification::findOrCreateEdge(int v1, int v2) {
    if (v1 > v2) std::swap(v1, v2);

    auto key = std::make_pair(v1, v2);
    auto it = vertex_pair_to_edge.find(key);

    if (it != vertex_pair_to_edge.end()) {
        return it->second;
    }

    // 创建新边
    Edge new_edge(v1, v2);
    new_edge.id = edges.size();
    edges.push_back(new_edge);

    vertex_pair_to_edge[key] = new_edge.id;

    // 更新顶点-边关系
    vertices[v1].edges.insert(new_edge.id);
    vertices[v2].edges.insert(new_edge.id);

    // ✅ 新增：建立邻接关系
    vertices[v1].neighbors.insert(v2);
    vertices[v2].neighbors.insert(v1);

    num_edges++;
    return new_edge.id;
}

void RobustMeshSimplification::computeVertexQuadrics() {
    // 重置所有顶点的Q矩阵
    for (auto& vertex : vertices) {
        vertex.Q = Eigen::Matrix4d::Zero();
    }

    // 为每个面计算平面矩阵并累加到顶点
    for (int f_id = 0; f_id < faces.size(); ++f_id) {
        if (!faces[f_id].valid) continue;

        Eigen::Matrix4d K;
        computeFacePlaneMatrix(f_id, K);

        // 将面的贡献加到三个顶点
        const Face& face = faces[f_id];
        for (int vid : face.vertices) {
            vertices[vid].Q += K;
        }
    }
}

void RobustMeshSimplification::computeFacePlaneMatrix(int face_id, Eigen::Matrix4d& K) const {
    const Face& face = faces[face_id];

    // 获取三个顶点
    Eigen::Vector3d v0 = vertices[face.vertices[0]].position;
    Eigen::Vector3d v1 = vertices[face.vertices[1]].position;
    Eigen::Vector3d v2 = vertices[face.vertices[2]].position;

    // 计算平面方程 ax + by + cz + d = 0
    Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0).normalized();
    double d = -normal.dot(v0);

    // 构建平面向量
    Eigen::Vector4d plane(normal.x(), normal.y(), normal.z(), d);

    // K = plane * plane^T
    K = plane * plane.transpose();
}

void RobustMeshSimplification::classifyEdges() {
    for (int e_id = 0; e_id < edges.size(); ++e_id) {
        if (!edges[e_id].valid) continue;
        updateEdgeClassification(e_id);
    }
}

void RobustMeshSimplification::updateEdgeClassification(int edge_id) {
    Edge& edge = edges[edge_id];

    // 计算有效相邻面的数量
    int valid_face_count = 0;
    for (int f_id : edge.faces) {
        if (faces[f_id].valid) {
            valid_face_count++;
        }
    }

    // 分类边
    if (valid_face_count == 1) {
        edge.is_boundary = true;
        edge.is_non_manifold = false;
    } else if (valid_face_count == 2) {
        edge.is_boundary = false;
        edge.is_non_manifold = false;
    } else if (valid_face_count > 2) {
        edge.is_boundary = false;
        edge.is_non_manifold = true;
    } else {
        // valid_face_count == 0，孤立边
        edge.is_boundary = true;
        edge.is_non_manifold = false;
    }
}

void RobustMeshSimplification::classifyVertices() {
    for (auto& vertex : vertices) {
        if (!vertex.valid) continue;

        vertex.is_boundary = false;
        vertex.is_non_manifold = false;

        for (int e_id : vertex.edges) {
            if (edges[e_id].valid) {
                if (edges[e_id].is_boundary) {
                    vertex.is_boundary = true;
                }
                if (edges[e_id].is_non_manifold) {
                    vertex.is_non_manifold = true;
                }
            }
        }
    }
}

// ========== 安全性检查（新增） ==========

bool RobustMeshSimplification::canSafelyCollapse(int edge_id) const {
    if (edge_id >= edges.size() || !edges[edge_id].valid) {
        return false;
    }

    const Edge& edge = edges[edge_id];

    if (!vertices[edge.v1].valid || !vertices[edge.v2].valid) {
        return false;
    }

    // 获取最优位置
    Eigen::Vector3d optimal_pos = getOptimalVertexPosition(edge_id);

    // 检查是否会创建退化面
    if (wouldCreateDegenerateFace(edge_id, optimal_pos)) {
        return false;
    }

    // 检查拓扑有效性（轻量级检查）
    if (wouldCreateInvalidTopology(edge.v1, edge.v2)) {
        return false;
    }

    return true;
}

bool RobustMeshSimplification::wouldCreateDegenerateFace(
    int edge_id,
    const Eigen::Vector3d& new_pos) const {

    const Edge& edge = edges[edge_id];

    // 检查v1的面
    for (int f_id : vertices[edge.v1].faces) {
        if (!faces[f_id].valid) continue;

        const Face& face = faces[f_id];

        // 跳过会被删除的面（同时包含v1和v2）
        if (face.hasVertex(edge.v2)) continue;

        // 计算新位置后的面积
        std::vector<Eigen::Vector3d> new_positions;
        for (int vid : face.vertices) {
            if (vid == edge.v1) {
                new_positions.push_back(new_pos);
            } else {
                new_positions.push_back(vertices[vid].position);
            }
        }

        Eigen::Vector3d edge1 = new_positions[1] - new_positions[0];
        Eigen::Vector3d edge2 = new_positions[2] - new_positions[0];
        double area = 0.5 * edge1.cross(edge2).norm();

        if (area < min_face_area_threshold) {
            return true;  // 会创建退化面
        }

        // 检查法向量翻转
        Eigen::Vector3d original_normal = computeFaceNormal(f_id);
        Eigen::Vector3d new_normal = edge1.cross(edge2).normalized();

        if (abs(original_normal.dot(new_normal)) < normal_flip_threshold) {
            std::cout<<"-----------------------------"<<std::endl;
            std::cout<<original_normal.transpose()<< std::endl;
            std::cout<<new_normal.transpose()<< std::endl;
            std::cout<<original_normal.dot(new_normal)<< std::endl;
            return true;  // 法向量会大幅翻转
        }
    }

    // 检查v2的面
    for (int f_id : vertices[edge.v2].faces) {
        if (!faces[f_id].valid) continue;

        const Face& face = faces[f_id];

        // 跳过会被删除的面
        if (face.hasVertex(edge.v1)) continue;

        // 计算新位置后的面积
        std::vector<Eigen::Vector3d> new_positions;
        for (int vid : face.vertices) {
            if (vid == edge.v2) {
                new_positions.push_back(new_pos);
            } else {
                new_positions.push_back(vertices[vid].position);
            }
        }

        Eigen::Vector3d edge1 = new_positions[1] - new_positions[0];
        Eigen::Vector3d edge2 = new_positions[2] - new_positions[0];
        double area = 0.5 * edge1.cross(edge2).norm();

        if (area < min_face_area_threshold) {
            return true;
        }

        // 检查法向量翻转
        Eigen::Vector3d original_normal = computeFaceNormal(f_id);
        Eigen::Vector3d new_normal = edge1.cross(edge2).normalized();

        if (abs(original_normal.dot(new_normal)) < normal_flip_threshold) {
            std::cout<<"-----------------------------"<<std::endl;
            std::cout<<original_normal.transpose()<< std::endl;
            std::cout<<new_normal.transpose()<< std::endl;
            std::cout<<original_normal.dot(new_normal)<< std::endl;
            return true;
        }
    }

    return false;
}

bool RobustMeshSimplification::wouldCreateInvalidTopology(int v1, int v2) const {
    // 获取共同邻居
    std::set<int> neighbors_v1, neighbors_v2;

    for (int e_id : vertices[v1].edges) {
        if (edges[e_id].valid) {
            int other = edges[e_id].getOtherVertex(v1);
            if (other != v2) {
                neighbors_v1.insert(other);
            }
        }
    }

    for (int e_id : vertices[v2].edges) {
        if (edges[e_id].valid) {
            int other = edges[e_id].getOtherVertex(v2);
            if (other != v1) {
                neighbors_v2.insert(other);
            }
        }
    }

    // 找到共同邻居
    std::set<int> common_neighbors;
    std::set_intersection(neighbors_v1.begin(), neighbors_v1.end(),
                         neighbors_v2.begin(), neighbors_v2.end(),
                         std::inserter(common_neighbors, common_neighbors.begin()));

    // 如果没有共同邻居，不能折叠（会产生孤立顶点）
    if (common_neighbors.empty() && !vertices[v1].faces.empty() && !vertices[v2].faces.empty()) {
        return true;
    }

    // 检查是否会创建重复边
    // 对于每个v1的邻居（非v2且非共同邻居），检查v2是否也与它相连
    for (int n1 : neighbors_v1) {
        if (common_neighbors.count(n1)) continue;

        // 如果v2也与n1相连，折叠后会创建重复边
        if (neighbors_v2.count(n1)) {
            // 这在非流形情况下是允许的，所以不返回true
            // 只是标记一下，后续处理时需要注意
        }
    }

    return false;
}

double RobustMeshSimplification::evaluateQEMCost(
    int v1,
    int v2,
    Eigen::Vector3d& optimal_pos) const {

    const Vertex& vertex1 = vertices[v1];
    const Vertex& vertex2 = vertices[v2];

    // 合并Q矩阵
    Eigen::Matrix4d Q_sum = vertex1.Q + vertex2.Q;

    // 构造求解系统
    Eigen::Matrix3d A = Q_sum.block<3, 3>(0, 0);
    Eigen::Vector3d b = -Q_sum.block<3, 1>(0, 3);

    // 使用更稳定的求解方法
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(A);

    if (qr.rank() == 3) {
        // 矩阵满秩，使用QR分解求解
        optimal_pos = qr.solve(b);
    } else {
        // 矩阵秩不足，使用中点
        optimal_pos = 0.5 * (vertex1.position + vertex2.position);
    }

    // 计算代价
    Eigen::Vector4d pos_homogeneous(optimal_pos.x(), optimal_pos.y(), optimal_pos.z(), 1.0);
    double cost = pos_homogeneous.transpose() * Q_sum * pos_homogeneous;

    return std::max(0.0, cost);  // 确保非负
}

// ========== 虚函数的默认实现 ==========

double RobustMeshSimplification::calculateEdgeCollapseEnergy(int edge_id) {
    const Edge& edge = edges[edge_id];

    // 使用QEM计算基础代价
    Eigen::Vector3d optimal_pos;
    double qem_cost = evaluateQEMCost(edge.v1, edge.v2, optimal_pos);

    // 对非流形边和边界边添加惩罚因子（而非禁止）
    if (edge.is_non_manifold) {
        qem_cost *= 100.0;  // 非流形边：高惩罚
    } else if (edge.is_boundary) {
        qem_cost *= 10.0;   // 边界边：中等惩罚
    }

    return qem_cost;
}

Eigen::Vector3d RobustMeshSimplification::getOptimalVertexPosition(int edge_id) const {
    const Edge& edge = edges[edge_id];
    Eigen::Vector3d optimal_pos;
    evaluateQEMCost(edge.v1, edge.v2, optimal_pos);
    return optimal_pos;
}

bool RobustMeshSimplification::isBoundaryEdge(int edge_id) const {
    return edges[edge_id].is_boundary;
}

bool RobustMeshSimplification::isNonManifoldEdge(int edge_id) const {
    return edges[edge_id].is_non_manifold;
}

void RobustMeshSimplification::onEdgeCollapseCompleted(int, int) {
    // 默认实现为空，用户可重载
}

// ========== 核心功能实现 ==========

bool RobustMeshSimplification::collapseEdge(int edge_id) {
    if (edge_id >= edges.size() || !edges[edge_id].valid) {
        return false;
    }

    Edge& edge = edges[edge_id];

    // // 使用新的安全性检查（而非硬性的拓扑限制）
    // if (!canSafelyCollapse(edge_id)) {
    //     return false;
    // }

    int v1 = edge.v1;
    int v2 = edge.v2;

    // 收集受影响的边（折叠前）
    std::set<int> affected_edges = collectAffectedEdges(v1, v2);

    // 获取最优位置
    Eigen::Vector3d optimal_pos = getOptimalVertexPosition(edge_id);

    // 执行边折叠（v1被移除，v2被保留）
    if (!performEdgeCollapse(v1, v2, optimal_pos)) {
        return false;
    }

    // 回调
    this->onEdgeCollapseCompleted(v1, v2);

    // 局部更新受影响的边
    updateAffectedEdgesInQueue(affected_edges);

    return true;
}

bool RobustMeshSimplification::performEdgeCollapse(
    int v_remove,
    int v_keep,
    const Eigen::Vector3d& new_pos) {

    if (!vertices[v_remove].valid || !vertices[v_keep].valid) {
        return false;
    }

    // 找到要折叠的边
    auto edge_key = std::make_pair(std::min(v_remove, v_keep), std::max(v_remove, v_keep));
    auto edge_it = vertex_pair_to_edge.find(edge_key);
    if (edge_it == vertex_pair_to_edge.end()) {
        return false;
    }

    int collapsed_edge_id = edge_it->second;

    // ✅ 关键新增：删除v_keep和v_remove之间的邻接关系
    // 必须在开头做，因为它们要合并成一个点
    vertices[v_keep].neighbors.erase(v_remove);
    vertices[v_remove].neighbors.erase(v_keep);

    // 更新保留顶点的位置和Q矩阵
    vertices[v_keep].position = new_pos;
    vertices[v_keep].Q += vertices[v_remove].Q;

    // 收集需要删除的面（同时包含v_remove和v_keep的面）
    std::set<int> faces_to_delete;
    for (int f_id : vertices[v_remove].faces) {
        if (faces[f_id].valid && faces[f_id].hasVertex(v_keep)) {
            faces_to_delete.insert(f_id);
        }
    }

    // 删除退化面
    for (int f_id : faces_to_delete) {
        removeFace(f_id);
    }

    // 重新链接v_remove的边到v_keep
    std::vector<int> edges_to_relink(vertices[v_remove].edges.begin(),
                                      vertices[v_remove].edges.end());

    for (int e_id : edges_to_relink) {
        if (!edges[e_id].valid || e_id == collapsed_edge_id) continue;

        Edge& edge = edges[e_id];
        int other_vertex = edge.getOtherVertex(v_remove);

        if (other_vertex == v_keep) continue;

        // 断开 other <-> v_remove
        vertices[other_vertex].neighbors.erase(v_remove);
        vertices[v_remove].neighbors.erase(other_vertex); // 可选，v_remove马上无效


        // // 2. 建立 other <-> v_keep（这两行是你缺少的！）
        // vertices[other_vertex].neighbors.insert(v_keep);
        // vertices[v_keep].neighbors.insert(other_vertex);

        // 检查v_keep和other_vertex之间是否已有边
        auto new_edge_key = std::make_pair(std::min(v_keep, other_vertex),
                                           std::max(v_keep, other_vertex));
        auto existing_edge_it = vertex_pair_to_edge.find(new_edge_key);

        if (existing_edge_it != vertex_pair_to_edge.end()) {
            // 已存在边，合并面信息
            int existing_edge_id = existing_edge_it->second;
            Edge& existing_edge = edges[existing_edge_id];

            // 合并面集合
            for (int f_id : edge.faces) {
                if (faces[f_id].valid) {
                    existing_edge.faces.insert(f_id);

                    // 更新面的边引用
                    faces[f_id].edges.erase(e_id);
                    faces[f_id].edges.insert(existing_edge_id);
                }
            }

            // 删除当前边
            edges[e_id].valid = false;
            num_edges--;
            vertices[v_remove].edges.erase(e_id);

            // ==========================================
            // ✅【修复 1】: 必须从 other_vertex 中移除这条失效的边
            // ==========================================
            vertices[other_vertex].edges.erase(e_id);

            vertex_pair_to_edge.erase(std::make_pair(std::min(edge.v1, edge.v2),
                                                      std::max(edge.v1, edge.v2)));
        } else {
            // 更新边的顶点
            vertex_pair_to_edge.erase(std::make_pair(std::min(edge.v1, edge.v2),
                                                      std::max(edge.v1, edge.v2)));

            if (edge.v1 == v_remove) {
                edge.v1 = v_keep;
            } else {
                edge.v2 = v_keep;
            }


            vertex_pair_to_edge[new_edge_key] = e_id;
            vertices[v_keep].edges.insert(e_id);
            vertices[v_remove].edges.erase(e_id);

            // ==========================================
            // ✅【修复】: 显式添加邻接关系
            // ==========================================
            vertices[v_keep].neighbors.insert(other_vertex);
            vertices[other_vertex].neighbors.insert(v_keep);
        }
    }


    // 更新所有包含v_remove的面
    std::vector<int> faces_to_update(vertices[v_remove].faces.begin(),
                                      vertices[v_remove].faces.end());

    for (int f_id : faces_to_update) {
        if (!faces[f_id].valid) continue;

        Face& face = faces[f_id];

        // 更新面的顶点
        for (int& vid : face.vertices) {
            if (vid == v_remove) {
                vid = v_keep;
            }
        }

        // 检查是否有重复顶点（退化面）
        std::set<int> unique_vertices(face.vertices.begin(), face.vertices.end());
        if (unique_vertices.size() < 3) {
            // 面已退化，删除
            removeFace(f_id);
            continue;
        }

        vertices[v_keep].faces.insert(f_id);
        vertices[v_remove].faces.erase(f_id);

        // 重新计算面法向量
        face.normal = computeFaceNormal(f_id);
    }

    // 删除被折叠的边
    edges[collapsed_edge_id].valid = false;
    num_edges--;
    vertices[v_remove].edges.erase(collapsed_edge_id);
    vertices[v_keep].edges.erase(collapsed_edge_id);
    vertex_pair_to_edge.erase(edge_key);

    // 标记被移除的顶点为无效
    vertices[v_remove].valid = false;
    num_vertices--;

    return true;
}

void RobustMeshSimplification::removeFace(int face_id) {
    if (!faces[face_id].valid) return;

    Face& face = faces[face_id];

    // 从顶点的面集合中移除
    for (int vid : face.vertices) {
        vertices[vid].faces.erase(face_id);
    }

    // 从边的面集合中移除
    for (int e_id : face.edges) {
        edges[e_id].faces.erase(face_id);
    }

    face.valid = false;
    num_faces--;
}

void RobustMeshSimplification::removeEdge(int edge_id) {
    if (!edges[edge_id].valid) return;

    Edge& edge = edges[edge_id];

    // 删除所有相邻的面
    std::vector<int> faces_to_remove(edge.faces.begin(), edge.faces.end());
    for (int f_id : faces_to_remove) {
        removeFace(f_id);
    }

    // 从顶点的边集合中移除
    vertices[edge.v1].edges.erase(edge_id);
    vertices[edge.v2].edges.erase(edge_id);

    // ✅ 新增：断开邻接关系
    vertices[edge.v1].neighbors.erase(edge.v2);
    vertices[edge.v2].neighbors.erase(edge.v1);

    // 从映射中移除
    auto key = std::make_pair(std::min(edge.v1, edge.v2), std::max(edge.v1, edge.v2));
    vertex_pair_to_edge.erase(key);

    edge.valid = false;
    num_edges--;
}

// ========== 优化后的受影响边收集 ==========

std::set<int> RobustMeshSimplification::collectAffectedEdges(int v1, int v2) {
    std::set<int> affected_edges;
    std::set<int> one_ring_vertices;

    // 收集v1和v2的所有相邻边（必定受影响）
    for (int e_id : vertices[v1].edges) {
        if (edges[e_id].valid) {
            affected_edges.insert(e_id);
            int neighbor = edges[e_id].getOtherVertex(v1);
            one_ring_vertices.insert(neighbor);
        }
    }

    for (int e_id : vertices[v2].edges) {
        if (edges[e_id].valid) {
            affected_edges.insert(e_id);
            int neighbor = edges[e_id].getOtherVertex(v2);
            one_ring_vertices.insert(neighbor);
        }
    }

    // 收集一环内顶点之间的边（两端都要在一环内）
    for (int va : one_ring_vertices) {
        if (!vertices[va].valid) continue;

        for (int e_id : vertices[va].edges) {
            if (!edges[e_id].valid) continue;

            int vb = edges[e_id].getOtherVertex(va);

            // 只有当边的两个端点都在一环内时才受影响
            if (one_ring_vertices.count(vb)) {
                affected_edges.insert(e_id);
            }
        }
    }

    return affected_edges;
}

std::set<int> RobustMeshSimplification::updateAffectedElements(int new_vertex_id) {
    std::set<int> affected_edges;
    std::set<int> affected_vertices;

    if (!vertices[new_vertex_id].valid) {
        return affected_edges;
    }

    affected_vertices.insert(new_vertex_id);

    // 收集新顶点的所有相邻边
    for (int e_id : vertices[new_vertex_id].edges) {
        if (edges[e_id].valid) {
            affected_edges.insert(e_id);

            int other_vertex = edges[e_id].getOtherVertex(new_vertex_id);
            affected_vertices.insert(other_vertex);

            // 收集其他顶点的相邻边
            for (int other_e_id : vertices[other_vertex].edges) {
                if (edges[other_e_id].valid) {
                    affected_edges.insert(other_e_id);
                }
            }
        }
    }

    // 重新分类受影响的边
    for (int e_id : affected_edges) {
        updateEdgeClassification(e_id);
    }

    // 重新分类受影响的顶点
    for (int v_id : affected_vertices) {
        if (!vertices[v_id].valid) continue;

        Vertex& vertex = vertices[v_id];
        vertex.is_boundary = false;
        vertex.is_non_manifold = false;

        for (int e_id : vertex.edges) {
            if (edges[e_id].valid) {
                if (edges[e_id].is_boundary) {
                    vertex.is_boundary = true;
                }
                if (edges[e_id].is_non_manifold) {
                    vertex.is_non_manifold = true;
                }
            }
        }
    }

    return affected_edges;
}

// ========== 边队列管理 ==========

void RobustMeshSimplification::updateEdgeQueue() {
    // 清空队列
    edge_queue = std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>>();

    // 重置全局时间戳
    global_timestamp = 0;

    // 重新计算所有有效边的代价并加入队列
    for (int e_id = 0; e_id < edges.size(); ++e_id) {
        if (!edges[e_id].valid) continue;

        Edge& edge = edges[e_id];
        edge.collapse_cost = calculateEdgeCollapseEnergy(e_id);
        edge.optimal_position = getOptimalVertexPosition(e_id);
        edge.timestamp = global_timestamp;
        edge_queue.push(edge);
    }

    std::cout << "Initialized edge queue with " << edge_queue.size() << " edges" << std::endl;
}

void RobustMeshSimplification::updateAffectedEdgesInQueue(
    const std::set<int>& affected_edge_ids) {

    global_timestamp++;
    std::set<int> affected_vertices;

    // 第一遍：更新所有边的分类
    for (int e_id : affected_edge_ids) {
        if (e_id >= edges.size() || !edges[e_id].valid) continue;
        Edge& edge = edges[e_id];
        updateEdgeClassification(e_id);
        affected_vertices.insert(edge.v1);
        affected_vertices.insert(edge.v2);
    }

    // 第二遍：更新所有受影响顶点的标志（🔴 这是新增的关键修复）
    for (int v_id : affected_vertices) {
        if (v_id >= vertices.size() || !vertices[v_id].valid) continue;
        Vertex& vertex = vertices[v_id];
        vertex.is_boundary = false;
        vertex.is_non_manifold = false;
        for (int e_id : vertex.edges) {
            if (e_id >= edges.size() || !edges[e_id].valid) continue;
            if (edges[e_id].is_boundary) vertex.is_boundary = true;
            if (edges[e_id].is_non_manifold) vertex.is_non_manifold = true;
        }
    }

    // 第三遍：重新计算边的代价并加入队列
    for (int e_id : affected_edge_ids) {
        if (e_id >= edges.size() || !edges[e_id].valid) continue;
        Edge& edge = edges[e_id];
        edge.collapse_cost = calculateEdgeCollapseEnergy(e_id);
        edge.optimal_position = getOptimalVertexPosition(e_id);
        edge.timestamp = global_timestamp;
        edge_queue.push(edge);
    }
}
bool RobustMeshSimplification::isEdgeValidInQueue(const Edge& edge) const {
    if (edge.id >= edges.size() || !edges[edge.id].valid) {
        return false;
    }

    // 检查时间戳
    // if (edge.timestamp < global_timestamp) {
    //     return false;
    // }
    if (edge.timestamp != edges[edge.id].timestamp) {
        // 如果它们不匹配，意味着在主'edges'数组中的边已经被更新
        // (在 updateAffectedEdgesInQueue 中)，并且一个更新的版本（具有新代价）
        // 已经被推入队列。因此，这个'edge'副本是过时的，应该丢弃。
        return false;
    }

    return true;
}

bool RobustMeshSimplification::getNextValidEdge(Edge& out_edge) {
    while (!edge_queue.empty()) {
        Edge candidate = edge_queue.top();
        edge_queue.pop();

        if (isEdgeValidInQueue(candidate)) {
        // if (isEdgeValidInQueue(candidate)&&isTopologyPreserving(candidate.id)) {
            out_edge = candidate;
            return true;
        }
    }

    return false;
}

// ========== 主简化函数 ==========

int RobustMeshSimplification::simplifyMesh(int target_face_count) {
    // 初始化时进行一次全局更新
    updateEdgeQueue();

    int collapse_attempts = 0;
    int collapse_failures = 0;

    while (num_faces > target_face_count) {
        Edge min_edge;

        if (!getNextValidEdge(min_edge)) {
            std::cout << "No more valid edges to collapse" << std::endl;
            break;
        }

        collapse_attempts++;

        std::cout << "Faces: " << num_faces << " / " << target_face_count
                  << ", Collapsing edge " << min_edge.id
                  << " (cost: " << min_edge.collapse_cost
                  << ", boundary: " << min_edge.is_boundary
                  << ", non-manifold: " << min_edge.is_non_manifold << ")" << std::endl;

        if (!collapseEdge(min_edge.id)) {
            collapse_failures++;
            if (collapse_failures > 100) {
                std::cout << "Too many consecutive failures, stopping" << std::endl;
                break;
            }
        } else {
            collapse_failures = 0;  // 重置失败计数
        }
    }

    std::cout << "Simplification completed: " << collapse_attempts << " attempts, "
              << (collapse_attempts - collapse_failures) << " successful" << std::endl;

    return num_faces;
}

int RobustMeshSimplification::simplifyMeshByEdgeCount(int collapse_count) {
    int collapsed = 0;

    updateEdgeQueue();

    while (collapsed < collapse_count) {
        Edge min_edge;

        if (!getNextValidEdge(min_edge)) {
            std::cout << "No more valid edges to collapse" << std::endl;
            break;
        }

        if (collapseEdge(min_edge.id)) {
            collapsed++;
            std::cout << "Collapsed " << collapsed << " / " << collapse_count
                      << " edges, remaining faces: " << num_faces << std::endl;
        }
    }

    return collapsed;
}

// ========== 几何计算 ==========

Eigen::Vector3d RobustMeshSimplification::computeFaceNormal(int face_id) const {
    const Face& face = faces[face_id];

    Eigen::Vector3d v0 = vertices[face.vertices[0]].position;
    Eigen::Vector3d v1 = vertices[face.vertices[1]].position;
    Eigen::Vector3d v2 = vertices[face.vertices[2]].position;

    Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0);
    double norm = normal.norm();

    if (norm < 1e-10) {
        return Eigen::Vector3d(0, 0, 1);  // 退化面返回默认法向量
    }

    return normal / norm;
}

double RobustMeshSimplification::computeFaceArea(int face_id) const {
    const Face& face = faces[face_id];

    Eigen::Vector3d v0 = vertices[face.vertices[0]].position;
    Eigen::Vector3d v1 = vertices[face.vertices[1]].position;
    Eigen::Vector3d v2 = vertices[face.vertices[2]].position;

    return 0.5 * (v1 - v0).cross(v2 - v0).norm();
}

// ========== 查询和输出函数 ==========

std::map<int, int> RobustMeshSimplification::getCurrentMesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out, std::map<int, int>& new_to_old_vertex_id) {
    // 收集有效顶点
    std::vector<int> valid_vertices;
    std::map<int, int> old_to_new_vertex_id;

    for (int i = 0; i < vertices.size(); ++i) {
        if (vertices[i].valid) {
            old_to_new_vertex_id[i] = valid_vertices.size();
            new_to_old_vertex_id[valid_vertices.size()]=i;
            valid_vertices.push_back(i);
        }
    }

    // 构建顶点矩阵
    V_out.resize(valid_vertices.size(), 3);
    for (int i = 0; i < valid_vertices.size(); ++i) {
        V_out.row(i) = vertices[valid_vertices[i]].position.transpose();
    }

    // 收集有效面
    std::vector<std::vector<int>> valid_faces;
    for (const Face& face : faces) {
        if (face.valid) {
            std::vector<int> new_face;
            for (int old_vid : face.vertices) {
                new_face.push_back(old_to_new_vertex_id[old_vid]);
            }
            valid_faces.push_back(new_face);
        }
    }

    // 构建面矩阵
    F_out.resize(valid_faces.size(), 3);
    for (int i = 0; i < valid_faces.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            F_out(i, j) = valid_faces[i][j];
        }
    }
    return old_to_new_vertex_id;
}

void RobustMeshSimplification::getCurrentMesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out) {
    // 收集有效顶点
    std::vector<int> valid_vertices;
    std::map<int, int> old_to_new_vertex_id;

    for (int i = 0; i < vertices.size(); ++i) {
        if (vertices[i].valid) {
            old_to_new_vertex_id[i] = valid_vertices.size();
            valid_vertices.push_back(i);
        }
    }

    // 构建顶点矩阵
    V_out.resize(valid_vertices.size(), 3);
    for (int i = 0; i < valid_vertices.size(); ++i) {
        V_out.row(i) = vertices[valid_vertices[i]].position.transpose();
    }

    // 收集有效面
    std::vector<std::vector<int>> valid_faces;
    for (const Face& face : faces) {
        if (face.valid) {
            std::vector<int> new_face;
            for (int old_vid : face.vertices) {
                new_face.push_back(old_to_new_vertex_id[old_vid]);
            }
            valid_faces.push_back(new_face);
        }
    }

    // 构建面矩阵
    F_out.resize(valid_faces.size(), 3);
    for (int i = 0; i < valid_faces.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            F_out(i, j) = valid_faces[i][j];
        }
    }
}

void RobustMeshSimplification::exportMesh(const std::string& filepath) {
    Eigen::MatrixXd V_out;
    Eigen::MatrixXi F_out;
    std::map<int, int> new_to_old_vertex_id;
    auto old_to_new_vertex_id = getCurrentMesh(V_out, F_out,new_to_old_vertex_id);

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return;
    }

    // 写入顶点
    for (int i = 0; i < V_out.rows(); ++i) {
        file << "v " << V_out(i, 0) << " " << V_out(i, 1) << " " << V_out(i, 2) << std::endl;
    }

    // 写入面
    for (int i = 0; i < F_out.rows(); ++i) {
        file << "f " << (F_out(i, 0) + 1) << " " << (F_out(i, 1) + 1) << " " << (F_out(i, 2) + 1) << std::endl;
    }

    // 3. ✅ 新增：写入线 (Wire Edges)
    for (const auto& edge : edges) {
        // 只有当边有效，且被标记为 wire (没有面) 时才导出
        if (edge.valid) {

            // 查找旧 ID 对应的新 ID
            // 必须使用 find，因为理论上 valid 的边连接的一定是 valid 的点，但为了安全
            auto it1 = old_to_new_vertex_id.find(edge.v1);
            auto it2 = old_to_new_vertex_id.find(edge.v2);
            if (it1 != old_to_new_vertex_id.end() && it2 != old_to_new_vertex_id.end()) {
                // 写入线：l v1 v2 (OBJ索引从1开始，所以 +1)
                file << "l " << (it1->second + 1) << " " << (it2->second + 1) << std::endl;
            }
        }
    }



    file.close();
    std::cout << "Mesh exported to " << filepath << std::endl;



}

void RobustMeshSimplification::printStatistics() {
    int boundary_edges = 0, non_manifold_edges = 0, regular_edges = 0;
    int boundary_vertices = 0, non_manifold_vertices = 0;

    for (const Edge& edge : edges) {
        if (!edge.valid) continue;
        if (edge.is_non_manifold) non_manifold_edges++;
        else if (edge.is_boundary) boundary_edges++;
        else regular_edges++;
    }

    for (const Vertex& vertex : vertices) {
        if (!vertex.valid) continue;
        if (vertex.is_non_manifold) non_manifold_vertices++;
        else if (vertex.is_boundary) boundary_vertices++;
    }

    std::cout << "=== Robust Mesh Statistics ===" << std::endl;
    std::cout << "Vertices: " << num_vertices << std::endl;
    std::cout << "  - Boundary: " << boundary_vertices << std::endl;
    std::cout << "  - Non-manifold: " << non_manifold_vertices << std::endl;
    std::cout << "Edges: " << num_edges << std::endl;
    std::cout << "  - Regular: " << regular_edges << std::endl;
    std::cout << "  - Boundary: " << boundary_edges << std::endl;
    std::cout << "  - Non-manifold: " << non_manifold_edges << std::endl;
    std::cout << "Faces: " << num_faces << std::endl;
    std::cout << "Queue size: " << edge_queue.size() << std::endl;
    std::cout << "Global timestamp: " << global_timestamp << std::endl;
}

void RobustMeshSimplification::analyzeMeshTopology() {
    printStatistics();

    std::cout << "\n=== Detailed Topology Analysis ===" << std::endl;

    std::map<int, int> edge_valence_histogram;
    for (const Edge& edge : edges) {
        if (!edge.valid) continue;

        int face_count = 0;
        for (int f_id : edge.faces) {
            if (faces[f_id].valid) face_count++;
        }
        edge_valence_histogram[face_count]++;
    }

    std::cout << "Edge valence histogram:" << std::endl;
    for (const auto& pair : edge_valence_histogram) {
        std::cout << "  " << pair.second << " edges with " << pair.first << " adjacent faces" << std::endl;
    }
}

const RobustMeshSimplification::Edge& RobustMeshSimplification::getEdge(int edge_id) const {
    if (edge_id >= static_cast<int>(edges.size())) {
        throw std::out_of_range("Edge ID out of range");
    }
    return edges[edge_id];
}

std::vector<int> RobustMeshSimplification::getValidEdgeIds() const {
    std::vector<int> valid_ids;
    for (int i = 0; i < edges.size(); ++i) {
        if (edges[i].valid) {
            valid_ids.push_back(i);
        }
    }
    return valid_ids;
}

bool RobustMeshSimplification::validateMeshTopology() const {
    for (int i = 0; i < edges.size(); ++i) {
        if (!edges[i].valid) continue;

        const Edge& edge = edges[i];
        if (!vertices[edge.v1].valid || !vertices[edge.v2].valid) {
            std::cerr << "Error: Edge " << i << " references invalid vertices" << std::endl;
            return false;
        }
    }

    return true;
}






// cpp中实现
bool RobustMeshSimplification::validateNeighborsConsistency() const {
    for (int v_id = 0; v_id < vertices.size(); ++v_id) {
        if (!vertices[v_id].valid) continue;

        // 通过edges计算应该有的neighbors
        std::set<int> expected_neighbors;
        for (int e_id : vertices[v_id].edges) {
            if (edges[e_id].valid) {
                expected_neighbors.insert(edges[e_id].getOtherVertex(v_id));
            }
        }

        // 比对
        if (vertices[v_id].neighbors != expected_neighbors) {
            std::cerr << "Vertex " << v_id << " neighbors inconsistent!" << std::endl;
            return false;
        }
    }
    return true;
}
