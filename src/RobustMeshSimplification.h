//
// Rewritten to support boundary and non-manifold edge collapse
// 重写以支持边界边和非流形边的折叠
//

#ifndef SKELETON_SIMPLIFICATION_ROBUSTMESHSIMPLIFICATION_H
#define SKELETON_SIMPLIFICATION_ROBUSTMESHSIMPLIFICATION_H

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <fstream>
#include <iostream>

class RobustMeshSimplification {
public:
    // 顶点结构
    struct Vertex {
        Eigen::Vector3d position;
        Eigen::Matrix4d Q;  // QEM矩阵

        std::set<int> neighbors; // ✅ 新增：相邻顶点
        std::set<int> edges;  // 相邻边
        std::set<int> faces;  // 相邻面
        bool valid = true;
        int id = -1;

        // 边界和流形性质
        bool is_boundary = false;
        bool is_non_manifold = false;

        Vertex() : Q(Eigen::Matrix4d::Zero()) {}
        Vertex(const Eigen::Vector3d& pos) : position(pos), Q(Eigen::Matrix4d::Zero()) {}
    };

    // 边结构
    struct Edge {
        int v1, v2;  // 两个顶点
        std::set<int> faces;  // 相邻面
        double collapse_cost = 0.0;
        Eigen::Vector3d optimal_position;
        bool valid = true;
        int id = -1;

        // 时间戳用于优先队列管理
        mutable long long timestamp = 0;

        // 边界和流形性质
        bool is_boundary = false;
        bool is_non_manifold = false;
        bool is_wire = false;

        Edge() = default;
        Edge(int v1, int v2) : v1(v1), v2(v2) {}

        bool hasVertex(int vid) const { return v1 == vid || v2 == vid; }
        int getOtherVertex(int vid) const { return (vid == v1) ? v2 : v1; }

        // 用于优先队列
        bool operator>(const Edge& other) const {
            return collapse_cost > other.collapse_cost;
        }
    };

    // 面结构
    struct Face {
        std::vector<int> vertices;  // 顶点（通常3个）
        std::set<int> edges;  // 边
        Eigen::Vector3d normal;
        bool valid = true;
        int id = -1;

        Face() = default;
        Face(const std::vector<int>& verts) : vertices(verts) {}

        bool hasVertex(int vid) const {
            return std::find(vertices.begin(), vertices.end(), vid) != vertices.end();
        }

        bool hasEdge(int eid) const {
            return edges.find(eid) != edges.end();
        }
    };

public:
    // 核心数据
    Eigen::MatrixXd V_original;
    Eigen::MatrixXi F_original;

    // 网格数据结构
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Face> faces;

    // 边的映射
    std::map<std::pair<int,int>, int> vertex_pair_to_edge;

    // 优先队列
    std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> edge_queue;

    // 全局时间戳
    long long global_timestamp = 0;

    // 统计信息
    int num_vertices = 0;
    int num_edges = 0;
    int num_faces = 0;

    // 配置选项
    double min_face_area_threshold = 1e-10;  // 最小面积阈值
    double normal_flip_threshold = -0.2;     // 法向量翻转阈值（约102度）

public:
    // 构造函数
    RobustMeshSimplification(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);
    virtual ~RobustMeshSimplification() = default;


    // ========== 用户可重写的虚函数 ==========
    virtual double calculateEdgeCollapseEnergy(int edge_id);
    virtual Eigen::Vector3d getOptimalVertexPosition(int edge_id) const;
    virtual bool isBoundaryEdge(int edge_id) const;
    virtual bool isNonManifoldEdge(int edge_id) const;
    bool validateNeighborsConsistency() const;

    // 边折叠完成后的回调
    virtual void onEdgeCollapseCompleted(int v_removed, int v_kept);

    // ========== 核心功能函数 ==========
    bool collapseEdge(int edge_id);
    int simplifyMesh(int target_face_count);
    int simplifyMeshByEdgeCount(int collapse_count);

    // ========== 查询和输出函数 ==========
    std::map<int, int> getCurrentMesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out, std::map<int, int>& new_to_old_vertex_id);
    void getCurrentMesh(Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out);
    void exportMesh(const std::string& filepath);
    void printStatistics();
    void analyzeMeshTopology();

    int getNumVertices() const { return num_vertices; }
    int getNumEdges() const { return num_edges; }
    int getNumFaces() const { return num_faces; }
    bool isTopologyPreserving(int edge_id) const;

    const Edge& getEdge(int edge_id) const;
    std::vector<int> getValidEdgeIds() const;

public:
    // ========== 内部辅助函数 ==========
    void initializeMesh();
    void buildTopology();
    void computeVertexQuadrics();
    void updateEdgeQueue();
    void updateAffectedEdgesInQueue(const std::set<int>& affected_edge_ids);
    std::set<int> collectAffectedEdges(int v1, int v2);

    int findOrCreateEdge(int v1, int v2);
    void removeEdge(int edge_id);
    void removeFace(int face_id);

    // 标准边折叠算法
    bool performEdgeCollapse(int v_remove, int v_keep, const Eigen::Vector3d& new_pos);

    // 安全性检查
    bool canSafelyCollapse(int edge_id) const;
    bool wouldCreateDegenerateFace(int edge_id, const Eigen::Vector3d& new_pos) const;
    bool wouldCreateInvalidTopology(int v1, int v2) const;

    // QEM计算
    void computeFacePlaneMatrix(int face_id, Eigen::Matrix4d& K) const;
    double evaluateQEMCost(int v1, int v2, Eigen::Vector3d& optimal_pos) const;

    // 边界和流形性分析
    void classifyEdges();
    void classifyVertices();
    void updateEdgeClassification(int edge_id);
    std::set<int> updateAffectedElements(int new_vertex_id);

    // 几何计算
    Eigen::Vector3d computeFaceNormal(int face_id) const;
    double computeFaceArea(int face_id) const;

    // 优先队列管理
    bool isEdgeValidInQueue(const Edge& edge) const;
    bool getNextValidEdge(Edge& out_edge);

    // 验证
    bool validateMeshTopology() const;
};

#endif //SKELETON_SIMPLIFICATION_ROBUSTMESHSIMPLIFICATION_H