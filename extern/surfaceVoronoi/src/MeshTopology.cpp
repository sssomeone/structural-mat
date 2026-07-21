#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>

namespace MeshTopology {

/**
 * 边结构体，用于表示两个顶点之间的边
 */
struct MeshEdge {
    int v1, v2;

    MeshEdge(int a, int b) {
        // 确保v1 <= v2，保证边的唯一性
        if (a <= b) {
            v1 = a; v2 = b;
        } else {
            v1 = b; v2 = a;
        }
    }

    bool operator==(const MeshEdge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }
};

/**
 * MeshEdge的哈希函数，用于unordered_map
 */
struct MeshEdgeHash {
    std::size_t operator()(const MeshEdge& edge) const {
        // 使用Cantor配对函数生成哈希值
        return std::hash<long long>()((long long)edge.v1 * 1000000007LL + edge.v2);
    }
};

/**
 * 面片邻接信息
 */
struct AdjacencyInfo {
    int faceId;
    MeshEdge sharedEdge;  // 共享的边

    AdjacencyInfo(int id, const MeshEdge& edge) : faceId(id), sharedEdge(edge) {}
};

/**
 * 网格拓扑类型枚举
 */
enum class TopologyType {
    CLOSED_MANIFOLD,      // 封闭流形
    MANIFOLD_WITH_BOUNDARY, // 有边界的流形
    NON_MANIFOLD          // 非流形
};

/**
 * 网格统计信息结构体
 */
struct TopologyStatistics {
    int numVertices;
    int numFaces;
    int numEdges;
    int numBoundaryEdges;
    int numNonManifoldEdges;
    int eulerCharacteristic;  // 欧拉特征数 V - E + F
    TopologyType meshType;

    void print() const {
        std::cout << "=== 网格统计信息 ===" << std::endl;
        std::cout << "顶点数: " << numVertices << std::endl;
        std::cout << "面片数: " << numFaces << std::endl;
        std::cout << "边数: " << numEdges << std::endl;
        std::cout << "边界边数: " << numBoundaryEdges << std::endl;
        std::cout << "非流形边数: " << numNonManifoldEdges << std::endl;
        std::cout << "欧拉特征数 (V-E+F): " << eulerCharacteristic << std::endl;

        std::string typeStr;
        switch (meshType) {
            case TopologyType::CLOSED_MANIFOLD:
                typeStr = "封闭流形";
                break;
            case TopologyType::MANIFOLD_WITH_BOUNDARY:
                typeStr = "有边界的流形";
                break;
            case TopologyType::NON_MANIFOLD:
                typeStr = "非流形";
                break;
        }
        std::cout << "网格类型: " << typeStr << std::endl;
    }
};

/**
 * 网格面片邻接关系处理器
 */
class FaceAdjacencyProcessor {
private:
    Eigen::MatrixXd vertices_;        // 顶点坐标 (V x 3)
    Eigen::MatrixXi faces_;           // 面片索引 (F x 3)

    // 边到面片的映射：每条边可能被多个面片共享（非流形情况）
    std::unordered_map<MeshEdge, std::vector<int>, MeshEdgeHash> edgeToFaces_;

    // 每个面片的邻接面片列表
    std::vector<std::vector<AdjacencyInfo>> faceAdjacency_;

    // 统计信息
    TopologyStatistics statistics_;

public:
    /**
     * 构造函数：构建邻接关系
     * @param V 顶点矩阵 (numVertices x 3)
     * @param F 面片矩阵 (numFaces x 3)
     */
    explicit FaceAdjacencyProcessor(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
        : vertices_(V), faces_(F) {

        if (faces_.cols() != 3) {
            throw std::invalid_argument("Face matrix must have 3 columns (triangular faces only)");
        }

        if (vertices_.cols() != 3) {
            throw std::invalid_argument("Vertex matrix must have 3 columns (3D coordinates)");
        }

        buildAdjacency();
        computeStatistics();
    }

private:
    /**
     * 构建邻接关系的主要函数
     */
    void buildAdjacency() {
        // 1. 构建边到面片的映射
        buildEdgeToFaceMapping();

        // 2. 构建面片邻接列表
        buildFaceAdjacencyList();
    }

    /**
     * 构建边到面片的映射
     */
    void buildEdgeToFaceMapping() {
        edgeToFaces_.clear();

        const int numFaces = faces_.rows();
        const int numVertices = vertices_.rows();

        for (int faceId = 0; faceId < numFaces; ++faceId) {
            // 获取面片的三个顶点
            int v0 = faces_(faceId, 0);
            int v1 = faces_(faceId, 1);
            int v2 = faces_(faceId, 2);

            // 检查顶点索引的有效性
            if (v0 < 0 || v0 >= numVertices ||
                v1 < 0 || v1 >= numVertices ||
                v2 < 0 || v2 >= numVertices) {
                throw std::out_of_range("Face " + std::to_string(faceId) +
                                       " contains invalid vertex index");
            }

            // 添加三条边
            MeshEdge edge1(v0, v1);
            MeshEdge edge2(v1, v2);
            MeshEdge edge3(v2, v0);

            edgeToFaces_[edge1].push_back(faceId);
            edgeToFaces_[edge2].push_back(faceId);
            edgeToFaces_[edge3].push_back(faceId);
        }
    }

    /**
     * 构建面片邻接列表
     */
    void buildFaceAdjacencyList() {
        const int numFaces = faces_.rows();
        faceAdjacency_.resize(numFaces);

        for (int faceId = 0; faceId < numFaces; ++faceId) {
            faceAdjacency_[faceId].clear();

            // 获取面片的三个顶点
            int v0 = faces_(faceId, 0);
            int v1 = faces_(faceId, 1);
            int v2 = faces_(faceId, 2);

            // 检查三条边的邻接面片
            std::vector<MeshEdge> edges = {MeshEdge(v0, v1), MeshEdge(v1, v2), MeshEdge(v2, v0)};

            std::unordered_set<int> adjacentFaces;  // 用于去重

            for (const MeshEdge& edge : edges) {
                auto it = edgeToFaces_.find(edge);
                if (it != edgeToFaces_.end()) {
                    for (int adjFaceId : it->second) {
                        if (adjFaceId != faceId && adjacentFaces.find(adjFaceId) == adjacentFaces.end()) {
                            faceAdjacency_[faceId].emplace_back(adjFaceId, edge);
                            adjacentFaces.insert(adjFaceId);
                        }
                    }
                }
            }
        }
    }

    /**
     * 计算网格统计信息
     */
    void computeStatistics() {
        statistics_.numVertices = vertices_.rows();
        statistics_.numFaces = faces_.rows();
        statistics_.numEdges = edgeToFaces_.size();
        statistics_.numBoundaryEdges = 0;
        statistics_.numNonManifoldEdges = 0;

        for (const auto& pair : edgeToFaces_) {
            int faceCount = pair.second.size();
            if (faceCount == 1) {
                statistics_.numBoundaryEdges++;
            } else if (faceCount > 2) {
                statistics_.numNonManifoldEdges++;
            }
        }

        // 计算欧拉特征数
        statistics_.eulerCharacteristic = statistics_.numVertices - statistics_.numEdges + statistics_.numFaces;

        // 确定网格类型
        if (statistics_.numNonManifoldEdges == 0 && statistics_.numBoundaryEdges == 0) {
            statistics_.meshType = TopologyType::CLOSED_MANIFOLD;
        } else if (statistics_.numNonManifoldEdges == 0) {
            statistics_.meshType = TopologyType::MANIFOLD_WITH_BOUNDARY;
        } else {
            statistics_.meshType = TopologyType::NON_MANIFOLD;
        }
    }

public:
    /**
     * 获取指定面片的所有邻接面片
     * @param faceId 面片ID
     * @return 邻接面片信息的vector
     */
    std::vector<AdjacencyInfo> getAdjacentFaces(int faceId) const {
        validateFaceId(faceId);
        return faceAdjacency_[faceId];
    }

    /**
     * 获取指定面片的邻接面片ID列表（简化版本）
     * @param faceId 面片ID
     * @return 邻接面片ID的vector
     */
    std::vector<int> getAdjacentFaceIds(int faceId) const {
        validateFaceId(faceId);

        std::vector<int> result;
        result.reserve(faceAdjacency_[faceId].size());

        for (const auto& info : faceAdjacency_[faceId]) {
            result.push_back(info.faceId);
        }
        return result;
    }

    /**
     * 检查两个面片是否相邻
     * @param faceId1 第一个面片ID
     * @param faceId2 第二个面片ID
     * @param sharedEdge 如果相邻，返回共享边（可选）
     * @return 如果相邻返回true
     */
    bool areAdjacent(int faceId1, int faceId2, MeshEdge* sharedEdge = nullptr) const {
        if (!isValidFaceId(faceId1) || !isValidFaceId(faceId2)) {
            return false;
        }

        for (const auto& info : faceAdjacency_[faceId1]) {
            if (info.faceId == faceId2) {
                if (sharedEdge) {
                    *sharedEdge = info.sharedEdge;
                }
                return true;
            }
        }
        return false;
    }

    /**
     * 获取指定边的所有相邻面片
     * @param v1 边的第一个顶点
     * @param v2 边的第二个顶点
     * @return 包含该边的所有面片ID
     */
    std::vector<int> getFacesOnEdge(int v1, int v2) const {
        MeshEdge edge(v1, v2);
        auto it = edgeToFaces_.find(edge);
        if (it != edgeToFaces_.end()) {
            return it->second;
        }
        return {};
    }

    /**
     * 判断边是否为边界边
     * @param v1 边的第一个顶点
     * @param v2 边的第二个顶点
     * @return 如果是边界边返回true
     */
    bool isBoundaryEdge(int v1, int v2) const {
        std::vector<int> faces = getFacesOnEdge(v1, v2);
        return faces.size() == 1;
    }

    /**
     * 判断边是否为非流形边
     * @param v1 边的第一个顶点
     * @param v2 边的第二个顶点
     * @return 如果是非流形边返回true
     */
    bool isNonManifoldEdge(int v1, int v2) const {
        std::vector<int> faces = getFacesOnEdge(v1, v2);
        return faces.size() > 2;
    }

    /**
     * 获取面片的顶点坐标
     * @param faceId 面片ID
     * @return 三个顶点的坐标
     */
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> getFaceVertices(int faceId) const {
        validateFaceId(faceId);

        int v0 = faces_(faceId, 0);
        int v1 = faces_(faceId, 1);
        int v2 = faces_(faceId, 2);

        Eigen::Vector3d p0 = vertices_.row(v0);
        Eigen::Vector3d p1 = vertices_.row(v1);
        Eigen::Vector3d p2 = vertices_.row(v2);

        return std::make_tuple(p0, p1, p2);
    }

    /**
     * 计算面片的法向量
     * @param faceId 面片ID
     * @return 单位法向量
     */
    Eigen::Vector3d getFaceNormal(int faceId) const {
        auto [p0, p1, p2] = getFaceVertices(faceId);

        Eigen::Vector3d v1 = p1 - p0;
        Eigen::Vector3d v2 = p2 - p0;
        Eigen::Vector3d normal = v1.cross(v2);

        double norm = normal.norm();
        if (norm > 1e-12) {
            return normal / norm;
        } else {
            return Eigen::Vector3d::Zero();  // 退化三角形
        }
    }

    /**
     * 计算面片的面积
     * @param faceId 面片ID
     * @return 面片面积
     */
    double getFaceArea(int faceId) const {
        auto [p0, p1, p2] = getFaceVertices(faceId);

        Eigen::Vector3d v1 = p1 - p0;
        Eigen::Vector3d v2 = p2 - p0;

        return 0.5 * v1.cross(v2).norm();
    }

    /**
     * 获取面片的顶点索引
     * @param faceId 面片ID
     * @return 三个顶点的索引
     */
    std::tuple<int, int, int> getFaceVertexIndices(int faceId) const {
        validateFaceId(faceId);
        return std::make_tuple(faces_(faceId, 0), faces_(faceId, 1), faces_(faceId, 2));
    }

    // Getter函数
    const TopologyStatistics& getStatistics() const { return statistics_; }
    void printStatistics() const { statistics_.print(); }

    int getNumVertices() const { return statistics_.numVertices; }
    int getNumFaces() const { return statistics_.numFaces; }
    int getNumEdges() const { return statistics_.numEdges; }
    int getNumBoundaryEdges() const { return statistics_.numBoundaryEdges; }
    int getNumNonManifoldEdges() const { return statistics_.numNonManifoldEdges; }
    TopologyType getMeshType() const { return statistics_.meshType; }

    const Eigen::MatrixXd& getVertices() const { return vertices_; }
    const Eigen::MatrixXi& getFaces() const { return faces_; }

private:
    /**
     * 验证面片ID的有效性
     */
    void validateFaceId(int faceId) const {
        if (!isValidFaceId(faceId)) {
            throw std::out_of_range("Face ID out of range: " + std::to_string(faceId));
        }
    }
    /**
     * 检查面片ID是否有效
     */
    bool isValidFaceId(int faceId) const {
        return faceId >= 0 && faceId < statistics_.numFaces;
    }
};

} // namespace MeshTopology
