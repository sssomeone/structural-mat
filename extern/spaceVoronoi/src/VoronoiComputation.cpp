#include "VoronoiComputation.h"

namespace VoronoiComputation {

// FastMeshInside 类实现
FastMeshInside::FastMeshInside(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces)
    : V(vertices), F(faces) {

    // 构建fast winding number BVH
    igl::fast_winding_number(
        V.cast<float>(), F,
        /* expansion_order = */ 2,
        fwn_bvh
    );
}

bool FastMeshInside::isInside(const Point_3& point) const {
    return isInside(pointToEigen(point));
}

bool FastMeshInside::isInside(const Eigen::Vector3d& point) const {
    std::lock_guard<std::mutex> lock(query_mutex);

    // 单点查询需要一个 1×3 的浮点矩阵
    Eigen::Matrix<float, 1, 3> query_point;
    query_point.row(0) = point.cast<float>();

    // 计算 fast winding-number
    Eigen::VectorXf winding_numbers;
    igl::fast_winding_number(
        fwn_bvh,
        /* accuracy_scale = */ 2.0f,
        query_point,
        winding_numbers
    );

    // winding_number > 0.5 表示在内部
    return winding_numbers(0) > 0.5f;
}

std::vector<bool> FastMeshInside::isInside(const std::vector<Eigen::Vector3d>& points) const {
    std::lock_guard<std::mutex> lock(query_mutex);

    if (points.empty()) return {};

    // 构建查询矩阵
    Eigen::MatrixXf query_points(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
        query_points.row(i) = points[i].cast<float>();
    }

    // 批量计算 winding numbers
    Eigen::VectorXf winding_numbers;
    igl::fast_winding_number(
        fwn_bvh,
        /* accuracy_scale = */ 2.0f,
        query_points,
        winding_numbers
    );

    // 转换为bool结果
    std::vector<bool> results;
    results.reserve(points.size());
    for (int i = 0; i < winding_numbers.size(); ++i) {
        results.push_back(winding_numbers(i) > 0.5f);
    }

    return results;
}

std::vector<bool> FastMeshInside::isInside(const std::vector<Point_3>& points) const {
    std::vector<Eigen::Vector3d> eigen_points;
    eigen_points.reserve(points.size());
    for (const auto& p : points) {
        eigen_points.push_back(pointToEigen(p));
    }
    return isInside(eigen_points);
}

// 辅助函数实现

Point_3 eigenToPoint(const Eigen::Vector3d& v) {
    return Point_3(v.x(), v.y(), v.z());
}

Eigen::Vector3d pointToEigen(const Point_3& p) {
    return Eigen::Vector3d(CGAL::to_double(p.x()),
                          CGAL::to_double(p.y()),
                          CGAL::to_double(p.z()));
}

Point_3 computeCircumcenter(Cell_handle cell) {
    std::array<Point_3, 4> points;
    for (int i = 0; i < 4; ++i) {
        points[i] = cell->vertex(i)->point();
    }
    return CGAL::circumcenter(points[0], points[1], points[2], points[3]);
}

std::vector<std::array<int, 3>> triangulatePolygon(const std::vector<int>& polygon) {
    std::vector<std::array<int, 3>> triangles;
    if (polygon.size() < 3) return triangles;

    // 简单的扇形三角化
    for (size_t i = 1; i < polygon.size() - 1; ++i) {
        triangles.push_back({polygon[0], polygon[i], polygon[i + 1]});
    }
    return triangles;
}

// 主要计算函数
VoronoiResult computeVoronoi(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const std::vector<Eigen::Vector3d>& sites) {

    VoronoiResult result;

    if (sites.empty()) {
        return result;
    }

    // 1. 构建快速内外判断器
    FastMeshInside mesh_inside(V, F);

    // 2. 构建Delaunay三角剖分
    Delaunay dt;
    for (size_t i = 0; i < sites.size(); ++i) {
        Vertex_handle vh = dt.insert(eigenToPoint(sites[i]));
        vh->info() = static_cast<int>(i);
    }

    // 3. 收集所有候选Voronoi顶点（批量处理以提高效率）
    std::vector<Point_3> candidate_vertices;
    std::vector<Cell_handle> candidate_cells;

    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        Point_3 circumcenter = computeCircumcenter(cit);
        candidate_vertices.push_back(circumcenter);
        candidate_cells.push_back(cit);
    }

    // 4. 批量检查哪些顶点在模型内部
    std::vector<bool> inside_flags = mesh_inside.isInside(candidate_vertices);

    // 5. 收集有效的Voronoi顶点
    std::vector<Point_3> voronoi_vertices;
    std::vector<Eigen::Vector4i> vertex_sites_info;
    std::map<Point_3, int> vertex_to_index;

    for (size_t i = 0; i < candidate_vertices.size(); ++i) {
        if (inside_flags[i]) {
            // 获取对应的四个站点
            Eigen::Vector4i sites_vec;
            for (int j = 0; j < 4; ++j) {
                sites_vec[j] = candidate_cells[i]->vertex(j)->info();
            }

            vertex_to_index[candidate_vertices[i]] = static_cast<int>(voronoi_vertices.size());
            voronoi_vertices.push_back(candidate_vertices[i]);
            vertex_sites_info.push_back(sites_vec);
        }
    }

    // 6. 收集Voronoi面
    std::vector<std::vector<int>> voronoi_faces;
    std::vector<std::pair<int,int>> faces_sites;

    // 遍历Delaunay边，获取对应的Voronoi面
    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        // 获取边相邻的所有四面体
        std::vector<Cell_handle> incident_cells;

        // 使用Cell_circulator正确遍历edge周围的cells
        auto cc = dt.incident_cells(*eit);
        auto done = cc;
        do {
            incident_cells.push_back(cc);
            ++cc;
        } while (cc != done);

        std::vector<int> face_vertices;
        bool all_vertices_valid = true;

        for (auto cell : incident_cells) {
            if (dt.is_infinite(cell)) {
                all_vertices_valid = false;
                break;
            }

            Point_3 circumcenter = computeCircumcenter(cell);
            auto it = vertex_to_index.find(circumcenter);

            if (it != vertex_to_index.end()) {
                face_vertices.push_back(it->second);
            } else {
                all_vertices_valid = false;
                break;
            }
        }

        // 只保留所有顶点都在模型内的面
        if (all_vertices_valid && face_vertices.size() >= 3) {
            voronoi_faces.push_back(face_vertices);

            Cell_handle c = eit->first;
            int i = eit->second;
            int j = eit->third;

            // 2. 获取真正的站点 ID (假设之前存入了 info)
            int site_id_1 = c->vertex(i)->info();
            int site_id_2 = c->vertex(j)->info();
            faces_sites.push_back({site_id_1, site_id_2});
        }
    }

    // 7. 转换结果为Eigen格式
    result.vertices.resize(voronoi_vertices.size(), 3);
    for (size_t i = 0; i < voronoi_vertices.size(); ++i) {
        result.vertices.row(i) = pointToEigen(voronoi_vertices[i]);
    }

    result.vertex_sites = vertex_sites_info;

    // 8. 三角化所有面并合并
    std::vector<std::array<int, 3>> all_triangles;
    for (int i=0;i<voronoi_faces.size();++i) {
        auto& face=voronoi_faces[i];
        auto triangles = triangulatePolygon(face);
        all_triangles.insert(all_triangles.end(), triangles.begin(), triangles.end());
        result.face_sites.insert(result.face_sites.end(), triangles.size(), faces_sites[i]);
    }

    result.faces.resize(all_triangles.size(), 3);
    for (size_t i = 0; i < all_triangles.size(); ++i) {
        result.faces.row(i) = Eigen::Vector3i(all_triangles[i][0],
                                             all_triangles[i][1],
                                             all_triangles[i][2]);
    }
    // std::cout<<"face size: "<<result.faces.rows()<<" "<<result.face_sites.size()<<std::endl;
    return result;
}

VoronoiResult computeVoronoi3D(const Eigen::MatrixXd& vertices,
                             const Eigen::MatrixXi& faces,
                             const std::vector<Eigen::Vector3d>& sites) {
    return computeVoronoi(vertices, faces, sites);
}

std::vector<VoronoiVertexInfo> getVoronoiVertexDetails(const VoronoiResult& result,
                                                      const std::vector<Eigen::Vector3d>& sites) {
    std::vector<VoronoiVertexInfo> details;
    details.reserve(result.vertices.rows());

    for (int i = 0; i < result.vertices.rows(); ++i) {
        VoronoiVertexInfo info;
        info.position = result.vertices.row(i);
        info.sites = result.vertex_sites[i];

        // 计算到第一个站点的距离作为外接球半径
        if (info.sites[0] >= 0 && info.sites[0] < static_cast<int>(sites.size())) {
            info.radius = (info.position - sites[info.sites[0]]).norm();
        } else {
            info.radius = 0.0;
        }

        details.push_back(info);
    }

    return details;
}

} // namespace VoronoiComputation