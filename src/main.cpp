#include "surfaceVoronoi.hpp"
#include "sampleOnMesh.cpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <igl/read_triangle_mesh.h>
#include "VoronoiComputation.h"
#include "RobustMeshSimplification.h"
#include "medial_simplification.h"
#include "portable-file-dialogs.h"

#include "polyscope/pick.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using namespace std;
using namespace surface_voronoi;

fcpw::Scene<3> scene;

int colorScheme = 0;
polyscope::SurfaceMesh* psMesh = nullptr;
Eigen::MatrixXd input_mesh_V;
Eigen::MatrixXi input_mesh_F;


void initFCPW() {

	scene.setObjectCount(1);

	scene.setObjectTypes({{fcpw::PrimitiveType::Triangle}});

	// 3. 设置顶点数据
	// FCPW 可能不直接接受 Eigen::MatrixXd (double)，通常建议转换为 std::vector<fcpw::Vector3>
	std::vector<fcpw::Vector3> fcpw_vertices;
	fcpw_vertices.reserve(input_mesh_V.rows());
	for(int i = 0; i < input_mesh_V.rows(); ++i) {
		fcpw_vertices.emplace_back(input_mesh_V(i, 0), input_mesh_V(i, 1), input_mesh_V(i, 2));
	}
	scene.setObjectVertices(fcpw_vertices, 0);

	// 3. 设置三角形数量，然后逐个添加三角形
	scene.setObjectTriangleCount(input_mesh_F.rows(), 0);
	for(int i = 0; i < input_mesh_F.rows(); ++i) {
		scene.setObjectTriangle(
			fcpw::Vector3i(
				input_mesh_F(i, 0),
				input_mesh_F(i, 1),
				input_mesh_F(i, 2)
			),
			i,  // 三角形索引
			0   // 对象索引
		);
	}


	// 5. 构建 BVH
	scene.build(fcpw::AggregateType::Bvh_SurfaceArea, true);
}

float mesh_transparency = 1.0f;

std::vector<Eigen::Vector3d> samples_sites;
int samples_num = 50000;
bool using_blue_noise = true;


VoronoiComputation::VoronoiResult voronoi_res;
vector<vector<vector<Eigen::Vector3d>>> voronoi_regions;
polyscope::SurfaceMesh* voronoi_2d_psMesh = nullptr;

CustomMeshSimplifier* custom_simplifier = nullptr;
Eigen::MatrixXd medial_V_ours;
Eigen::MatrixXi medial_F_ours;
CustomMeshSimplifier::Edge min_edge;

bool GoOn=false;


void displaySamples() {
	if (samples_sites.empty()) {
		std::cout << "No samples to display" << std::endl;
		return;
	}

	// 转换 Eigen::Vector3d 为 glm::vec3
	std::vector<glm::vec3> points;
	points.reserve(samples_sites.size());

	for (const auto& p : samples_sites) {
		points.push_back(glm::vec3{p.x(), p.y(), p.z()});
	}

	// 注册点云
	polyscope::PointCloud* psCloud = polyscope::registerPointCloud("Sample Sites", points);

	// 设置选项
	psCloud->setPointRadius(0.0015);
	// psCloud->setPointRenderMode(polyscope::PointRenderMode::Quad);
	psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
	psCloud->setEnabled(true);

	std::cout << "Displayed " << samples_sites.size() << " sample points" << std::endl;
}





void displayVoronoiRegions(const vector<vector<vector<Eigen::Vector3d>>>& voronoi_regions,const string region_name="2d voronoi") {
    if (voronoi_regions.empty()) {
        std::cout << "No Voronoi regions to display" << std::endl;
        return;
    }

    // 存储所有顶点和面片
    std::vector<glm::vec3> all_vertices;
    std::vector<std::array<size_t, 3>> all_faces;
    std::vector<glm::vec3> face_colors;  // 每个面片的颜色

    // 为每个站点生成随机颜色
    std::vector<glm::vec3> site_colors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> color_dist(0.3f, 1.0f);

    for (size_t i = 0; i < voronoi_regions.size(); i++) {
        site_colors.push_back(glm::vec3(color_dist(gen), color_dist(gen), color_dist(gen)));
    }

    // 处理每个站点的Voronoi区域
    for (size_t site_idx = 0; site_idx < voronoi_regions.size(); site_idx++) {
        const auto& site_region = voronoi_regions[site_idx];
        const glm::vec3& site_color = site_colors[site_idx];

        // 处理该站点的每个面片
        for (const auto& face_vertices : site_region) {
            if (face_vertices.size() < 3) {
                continue;  // 跳过无效面片
            }

            // 对多边形进行扇形三角化
            for (size_t i = 1; i < face_vertices.size() - 1; i++) {
                // 当前三角形的三个顶点
                const auto& v0 = face_vertices[0];
                const auto& v1 = face_vertices[i];
                const auto& v2 = face_vertices[i + 1];

                // 添加顶点
                size_t idx0 = all_vertices.size();
                all_vertices.push_back(glm::vec3(v0.x(), v0.y(), v0.z()));

                size_t idx1 = all_vertices.size();
                all_vertices.push_back(glm::vec3(v1.x(), v1.y(), v1.z()));

                size_t idx2 = all_vertices.size();
                all_vertices.push_back(glm::vec3(v2.x(), v2.y(), v2.z()));

                // 添加面片
                all_faces.push_back({idx0, idx1, idx2});

                // 该三角形使用该站点的颜色
                face_colors.push_back(site_color);
            }
        }
    }

    // std::cout << "Voronoi regions: " << voronoi_regions.size() << " sites" << std::endl;
    // std::cout << "Total vertices: " << all_vertices.size() << std::endl;
    // std::cout << "Total faces: " << all_faces.size() << std::endl;


    voronoi_2d_psMesh = polyscope::registerSurfaceMesh(
        region_name,
        all_vertices,
        all_faces
    );
	voronoi_2d_psMesh->setEnabled(true);
    // 添加面片颜色
    voronoi_2d_psMesh->addFaceColorQuantity("Color", face_colors)->setEnabled(true);

    // 可选：显示边框
    voronoi_2d_psMesh->setEdgeWidth(0.5);
}
// ================== MODIFIED FUNCTION ==================
void visualizeEdgeCollapse(const Eigen::Vector3d& v1,
                           const Eigen::Vector3d& v2) {
    // === Prepare data for visualization ===
    std::vector<glm::vec3> edge_nodes = {
        glm::vec3(v1.x(), v1.y(), v1.z()),
        glm::vec3(v2.x(), v2.y(), v2.z())
    };
    std::vector<std::array<size_t, 2>> edge_indices = {{0, 1}};

    // === Update or Register logic ===

    // --- 1. Collapse Edge (Curve Network) ---
    if (polyscope::hasCurveNetwork("Collapse Edge")) {
        // If it exists, just update its vertex positions
        polyscope::getCurveNetwork("Collapse Edge")->updateNodePositions(edge_nodes);
    } else {
        // If it doesn't exist, register it for the first time with initial settings
        auto edge = polyscope::registerCurveNetwork("Collapse Edge", edge_nodes, edge_indices);
        edge->setRadius(0.00032);
        edge->setColor(glm::vec3{1.0, 0.0, 0.0}); // Red
    }



    // --- 3. Edge Endpoints (Point Cloud) ---
    if (polyscope::hasPointCloud("Edge Endpoints")) {
        // If it exists, just update its point positions
        polyscope::getPointCloud("Edge Endpoints")->updatePointPositions(edge_nodes);
    } else {
        // If it doesn't exist, register it for the first time
        auto endpoints = polyscope::registerPointCloud("Edge Endpoints", edge_nodes);
        endpoints->setPointRadius(0.0025);
        endpoints->setPointColor(glm::vec3{1.0, 1.0, 0.0}); // Yellow
    }
}

double w_triangle = 0.000005;
double w_stable_radio = 0.025;

constexpr const char* kMedialAxisEdgesName = "Medial Axis Edges (Ours)";

void updateCurrentMedialAxisEdges(const std::map<int, int>& new_to_old_vertex_id) {
	if (polyscope::hasCurveNetwork(kMedialAxisEdgesName)) {
		polyscope::removeStructure(kMedialAxisEdgesName);
	}

	if (custom_simplifier == nullptr || medial_V_ours.rows() == 0) {
		return;
	}

	std::vector<glm::vec3> nodes;
	nodes.reserve(medial_V_ours.rows());
	std::map<int, size_t> old_to_new_vertex_id;

	for (int new_id = 0; new_id < medial_V_ours.rows(); ++new_id) {
		auto old_id = new_to_old_vertex_id.find(new_id);
		if (old_id == new_to_old_vertex_id.end()) {
			continue;
		}

		old_to_new_vertex_id[old_id->second] = nodes.size();
		nodes.emplace_back(
			medial_V_ours(new_id, 0),
			medial_V_ours(new_id, 1),
			medial_V_ours(new_id, 2)
		);
	}

	std::vector<std::array<size_t, 2>> current_edges;
	current_edges.reserve(custom_simplifier->num_edges);
	for (const auto& edge : custom_simplifier->edges) {
		if (!edge.valid) {
			continue;
		}

		auto v1 = old_to_new_vertex_id.find(edge.v1);
		auto v2 = old_to_new_vertex_id.find(edge.v2);
		if (v1 != old_to_new_vertex_id.end() && v2 != old_to_new_vertex_id.end()) {
			current_edges.push_back({v1->second, v2->second});
		}
	}

	if (current_edges.empty()) {
		return;
	}

	auto edge_network = polyscope::registerCurveNetwork(
		kMedialAxisEdgesName,
		nodes,
		current_edges
	);
	edge_network->setRadius(0.0004);
	edge_network->setColor(glm::vec3{0.15f, 0.15f, 0.15f});
	edge_network->setEnabled(true);
}

std::string formatFixed(double value, int precision = 6) {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(precision) << value;
	return oss.str();
}

int edge_collapse_per_view=1000000;
struct SelectedElement {
	polyscope::MeshElement type;
	int64_t index = -1;
	bool hasSelection = false;
} selectedElement;


auto getEdgeID = [](size_t v1, size_t v2) -> size_t {
	// 这里是你自己的边索引逻辑
	// 例如：从你的 halfedge 数据结构中查询
	// 或者从你的边列表中查找

	auto it = custom_simplifier->vertex_pair_to_edge.find({v1,v2});

	if (it != custom_simplifier->vertex_pair_to_edge.end()) {
		return it->second; // 找到了，直接返回边的ID
	}
	else {
		cout<<"cant find edge"<<endl;
		exit(-1);
	}
};


struct select_edge_info {
	int v1,v2;
} info_of_selected_edge;
std::vector<size_t> edgePerm;
std::map<int, int> new_to_old_vertex_id;
std::vector<size_t> buildEdgePermutationFromCustomIndex(
	const Eigen::MatrixXi& F,
	std::function<size_t(size_t, size_t)> getEdgeIDFunc_Undirected,
	std::map<int, int>& new_to_old_vertex_id)
{
	std::vector<size_t> edgePermutation;
	std::set<std::pair<size_t, size_t>> processedEdges;  // ✓ 需要去重！

	for (int f = 0; f < F.rows(); f++) {
		for (int i = 0; i < F.cols(); i++) {
			size_t v0 = new_to_old_vertex_id[F(f, i)];
			size_t v1 = new_to_old_vertex_id[F(f, (i + 1) % 3)];

			// 规范化，因为是无向边
			if (v0 > v1) std::swap(v0, v1);

			// ✓ 只在第一次遇到这条边时添加
			if (processedEdges.find({v0, v1}) == processedEdges.end()) {
				processedEdges.insert({v0, v1});

				// 获取用户的边 ID
				size_t userEdgeID = getEdgeIDFunc_Undirected(v0, v1);
				if (userEdgeID>1e8) {
					cout<<"what the fuck?"<<userEdgeID<<endl;
					exit(-1);
				}
				edgePermutation.push_back(userEdgeID);

			}

		}
	}

	return edgePermutation;  // ✓ 大小 = E (117313)
}
int edgeIndex=-1;


// ================== FEATURE LINE VISUALIZATION ==================

float convex_feature_angle_threshold = 50; // 默认 30 度
float concave_feature_angle_threshold = 30.0f; // 默认 30 度

void displayFeatureEdges(const Eigen::MatrixXd& V, const std::vector<FeatureEdge>& features) {

	// 准备两组数据容器
	std::vector<glm::vec3> nodesYang, nodesYin;
	std::vector<std::array<size_t, 2>> edgesYang, edgesYin;

	for (const auto& feat : features) {
		// 获取顶点坐标
		Eigen::Vector3d p1_eigen = V.row(feat.v1);
		Eigen::Vector3d p2_eigen = V.row(feat.v2);

		glm::vec3 p1 = {(float)p1_eigen.x(), (float)p1_eigen.y(), (float)p1_eigen.z()};
		glm::vec3 p2 = {(float)p2_eigen.x(), (float)p2_eigen.y(), (float)p2_eigen.z()};

		if (feat.type == YANG_CONVEX) {
			// 存入阳角数据
			size_t idx = nodesYang.size();
			nodesYang.push_back(p1);
			nodesYang.push_back(p2);
			edgesYang.push_back({idx, idx + 1});
		}
		else if (feat.type == YIN_CONCAVE) {
			// 存入阴角数据
			size_t idx = nodesYin.size();
			nodesYin.push_back(p1);
			nodesYin.push_back(p2);
			edgesYin.push_back({idx, idx + 1});
		}
	}

	// --- 注册阳角 (红色) ---
	if (!edgesYang.empty()) {
		auto psYang = polyscope::registerCurveNetwork("Features Yang (Convex)", nodesYang, edgesYang);
		psYang->setColor(glm::vec3(1.0f, 0.0f, 0.0f)); // 纯红
		psYang->setRadius(0.0015);
		psYang->setEnabled(true);
	} else {
		polyscope::removeStructure("Features Yang (Convex)");
	}

	// --- 注册阴角 (蓝色) ---
	if (!edgesYin.empty()) {
		auto psYin = polyscope::registerCurveNetwork("Features Yin (Concave)", nodesYin, edgesYin);
		psYin->setColor(glm::vec3(0.0f, 0.5f, 1.0f)); // 亮蓝
		psYin->setRadius(0.0015);
		psYin->setEnabled(true);
	} else {
		polyscope::removeStructure("Features Yin (Concave)");
	}

	std::cout << "Displayed Features -> Yang: " << edgesYang.size()
			  << ", Yin: " << edgesYin.size() << std::endl;
}


/**
 * @brief 将顶点矩阵V就地归一化到 [-0.5, 0.5]^3 的立方体中
 *
 * 该函数会平移模型的几何中心到 (0,0,0)，
 * 然后进行 *均匀缩放* (保持长宽比)，
 * 使得模型最长的维度刚好等于 1.0。
 *
 * @param V 一个 NxC 矩阵 (例如 Nx3)，N 是顶点数。该矩阵将被就地修改。
 */
void normalizeMesh(Eigen::MatrixXd& V) {
	if (V.rows() == 0) {
		std::cout << "Mesh is empty, skipping normalization." << std::endl;
		return;
	}

	// 1. 找到当前模型的包围盒 (AABB)
	//    minCoeff() / maxCoeff() 会返回一个列向量 (3x1)
	//    我们需要的是行向量 (1x3)
	Eigen::RowVector3d min_coords = V.colwise().minCoeff();
	Eigen::RowVector3d max_coords = V.colwise().maxCoeff();

	// 2. 计算包围盒的中心
	Eigen::RowVector3d center = (min_coords + max_coords) / 2.0;

	// 3. 计算包围盒的范围 (extents)
	Eigen::RowVector3d extents = max_coords - min_coords;

	// 4. 找到最长的维度
	//    extents.maxCoeff() 返回 (width, height, depth) 中的最大值
	double largest_dimension = extents.maxCoeff();

	// 5. 计算缩放因子
	//    目标是让最长的维度等于 1.0 (因为 -0.5 到 0.5 的宽度是 1.0)
	double scale_factor = 20;
	if (largest_dimension > 1e-9) { // 避免除以零
		scale_factor = 1.0 / largest_dimension;
	}

	// 6. 应用变换：先平移到中心，然后缩放
	//    V.rowwise() - center:  Eigen的广播操作，将 'center' (1x3) 从 V 的 *每一行* 减去
	//    (...) * scale_factor: 将结果矩阵中的每个元素乘以 'scale_factor'
	V = (V.rowwise() - center) * scale_factor;

	std::cout << "Mesh normalized: " << V.rows() << " vertices." << std::endl;
	std::cout << " - Original center: (" << center << ")" << std::endl;
	std::cout << " - Scale factor applied: " << scale_factor << std::endl;
}



bool _boundry_feature_preserve=true;

vector<FeatureType> sites_feature_crossing_flags;
map<pair<int,int>,FeatureType> boundry_of_faces;


void displayCrossingSites(const std::vector<Eigen::Vector3d>& sites,
                          const std::vector<FeatureType>& sites_feature_crossing_flags) {

    // 1. 安全检查：确保两个数组大小一致
    if (sites.size() != sites_feature_crossing_flags.size()) {
        std::cerr << "[Error] displayCrossingSites: Input vectors have different sizes!" << std::endl;
        return;
    }

    // 2. 准备两个容器，分别存储阳角和阴角的点
    std::vector<glm::vec3> points_yang; // 存储阳角 (Convex)
    std::vector<glm::vec3> points_yin;  // 存储阴角 (Concave)

    // 预留空间 (估算)
    points_yang.reserve(sites.size() / 20);
    points_yin.reserve(sites.size() / 20);

    // 3. 遍历并分类
    for (size_t i = 0; i < sites.size(); ++i) {
        if (sites_feature_crossing_flags[i] == FeatureType::YANG_CONVEX) {
            points_yang.push_back({
                (float)sites[i].x(), (float)sites[i].y(), (float)sites[i].z()
            });
        }
        else if (sites_feature_crossing_flags[i] == FeatureType::YIN_CONCAVE) {
            points_yin.push_back({
                (float)sites[i].x(), (float)sites[i].y(), (float)sites[i].z()
            });
        }
    }

    // --- 4. 注册阳角点云 (Yang/Convex) ---
    std::string name_yang = "sites_feature_crossing_yang";
    if (!points_yang.empty()) {
        auto psYang = polyscope::registerPointCloud(name_yang, points_yang);
        // 设置为红色
        psYang->setPointColor(glm::vec3{1.0f, 0.0f, 0.0f});
        psYang->setPointRadius(0.002);
        psYang->setEnabled(true);
    } else {
        // 如果没有点，移除旧的结构
        if (polyscope::hasPointCloud(name_yang)) {
            polyscope::removeStructure(name_yang);
        }
    }

    // --- 5. 注册阴角点云 (Yin/Concave) ---
    std::string name_yin = "sites_feature_crossing_yin";
    if (!points_yin.empty()) {
        auto psYin = polyscope::registerPointCloud(name_yin, points_yin);
        // 设置为蓝色 (或青色)
        psYin->setPointColor(glm::vec3{0.0f, 0.5f, 1.0f});
        psYin->setPointRadius(0.002);
        psYin->setEnabled(true);
    } else {
        if (polyscope::hasPointCloud(name_yin)) {
            polyscope::removeStructure(name_yin);
        }
    }

    // 清理旧的合并结构 (如果你之前运行过旧代码)
    if (polyscope::hasPointCloud("sites_feature_crossing")) {
        polyscope::removeStructure("sites_feature_crossing");
    }

    std::cout << "Displaying Feature Crossing Sites -> Yang: " << points_yang.size()
              << ", Yin: " << points_yin.size() << std::endl;
}
int target_vertex_num=5000;
struct TimeData {
	double ours_samples_time=0;
	double ours_feature_extract_time=0;
	double voronoi_3d_time=0;
	double vorono_2d_time=0;
	double init_time=0;
	double simplify_time=0;
	double ours_all() {
		return ours_samples_time+ours_feature_extract_time+voronoi_3d_time+vorono_2d_time+init_time+simplify_time;
	}
};
TimeData time_stats;
std::vector<FeatureEdge> features;





// void delete_usefuless_faces(VoronoiComputation::VoronoiResult& voronoi_res) {
// 	int total_faces = voronoi_res.faces.rows();
// 	int i = 0;
//
// 	// 使用 while 循环，因为删除时我们不移动 i，而是检查被换过来的新元素
// 	while (i < total_faces) {
//
// 		// 获取当前面的信息，用于判断
// 		// Eigen::Vector3i face = voronoi_res.faces.row(i);
// 		std::pair<int, int> sites = voronoi_res.face_sites[i];
//
// 		bool should_delete = false;
//
// 		if (sites_feature_crossing_flags[sites.first]==FeatureType::YIN_CONCAVE&&sites_feature_crossing_flags[sites.second]==FeatureType::YIN_CONCAVE) {
// 			Eigen::Vector3d query_point=(samples_sites[sites.first]+samples_sites[sites.second])/2;
//
// 			fcpw::Interaction<3> interaction;
// 			fcpw::Vector3 q(query_point.x(), query_point.y(), query_point.z());
// 			bool found = scene.findClosestPoint(q, interaction);
//
// 			double medial_r = 0.5 * (samples_sites[sites.first]-samples_sites[sites.second]).norm();
// 			// if (abs(interaction.d)<0.001){
// 			// 	should_delete=true;
// 			// }
// 			if (medial_r*0.5>abs(interaction.d)) {
// 				should_delete = true;
// 			}
// 			// cout<<"end"<<endl;
//
// 		}
//
// 		if (should_delete) {
// 			// 核心逻辑：将当前元素与数组末尾的有效元素交换
// 			// 1. 交换 Eigen 矩阵的行 (注意：Eigen 的 swap 非常快)
// 			voronoi_res.faces.row(i).swap(voronoi_res.faces.row(total_faces - 1));
//
// 			// 2. 交换 std::vector 的元素
// 			std::swap(voronoi_res.face_sites[i], voronoi_res.face_sites[total_faces - 1]);
//
// 			// 3. 缩小有效范围 (逻辑删除)
// 			total_faces--;
//
// 			// 注意：这里不要执行 i++
// 			// 因为刚才从末尾换过来的那个新元素还没有被检查过，
// 			// 下次循环需要重新检查位置 i 的元素。
// 		} else {
// 			// 如果保留，则检查下一个
// 			i++;
// 		}
// 	}
//
// 	// 循环结束后，统一调整大小 (物理删除)
// 	// 这一步是必须的，释放多余的内存并更新矩阵维度信息
// 	if (total_faces < voronoi_res.faces.rows()) {
// 		voronoi_res.faces.conservativeResize(total_faces, 3);
// 		voronoi_res.face_sites.resize(total_faces);
// 	}
//
// 	// 1. 建立旧ID到新ID的映射表，初始化为 -1 表示未被使用
// 	std::vector<int> old_to_new_id(voronoi_res.vertices.rows(), -1);
// 	int new_vertex_count = 0;
//
// 	// 2. 遍历剩余的有效面片，标记所有被用到的顶点
// 	for (int i = 0; i < voronoi_res.faces.rows(); ++i) {
// 		for (int j = 0; j < 3; ++j) {
// 			int old_id = voronoi_res.faces(i, j);
// 			if (old_to_new_id[old_id] == -1) {
// 				// 这是一个新发现的有效顶点，给它分配一个新的顺序ID
// 				old_to_new_id[old_id] = new_vertex_count++;
// 			}
// 		}
// 	}
//
// 	// 3. 如果有效顶点数小于总顶点数，说明存在游离点，需要清理
// 	if (new_vertex_count < voronoi_res.vertices.rows()) {
// 		std::cout << "Cleaning up " << (voronoi_res.vertices.rows() - new_vertex_count)
// 				  << " isolated vertices." << std::endl;
//
// 		Eigen::MatrixXd new_vertices(new_vertex_count, 3);
// 		std::vector<Eigen::Vector4i> new_vertex_sites;
// 		new_vertex_sites.reserve(new_vertex_count); // 预分配内存
//
// 		// 4. 构建新的顶点数组 和 属性数组
// 		// 我们遍历旧数组，如果该旧ID有对应的新ID，就拷贝数据
// 		for (int old_id = 0; old_id < voronoi_res.vertices.rows(); ++old_id) {
// 			int new_id = old_to_new_id[old_id];
// 			if (new_id != -1) {
// 				// 拷贝坐标
// 				new_vertices.row(new_id) = voronoi_res.vertices.row(old_id);
//
// 				// 拷贝属性 (vertex_sites)
// 				// 注意：一定要做这一步，否则顶点和站点信息的对应关系就乱了！
// 				new_vertex_sites.push_back(voronoi_res.vertex_sites[old_id]);
// 			}
// 		}
//
// 		// 5. 更新面片索引 (Remap indices)
// 		for (int i = 0; i < voronoi_res.faces.rows(); ++i) {
// 			for (int j = 0; j < 3; ++j) {
// 				int old_id = voronoi_res.faces(i, j);
// 				voronoi_res.faces(i, j) = old_to_new_id[old_id];
// 			}
// 		}
//
// 		// 6. 覆盖旧数据
// 		voronoi_res.vertices = new_vertices;
// 		voronoi_res.vertex_sites = new_vertex_sites;
// 	}
// }
// void delete_usefuless_faces(VoronoiComputation::VoronoiResult& voronoi_res) {
//     // ================= Part 1: 删除面片 =================
//     int total_faces = voronoi_res.faces.rows();
//     int i = 0;
//
//     while (i < total_faces) {
//         std::pair<int, int> sites = voronoi_res.face_sites[i];
//         bool should_delete = false;
//
//         // 特征线检查
//         if (sites_feature_crossing_flags[sites.first] == FeatureType::YIN_CONCAVE &&
//             sites_feature_crossing_flags[sites.second] == FeatureType::YIN_CONCAVE) {
//
//             Eigen::Vector3d query_point = (samples_sites[sites.first] + samples_sites[sites.second]) / 2.0;
//
//             fcpw::Interaction<3> interaction;
//             fcpw::Vector3 q(query_point.x(), query_point.y(), query_point.z());
//
//             // 注意：这里需要确保 scene 已经 build 过了
//             scene.findClosestPoint(q, interaction);
//
//             double medial_r = 0.5 * (samples_sites[sites.first] - samples_sites[sites.second]).norm();
//
//             // 启发式过滤：如果中轴球半径 远大于 实际到表面的距离，说明是个不稳定的“扁平”球
//             if (medial_r * 0.5 > std::abs(interaction.d)) {
//                 should_delete = true;
//             }
//         }
//
//         if (should_delete) {
//             // Swap and Pop
//             voronoi_res.faces.row(i).swap(voronoi_res.faces.row(total_faces - 1));
//             std::swap(voronoi_res.face_sites[i], voronoi_res.face_sites[total_faces - 1]);
//             total_faces--;
//             // i 不自增，重新检查换过来的面
//         } else {
//             i++;
//         }
//     }
//
//     // 物理调整大小
//     if (total_faces < voronoi_res.faces.rows()) {
//         voronoi_res.faces.conservativeResize(total_faces, 3);
//         voronoi_res.face_sites.resize(total_faces);
//     }
//
//     // ================= Part 2: 清理游离点 (Fix Bug Here) =================
//
//     // 1. 建立映射: old_id -> new_id
//     std::vector<int> old_to_new_id(voronoi_res.vertices.rows(), -1);
//     int new_vertex_count = 0;
//
//     // 通过遍历剩余的有效面片，找出所有被引用的顶点
//     for (int f = 0; f < voronoi_res.faces.rows(); ++f) {
//         for (int j = 0; j < 3; ++j) {
//             int old_id = voronoi_res.faces(f, j);
//             if (old_to_new_id[old_id] == -1) {
//                 old_to_new_id[old_id] = new_vertex_count++;
//             }
//         }
//     }
//
//     // 如果有顶点被移除了
//     if (new_vertex_count < voronoi_res.vertices.rows()) {
//         std::cout << "Cleaning up " << (voronoi_res.vertices.rows() - new_vertex_count)
//                   << " isolated vertices." << std::endl;
//
//         Eigen::MatrixXd new_vertices(new_vertex_count, 3);
//         std::vector<Eigen::Vector4i> new_vertex_sites;
//
//         // 【关键修复】必须先 resize，然后用下标赋值，不能用 push_back
//         // 否则数据顺序会因为 old_id 的遍历顺序而乱掉
//         new_vertex_sites.resize(new_vertex_count);
//
//         // 2. 迁移数据
//         for (int old_id = 0; old_id < voronoi_res.vertices.rows(); ++old_id) {
//             int new_id = old_to_new_id[old_id];
//             if (new_id != -1) {
//                 // 拷贝几何位置
//                 new_vertices.row(new_id) = voronoi_res.vertices.row(old_id);
//                 // 拷贝属性 (Fix: 使用下标访问，确保 new_id 对应正确的数据)
//                 new_vertex_sites[new_id] = voronoi_res.vertex_sites[old_id];
//             }
//         }
//
//         // 3. 更新面片索引
//         for (int f = 0; f < voronoi_res.faces.rows(); ++f) {
//             for (int j = 0; j < 3; ++j) {
//                 int old_id = voronoi_res.faces(f, j);
//                 voronoi_res.faces(f, j) = old_to_new_id[old_id];
//             }
//         }
//
//         // 4. 覆盖旧数据
//         voronoi_res.vertices = new_vertices;
//         voronoi_res.vertex_sites = new_vertex_sites;
//     }
// }
void delete_usefuless_faces(VoronoiComputation::VoronoiResult& voronoi_res) {
	int total_faces = voronoi_res.faces.rows();
	int i = 0;

	// === 第一部分：删除无用面片 ===
	while (i < total_faces) {
		std::pair<int, int> sites = voronoi_res.face_sites[i];
		bool should_delete = false;

		if (sites_feature_crossing_flags[sites.first]==FeatureType::YIN_CONCAVE &&
		    sites_feature_crossing_flags[sites.second]==FeatureType::YIN_CONCAVE) {
			Eigen::Vector3d query_point = (samples_sites[sites.first] + samples_sites[sites.second]) / 2;
			fcpw::Interaction<3> interaction;
			fcpw::Vector3 q(query_point.x(), query_point.y(), query_point.z());
			scene.findClosestPoint(q, interaction);

			double medial_r = 0.5 * (samples_sites[sites.first] - samples_sites[sites.second]).norm();
			if (medial_r * 0.5 > abs(interaction.d)) {
				should_delete = true;
			}
		}

		if (should_delete) {
			voronoi_res.faces.row(i).swap(voronoi_res.faces.row(total_faces - 1));
			std::swap(voronoi_res.face_sites[i], voronoi_res.face_sites[total_faces - 1]);
			total_faces--;
		} else {
			i++;
		}
	}

	// 物理删除面片
	if (total_faces < voronoi_res.faces.rows()) {
		voronoi_res.faces.conservativeResize(total_faces, 3);
		voronoi_res.face_sites.resize(total_faces);
	}

	// === 第二部分：删除游离顶点 ===

	// 1. 建立旧ID到新ID的映射
	std::vector<int> old_to_new_id(voronoi_res.vertices.rows(), -1);
	int new_vertex_count = 0;

	// 2. 遍历剩余面片，标记所有被用到的顶点
	for (int i = 0; i < voronoi_res.faces.rows(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int old_id = voronoi_res.faces(i, j);
			if (old_to_new_id[old_id] == -1) {
				old_to_new_id[old_id] = new_vertex_count++;
			}
		}
	}

	// 3. 如果有游离点，需要清理
	if (new_vertex_count < voronoi_res.vertices.rows()) {
		std::cout << "Cleaning up " << (voronoi_res.vertices.rows() - new_vertex_count)
		          << " isolated vertices." << std::endl;

		// ✓ 修复：预先分配完整大小
		Eigen::MatrixXd new_vertices(new_vertex_count, 3);
		std::vector<Eigen::Vector4i> new_vertex_sites(new_vertex_count);  // ✓ 预分配大小

		// 4. ✓ 修复：按 new_id 索引赋值
		for (int old_id = 0; old_id < voronoi_res.vertices.rows(); ++old_id) {
			int new_id = old_to_new_id[old_id];
			if (new_id != -1) {
				// 拷贝坐标
				new_vertices.row(new_id) = voronoi_res.vertices.row(old_id);
				// ✓ 修复：用索引赋值而不是 push_back
				new_vertex_sites[new_id] = voronoi_res.vertex_sites[old_id];
			}
		}

		// 5. 更新面片索引
		for (int i = 0; i < voronoi_res.faces.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				int old_id = voronoi_res.faces(i, j);
				voronoi_res.faces(i, j) = old_to_new_id[old_id];
			}
		}

		// 6. 覆盖旧数据
		voronoi_res.vertices = new_vertices;
		voronoi_res.vertex_sites = new_vertex_sites;
	}
}


bool cad_input = true;
struct thres_hold {
	float cad_convex_feature_angle_threshold = 50; // 默认 30 度
	float cad_concave_feature_angle_threshold = 30.0f; // 默认 30 度

	float organic_convex_feature_angle_threshold = 181.0f; // 默认 30 度
	float organic_concave_feature_angle_threshold = 30.0f; // 默认 30 度

	float sigma=45;
} thres_hold_ours;
void myCallback() {
	if (ImGui::IsMouseDoubleClicked(0)) {
		// 获取选中信息
		auto selection = polyscope::getSelection();

		polyscope::view::setViewCenterAndLookAt(selection.position, true);
		std::cout << "Rotation center set!" << std::endl;
	}

	ImGui::Text("Input Mesh");
	ImGui::Separator();

	if (ImGui::Button("Browse Model File...")) {
		// 打开文件选择对话框
		auto selection = pfd::open_file(
			"Select a model file",                    // 标题
			".",                                       // 默认路径
			{"Mesh Files", "*.obj *.ply *.off *.stl", // 文件过滤器
			 "All Files", "*"},
			pfd::opt::none                            // 选项
		).result();

		if (!selection.empty()) {
			if (!igl::read_triangle_mesh(selection[0], input_mesh_V, input_mesh_F)) {
				std::cerr << "Failed to load mesh: " << selection[0] << std::endl;
				return;
			}
			normalizeMesh(input_mesh_V);
			std::cout << "Mesh Loaded: "<< selection[0] << std::endl;
			polyscope::removeAllStructures();
			psMesh = polyscope::registerSurfaceMesh("input_mesh", input_mesh_V, input_mesh_F);
			polyscope::view::resetCameraToHomeView();
			psMesh->setEnabled(true);
			time_stats.simplify_time=0;
			initFCPW();
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Save Model File...")) {
		if (custom_simplifier == nullptr) {
			std::cerr << "No simplified medial axis is available to save." << std::endl;
			return;
		}

		auto save_path = pfd::save_file(
			"Save medial axis as...",
			".",
			{"OBJ file", "*.obj"},
			pfd::opt::force_overwrite
		).result();

		if (!save_path.empty()) {
			custom_simplifier->exportMesh(save_path);
			std::cout << "Medial axis saved to: " << save_path << std::endl;
		} else {
			std::cout << "Save operation cancelled." << std::endl;
		}
	}

	if (input_mesh_V.rows() == 0 || input_mesh_F.rows() == 0) {
		ImGui::TextDisabled("Load a triangle mesh to begin.");
		return;
	}

	ImGui::Text("Parametic");
	ImGui::Separator();

	// double x=0.025;
	ImGui::InputDouble("Quality w1: ",&w_triangle);
	ImGui::InputDouble("Stability ratio w: ",&w_stable_radio);
	// ImGui::InputDouble("Euclidean w: ", &w_triangle, 0.0, 0.0, "%.10g");
	// ImGui::InputDouble("Euclidean h: ", &w_omega, 0.0, 0.0, "%.10g");
	// ImGui::InputDouble("Stability ratio w: ", &w_stable_radio, 0.0, 0.0, "%.10g");

	ImGui::Text("Feature Analysis");
	ImGui::SameLine();
	ImGui::Checkbox("CAD Input",&cad_input);
	ImGui::Separator();
	ImGui::SliderFloat("Convex Angle Threshold", &thres_hold_ours.sigma, 1.0f, 90.0f);
	if (cad_input) {
		convex_feature_angle_threshold=thres_hold_ours.sigma;
		concave_feature_angle_threshold=thres_hold_ours.sigma;
	}
	else {
		convex_feature_angle_threshold=thres_hold_ours.organic_convex_feature_angle_threshold;
		concave_feature_angle_threshold=thres_hold_ours.sigma;
	}

	// if (cad_input) {
	// 	ImGui::SliderFloat("Convex Angle Threshold", &thres_hold_ours.cad_convex_feature_angle_threshold, 1.0f, 90.0f);
	// 	ImGui::SliderFloat("Concave Angle Threshold", &thres_hold_ours.cad_concave_feature_angle_threshold, 1.0f, 90.0f);
	// 	convex_feature_angle_threshold=thres_hold_ours.cad_convex_feature_angle_threshold;
	// 	concave_feature_angle_threshold=thres_hold_ours.cad_concave_feature_angle_threshold;
	// }
	// else {
	// 	ImGui::SliderFloat("Concave Angle Threshold", &thres_hold_ours.organic_concave_feature_angle_threshold, 1.0f, 90.0f);
	// 	convex_feature_angle_threshold=thres_hold_ours.organic_convex_feature_angle_threshold;
	// 	concave_feature_angle_threshold=thres_hold_ours.organic_concave_feature_angle_threshold;
	// }


	if (ImGui::Button("Detect Feature Lines")) {

		auto start_feature = std::chrono::high_resolution_clock::now();

		// auto features = extractFeatureLines(input_mesh_V, input_mesh_F, feature_angle_threshold);

		 features = extractFeatureLines(input_mesh_V, input_mesh_F, convex_feature_angle_threshold,concave_feature_angle_threshold);

		boundry_of_faces.clear();
		for (auto& l:features) {
			if (l.type!=FeatureType::NON_FEATURE) {
				boundry_of_faces[{l.f1,l.f2}]=l.type;
			}
		}
		auto end_feature = std::chrono::high_resolution_clock::now();
		auto duration_feature = std::chrono::duration<double, std::milli>(end_feature - start_feature);
		std::cout << "Extract Feature Time: " << duration_feature.count() << " ms" << std::endl;
		time_stats.ours_feature_extract_time = duration_feature.count();
		displayFeatureEdges(input_mesh_V, features);
	}



 //    // 添加文本
 //    ImGui::Text("Sampling");
 //    ImGui::Separator();
 //
 //
	// // 或者用输入框（任意数值）
	// ImGui::InputInt("Samples Number", &samples_num);
 //
	// // 复选框控制是否使用蓝噪声
	// ImGui::Checkbox("Using Blue Noise", &using_blue_noise);
	// ImGui::SameLine();


	// --- 标题 ---
	ImGui::Spacing();
	ImGui::Text("Sampling");
	ImGui::Separator();

	// --- 采样数量 ---
	ImGui::AlignTextToFramePadding();
	ImGui::Text("Samples:");         // 简洁命名

	ImGui::SameLine();
	ImGui::SetNextItemWidth(100);    // 这里可以稍微窄一点
	ImGui::InputInt("##samples", &samples_num, 0, 0);

	ImGui::SameLine();               // 继续在同一行显示复选框

	// --- 蓝噪声开关 ---
	// 这种写法：左边是框，右边是文字 "Blue Noise"
	ImGui::Checkbox("Blue Noise", &using_blue_noise);


	// 按钮 1: 加载立方体
	if (ImGui::Button("Sampling")) {

		auto start_sampling = std::chrono::high_resolution_clock::now();
		samples_sites = sample_mesh_surface_with_faces(input_mesh_V, input_mesh_F, samples_num,using_blue_noise).first;
		auto end_sampling = std::chrono::high_resolution_clock::now();
		auto duration_sampleing = std::chrono::duration<double, std::milli>(end_sampling - start_sampling);
		time_stats.ours_samples_time = duration_sampleing.count();
		std::cout << "Sampling Time: " << duration_sampleing.count() << " ms" << std::endl;

		displaySamples();
	}


    ImGui::Text("Voronoi");
	ImGui::Separator();

	if (ImGui::Button("2D Voronoi")) {
		vector<vector<int>> sites_of_face;

		auto start_2dvoronoi = std::chrono::high_resolution_clock::now();

		voronoi_regions = GetLRVD_Regions(input_mesh_V,input_mesh_F,samples_sites,sites_of_face);

		auto end_2dvoronoi = std::chrono::high_resolution_clock::now();
		auto duration_2dvoronoi = std::chrono::duration<double, std::milli>(end_2dvoronoi - start_2dvoronoi);
		std::cout << "2dvoronoi Time: " << duration_2dvoronoi.count() << " ms" << std::endl;
		time_stats.vorono_2d_time= duration_2dvoronoi.count();


		displayVoronoiRegions(voronoi_regions);
		if (psMesh!=nullptr) {
			psMesh->setEnabled(false);  // 隐藏
		}

		sites_feature_crossing_flags.clear();
		sites_feature_crossing_flags.resize(samples_sites.size(),FeatureType::NON_FEATURE);
		for (auto f:boundry_of_faces) {
			int f1=f.first.first, f2=f.first.second;
			for (int s1:sites_of_face[f1]) {
				for (int s2:sites_of_face[f2]) {
					if (s1==s2) {
						sites_feature_crossing_flags[s1]=f.second;
						break;
					}
				}
			}
		}

		cout<<"2d voronoi end end"<<endl;
		displayCrossingSites(samples_sites,sites_feature_crossing_flags);
	}

	ImGui::SameLine();
	if (ImGui::Button("3D Voronoi")) {

		auto start_3dvoronoi = std::chrono::high_resolution_clock::now();
		voronoi_res = VoronoiComputation::computeVoronoi3D(input_mesh_V,input_mesh_F,samples_sites);

		// if (cad_input) {
		if (true){
			delete_usefuless_faces(voronoi_res);
		}
		auto end_3dvoronoi = std::chrono::high_resolution_clock::now();
		auto duration_3dvoronoi = std::chrono::duration<double, std::milli>(end_3dvoronoi - start_3dvoronoi);
		std::cout << "3dvoronoi Time: " << duration_3dvoronoi.count() << " ms" << std::endl;
		time_stats.voronoi_3d_time= duration_3dvoronoi.count();

		auto voronoi_3d = polyscope::registerSurfaceMesh("3d_voronoi", voronoi_res.vertices, voronoi_res.faces);
		if (psMesh!=nullptr) {
			psMesh->setEnabled(false);  // 隐藏
		}
		voronoi_3d->setEnabled(true);  // 隐藏
	}



	ImGui::Text("Current Result");
	ImGui::Separator();
	// ImGui::Checkbox("boundry_feature_preserve",&_boundry_feature_preserve);
	// ImGui::SameLine();
	if (ImGui::Button("Init Queue")) {

		if (custom_simplifier!=nullptr) {
			delete custom_simplifier;
		}

		auto start_init = std::chrono::high_resolution_clock::now();

		custom_simplifier = new CustomMeshSimplifier(scene,voronoi_res.vertices, voronoi_res.faces, voronoi_regions, voronoi_res.vertex_sites,samples_sites,_boundry_feature_preserve,sites_feature_crossing_flags,w_triangle,w_stable_radio);

		custom_simplifier->min_face_area_threshold = 0.0;    // 禁用面积检查
		custom_simplifier->normal_flip_threshold = -1;     // 禁用法向量检查（180度）
		// custom_simplifier->analyzeMeshTopology();


		medial_V_ours=voronoi_res.vertices;
		medial_F_ours=voronoi_res.faces;

		cout<<"start updateEdgeQueue---------------------------------------------------------"<<endl;

		custom_simplifier->updateEdgeQueue();
		cout<<"end updateEdgeQueue---------------------------------------------------------"<<endl;

		custom_simplifier->getNextValidEdge(min_edge);


		auto end_init = std::chrono::high_resolution_clock::now();
		auto duration_init = std::chrono::duration<double, std::milli>(end_init - start_init);
		std::cout << "Init Time: " << duration_init.count() << " ms" << std::endl;
		time_stats.init_time= duration_init.count();



		auto ps_mesh = polyscope::registerSurfaceMesh("Medial Axis (Ours)", medial_V_ours, medial_F_ours);
		ps_mesh->setEdgeWidth(1.0);

		polyscope::SurfaceMesh* mesh = polyscope::getSurfaceMesh("2d voronoi");
		if (mesh!=nullptr) {
			mesh->setEnabled(false);
		}

		mesh = polyscope::getSurfaceMesh("3d_voronoi");
		if (mesh!=nullptr) {
			mesh->setEnabled(false);
		}

		auto sampling_point_cloud = polyscope::getPointCloud("Sample Sites");
		if (sampling_point_cloud!=nullptr) {
			sampling_point_cloud->setEnabled(false);
		}

		new_to_old_vertex_id.clear();
		custom_simplifier->getCurrentMesh(medial_V_ours, medial_F_ours,new_to_old_vertex_id);
		updateCurrentMedialAxisEdges(new_to_old_vertex_id);
		edgePerm = buildEdgePermutationFromCustomIndex(medial_F_ours,getEdgeID,new_to_old_vertex_id);
		ps_mesh->setEdgePermutation(edgePerm);

		cout<<"-6"<<endl;
		ps_mesh->setSelectionMode(polyscope::MeshSelectionMode::Auto);

		// 选择边
		// ps_mesh->setSelectionMode(polyscope::MeshSelectionMode::Auto);
		ps_mesh->markEdgesAsUsed();
	}
	ImGui::SameLine();
	if (ImGui::Button("Simplify One Edge")) {

		if (!custom_simplifier->collapseEdge(min_edge.id)) {
			std::cout << "consecutive failures" << std::endl;
		}
		else {
			std::cout << "consecutive success" << std::endl;
		}
		new_to_old_vertex_id.clear();
		custom_simplifier->getCurrentMesh(medial_V_ours, medial_F_ours,new_to_old_vertex_id);
		updateCurrentMedialAxisEdges(new_to_old_vertex_id);
		custom_simplifier->getNextValidEdge(min_edge);

		auto ps_mesh = polyscope::registerSurfaceMesh("Medial Axis (Ours)", medial_V_ours, medial_F_ours);
		edgePerm = buildEdgePermutationFromCustomIndex(medial_F_ours,getEdgeID,new_to_old_vertex_id);
		ps_mesh->setEdgePermutation(edgePerm);
		ps_mesh->setSelectionMode(polyscope::MeshSelectionMode::Auto);
		ps_mesh->markEdgesAsUsed();

		auto sites = custom_simplifier->sites_of_each_vertex[min_edge.v1];
		auto sites_2=custom_simplifier->sites_of_each_vertex[min_edge.v2];

		vector<vector<vector<Eigen::Vector3d>>> sub_regions;
		for (auto s:sites) {
			sub_regions.push_back(voronoi_regions[s.first]);
		}
		for (auto s:sites_2) {
			if (!sites.count(s.first)) {
				sub_regions.push_back(voronoi_regions[s.first]);
			}
		}
		Eigen::Vector3d v1=custom_simplifier->vertices[min_edge.v1].position;
		Eigen::Vector3d v2=custom_simplifier->vertices[min_edge.v2].position;
		visualizeEdgeCollapse(v1,v2);
		displayVoronoiRegions(sub_regions,"sub regions");
		// std::cout << "collapse_cost: "<<min_edge.collapse_cost << std::endl;
	}

	// ImGui::InputInt("Step Size: ", &edge_collapse_per_view);
	// // target_face_num
	// ImGui::InputInt("Target Vertices: ", &target_vertex_num);
	// 1. 设置步长 (Batch Size)
	// AlignTextToFramePadding 保证文字和输入框垂直居中对齐，非常重要！

	//
	ImGui::AlignTextToFramePadding();
	ImGui::Text("Step Size:");       // 显示的文字
	ImGui::SameLine();               // 放到同一行
	// "##BatchSize" 中 ## 后面的内容是ID，不会显示出来，但必须唯一
	ImGui::InputInt("##BatchSize", &edge_collapse_per_view);

	// 2. 设置目标点数 (Target Vertices)
	ImGui::AlignTextToFramePadding();
	ImGui::Text("Target Vertices:");
	ImGui::SameLine();
	ImGui::InputInt("##TargetVerts", &target_vertex_num);

	// 创建一个两列的表格，ImGuiTableFlags_SizingStretchProp 让列宽自动适配
	// if (ImGui::BeginTable("SettingsTable", 2, ImGuiTableFlags_SizingFixedFit))
	// {
	//
	// 	// --- 第一行 ---
	// 	ImGui::TableNextColumn();
	// 	ImGui::AlignTextToFramePadding(); // 垂直居中
	// 	ImGui::Text("Batch Size");        // 简洁的命名
	//
	// 	ImGui::TableNextColumn();
	// 	// SetNextItemWidth(-1) 让输入框填满剩余宽度
	// 	ImGui::SetNextItemWidth(-1);
	// 	ImGui::InputInt("##step", &edge_collapse_per_view);
	//
	// 	// --- 第二行 ---
	// 	ImGui::TableNextColumn();
	// 	ImGui::AlignTextToFramePadding();
	// 	ImGui::Text("Target Vertices");   // 简洁的命名
	//
	// 	ImGui::TableNextColumn();
	// 	ImGui::SetNextItemWidth(-1);
	// 	ImGui::InputInt("##target", &target_vertex_num);
	//
	// 	ImGui::EndTable();
	// }

	if (ImGui::Button("Simplify")) {
		GoOn=true;
	}

	ImGui::SameLine();
	if (ImGui::Button("Stop")) {
		GoOn=false;
	}

	if (GoOn) {
		int temp=edge_collapse_per_view;


		auto start_Simplification = std::chrono::high_resolution_clock::now();

		while (temp--&&custom_simplifier->num_vertices>target_vertex_num) {
			if (!custom_simplifier->collapseEdge(min_edge.id)) {
				std::cout << "consecutive failures" << std::endl;
			}
			if (!custom_simplifier->getNextValidEdge(min_edge)) {
				cout<<"no edge can collapse because TopologyPreserving!!!"<<endl;
				cout<<custom_simplifier->num_faces<<" / "<<target_vertex_num<<" "<<custom_simplifier->num_vertices<<" "<<temp<<endl;
			}
			if (custom_simplifier->num_faces%2000==0) {
				cout<<custom_simplifier->num_faces<<" / "<<target_vertex_num<<" "<<custom_simplifier->num_vertices<<" "<<temp<<endl;
			}
		}


		auto end_Simplification = std::chrono::high_resolution_clock::now();
		if (GoOn)time_stats.simplify_time += std::chrono::duration<double, std::milli>(end_Simplification - start_Simplification).count();
		if (custom_simplifier->num_vertices<=target_vertex_num) {
			GoOn=false;
		}

		cout<<"-----------------------------------"<<endl;
		cout<<"ours_samples_time ours_feature_extract_time voronoi_3d_time vorono_2d_time init_time simplify_time all"<<endl;
		cout<<time_stats.ours_samples_time<<" "<<time_stats.ours_feature_extract_time<<" "<<time_stats.voronoi_3d_time<<" "<<time_stats.vorono_2d_time<<" "<<time_stats.init_time<<" "<<time_stats.simplify_time<<" "<<time_stats.ours_all()<<endl;
		cout<<"w_triangle, w_stable_radio"<<endl;
		cout<<w_triangle<<" "<<w_stable_radio<<endl;

		Eigen::Vector3d v1=custom_simplifier->vertices[min_edge.v1].position;
		Eigen::Vector3d v2=custom_simplifier->vertices[min_edge.v2].position;
		visualizeEdgeCollapse(v1,v2);

		new_to_old_vertex_id.clear();

		custom_simplifier->getCurrentMesh(medial_V_ours, medial_F_ours,new_to_old_vertex_id);
		updateCurrentMedialAxisEdges(new_to_old_vertex_id);

		auto ps_mesh = polyscope::registerSurfaceMesh("Medial Axis (Ours)", medial_V_ours, medial_F_ours);


		psMesh->setBackFacePolicy(polyscope::BackFacePolicy::Identical);

		edgePerm = buildEdgePermutationFromCustomIndex(medial_F_ours,getEdgeID,new_to_old_vertex_id);
		ps_mesh->setEdgePermutation(edgePerm);

		ps_mesh->setSelectionMode(polyscope::MeshSelectionMode::Auto);

		ps_mesh->markEdgesAsUsed();

		auto sites = custom_simplifier->sites_of_each_vertex[min_edge.v1];
		auto sites_2=custom_simplifier->sites_of_each_vertex[min_edge.v2];

		vector<vector<vector<Eigen::Vector3d>>> sub_regions;
		for (auto s:sites) {
			sub_regions.push_back(voronoi_regions[s.first]);
		}
		for (auto s:sites_2) {
			if (!sites.count(s.first)) {
				sub_regions.push_back(voronoi_regions[s.first]);
			}
		}
		// std::cout << "collapse_cost: "<<min_edge.collapse_cost << std::endl;
		displayVoronoiRegions(sub_regions,"sub regions");
	}
	ImGui::Text("Visable");
	ImGui::Separator();


	if (ImGui::Button("See Collapse Edge")) {
		bool vis = polyscope::getCurveNetwork("Collapse Edge")->isEnabled();
		vis=!vis;
		polyscope::getCurveNetwork("Collapse Edge")->setEnabled(vis);
		polyscope::getPointCloud("Target Position")->setEnabled(vis);
		polyscope::getPointCloud("Edge Endpoints")->setEnabled(vis);
		polyscope::getSurfaceMesh("sub regions")->setEnabled(vis);
	}




	const std::string collapse_cost_text = "collapse_cost: " + formatFixed(min_edge.collapse_cost, 12);
	ImGui::Text("%s", collapse_cost_text.c_str());



	ImGui::Text("Select Info:");
	ImGui::Separator();

	ImGuiIO& io = ImGui::GetIO();

    // 检测左键点击
    if (ImGui::IsMouseClicked(0)) {
        glm::vec2 screenCoords = glm::vec2(io.MousePos.x, io.MousePos.y);

        // 执行拾取
        polyscope::PickResult pick = polyscope::pickAtScreenCoords(screenCoords);

        if (pick.isHit) {
            std::cout << "Hit structure: " << pick.structureName << std::endl;

            // 如果点击的是我们的网格
            if (pick.structureName == "Medial Axis (Ours)") {
                // 获取网格对象
                polyscope::SurfaceMesh* mesh =
                polyscope::getSurfaceMesh("Medial Axis (Ours)");
            	// Medial Axis (Ours)

                // 解析网格特定的拾取结果
                polyscope::SurfaceMeshPickResult meshPick =
                    mesh->interpretPickResult(pick);

                selectedElement.type = meshPick.elementType;
                selectedElement.index = meshPick.index;
                selectedElement.hasSelection = true;

                std::cout << "Element index: " << meshPick.index << std::endl;
            	int edge_index_to_process = -1; // 用于存储我们真正要处理的边的ID
            	switch (meshPick.elementType) {

			        // --- !! 新增 CASE !! ---
	        case polyscope::MeshElement::CORNER: {
	           std::cout << "✓ Selected CORNER (User Edge ID: " << meshPick.index << ")" << std::endl;
	           // 'index' 现在 *就是* 你的自定义边 ID
	           int h_idx = meshPick.index;


		        int f_idx = h_idx / 3; // 面索引 (Face index)
		        int l_idx = h_idx % 3; // 面内本地索引 (Local index: 0, 1, or 2)

	        	// 从 *全局* F 矩阵中获取 *new* (Polyscope) 顶点 ID
	        	int new_v_start = medial_F_ours(f_idx, l_idx);
	        	int new_v_end   = medial_F_ours(f_idx, (l_idx + 1) % 3);

	        	// 从 *全局* 映射中转译回 *old* (Simplifier) 顶点 ID
	        	int old_v_start = new_to_old_vertex_id[new_v_start];
	        	int old_v_end   = new_to_old_vertex_id[new_v_end];
	        	if (old_v_start>old_v_end) {
	        		swap(old_v_start, old_v_end);
	        	}
	        	edge_index_to_process = getEdgeID(old_v_start, old_v_end);
		           break;
		        }

		        case polyscope::MeshElement::HALFEDGE: {
		           std::cout << "✓ Selected HALFEDGE (User Edge ID: " << meshPick.index << ")" << std::endl;
		           // 'index' 现在 *就是* T 你的自定义边 ID
		           // edge_index_to_process = meshPick.index;
	        	int h_idx = meshPick.index;

	        	// int ff_idx = meshPick.faceInd;     // <-- 使用 Polyscope 提供的面索引
	        	// int ll_idx = meshPick.halfedgeInd; // <-- 使用 Polyscope 提供的半边索引 (0, 1, 或 2)

	        	int f_idx = h_idx / 3; // 面索引 (Face index)
	        	int l_idx = h_idx % 3; // 面内本地索引 (Local index: 0, 1, or 2)

	        	// 从 *全局* F 矩阵中获取 *new* (Polyscope) 顶点 ID
	        	int new_v_start = medial_F_ours(f_idx, l_idx);
	        	int new_v_end   = medial_F_ours(f_idx, (l_idx + 1) % 3);

	        	// 从 *全局* 映射中转译回 *old* (Simplifier) 顶点 ID
	        	int old_v_start = new_to_old_vertex_id[new_v_start];
	        	int old_v_end   = new_to_old_vertex_id[new_v_end];
	        	if (old_v_start>old_v_end) {
	        		swap(old_v_start, old_v_end);
	        	}
	        	edge_index_to_process = getEdgeID(old_v_start, old_v_end);
	        	break;

		        }

		        case polyscope::MeshElement::EDGE: {
		           std::cout << "✓ Selected EDGE (User Edge ID: " << meshPick.index << ")" << std::endl;
		           // 'index' *已经* 是你的自定义边 ID
		           edge_index_to_process = meshPick.index;
		           break;
		        }

		        case polyscope::MeshElement::VERTEX:
		            std::cout << "✓ Selected VERTEX (ID: " << meshPick.index << ")" << std::endl;
		            break;
		        case polyscope::MeshElement::FACE:
		            std::cout << "✓ Selected FACE (ID: " << meshPick.index << ")" << std::endl;

            		// vector<vector<vector<Eigen::Vector3d>>> sub_regions;
            		// 	sub_regions.push_back(voronoi_regions[voronoi_res.face_sites[meshPick.index].first]);
            		// 	sub_regions.push_back(voronoi_regions[voronoi_res.face_sites[meshPick.index].second]);
              //
              //
              //
            		// 	// Eigen::Vector3d v1=voronoi_res.vertices.row(voronoi_res.faces(meshPick.index,0)).transpose();
            		// 	// Eigen::Vector3d v2=voronoi_res.vertices.row(voronoi_res.faces(meshPick.index,1)).transpose();
            		// 	// Eigen::Vector3d v3=voronoi_res.vertices.row(voronoi_res.faces(meshPick.index,2)).transpose();
            		// 	// double r1=abs(ptm->signed_distance(v1).signedDistance);
            		// 	// double r2=abs(ptm->signed_distance(v2).signedDistance);
            		// 	// double r3=abs(ptm->signed_distance(v3).signedDistance);
            		// 	// double grat=computeTriangleGradientNorm(v1,v2,v3,r1,r2,r3);
            		// 	// cout<<"norm: "<<grat<<endl;
              //
              //
              //
            		// // std::cout << "collapse_cost: "<<min_edge.collapse_cost << std::endl;
            		// displayVoronoiRegions(sub_regions,"sub regions");

		            break;
		    }



		    // 2. 在 switch *之后*，统一处理
		    if (edge_index_to_process != -1 && edge_index_to_process < custom_simplifier->edges.size()) {


		        // --- 这里是你的所有原始逻辑 ---
		        // ... (你显示和更新 GUI 的所有代码) ...

		        info_of_selected_edge.v1=custom_simplifier->edges[edge_index_to_process].v1;
		        info_of_selected_edge.v2=custom_simplifier->edges[edge_index_to_process].v2;

		        Eigen::Vector3d v1=custom_simplifier->vertices[info_of_selected_edge.v1].position;
		        Eigen::Vector3d v2=custom_simplifier->vertices[info_of_selected_edge.v2].position;
		        visualizeEdgeCollapse(v1,v2);


		    	auto sites = custom_simplifier->sites_of_each_vertex[info_of_selected_edge.v1];
		    	auto sites_2=custom_simplifier->sites_of_each_vertex[info_of_selected_edge.v2];

		    	vector<vector<vector<Eigen::Vector3d>>> sub_regions;
		    	for (auto s:sites) {
		    		sub_regions.push_back(voronoi_regions[s.first]);
		    	}

		    	for (auto s:sites_2) {
		    		if (!sites.count(s.first)) {
		    			sub_regions.push_back(voronoi_regions[s.first]);
		    		}
		    	}
		    	// std::cout << "collapse_cost: "<<min_edge.collapse_cost << std::endl;
		    	displayVoronoiRegions(sub_regions,"sub regions");

		        // 更新 ImGui 显示
		        edgeIndex=edge_index_to_process;


		    } else if (edge_index_to_process != -1) {
		        // ... (处理错误 ID) ...
		    }
            }
        }
    }



	ImGui::Separator();
    // 显示选择信息的 GUI
    // ImGui::Begin("Selection Info");
    if (selectedElement.hasSelection) {
        ImGui::Text("Index: %lld", selectedElement.index);
    } else {
        ImGui::Text("No selection");
    }
}






int main(int argc, char** argv) {
	if (argc > 2) {
		std::cerr << "Usage: " << argv[0] << " [input_mesh]" << std::endl;
		return 1;
	}

	if (argc == 2) {
		if (!igl::read_triangle_mesh(argv[1], input_mesh_V, input_mesh_F)) {
			std::cerr << "Failed to load mesh: " << argv[1] << std::endl;
			return 1;
		}
		normalizeMesh(input_mesh_V);
		initFCPW();
	}

	polyscope::options::programName = "Structural MAT";
	polyscope::init();
	polyscope::state::userCallback = myCallback;

	if (input_mesh_V.rows() > 0 && input_mesh_F.rows() > 0) {
		psMesh = polyscope::registerSurfaceMesh("input_mesh", input_mesh_V, input_mesh_F);
	}

	polyscope::show();
	delete custom_simplifier;
	custom_simplifier = nullptr;
	return 0;
}
