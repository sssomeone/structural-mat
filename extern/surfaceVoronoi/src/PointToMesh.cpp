//
// Created by pengfei on 2025/9/6.
//
#include <Eigen/Dense>
#include <vector>
#include <utility> // For std::pair
#include <limits> // For std::numeric_limits
#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <utility>
#include <limits>
#include <vector>
#include <algorithm>

// libigl 真实存在的头文件
#include <igl/AABB.h>
#include <igl/point_simplex_squared_distance.h>
// 4. 用于计算点到三角形的距离 (通过 AABB 树)

using namespace std;
struct DistanceResult {
	double signedDistance;    // 有符号距离
	Eigen::Vector3d closestPoint;  // 最近点
	int faceID;
};
#include <mutex>
#include <igl/signed_distance.h>
#include <igl/AABB.h>
#include <igl/WindingNumberAABB.h>
#include <Eigen/Core>

enum QUERY_TYPE {
	QUERY_TYPE_PSEUDONORMAL,
	QUERY_TYPE_WINDING_NUMBER,
	QUERY_TYPE_FAST_WINDING_NUMBER  // 新增快速缠绕数选项
};
class PointToMeshDistance {
	typedef Eigen::MatrixXd DerivedV;
	typedef Eigen::MatrixXi DerivedF;
	typedef Eigen::Vector3d DerivedQ;


public:
	std::mutex query_time_mutex;
	int query_time = 0;
	DerivedV V;
	DerivedF F;
	igl::AABB<DerivedV, 3> tree;


	Eigen::MatrixXd FN, VN, EN;
	Eigen::MatrixXi E;
	Eigen::VectorXi EMAP;

	igl::FastWindingNumberBVH             fwn_bvh;

	QUERY_TYPE query_type;

public:
	PointToMeshDistance(const DerivedV& vertices, const DerivedF& faces, QUERY_TYPE _query_type = QUERY_TYPE::QUERY_TYPE_PSEUDONORMAL)
		: V(vertices),
		F(faces),
		query_type(_query_type)
	{
		tree.init(V, F);

		igl::per_face_normals(V, F, FN);
		igl::per_vertex_normals(
			V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
		igl::per_edge_normals(
			V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);

		igl::fast_winding_number(
			V.cast<float>(), F,
			/* expansion_order = */ 2,
			fwn_bvh
		);
	}
	DistanceResult signed_distance(const Eigen::Vector3d& query_point) {
		return signed_distance(vector<Eigen::Vector3d>{query_point})[0];
	}
	vector<DistanceResult> signed_distance(const vector<Eigen::Vector3d>& query_point) {
		vector<DistanceResult> res;
		switch (query_type) {
			case QUERY_TYPE_PSEUDONORMAL: {
				for(auto v:query_point) {
					res.push_back(signed_distance_using_pseudonormal(v));
				}
			}
			case QUERY_TYPE_FAST_WINDING_NUMBER: {
				for(auto v:query_point) {
					res.push_back(signed_distance_using_fast_winding(v));
				}
			}
			default: {
				for(auto v:query_point) {
					res.push_back(signed_distance_using_pseudonormal(v));
				}
			}
		}
		return res;
	}

	// ——— 3. fast winding-number + pseudonormal 最近点（内部取负，外部取正） ———
	DistanceResult signed_distance_using_fast_winding(
		const Eigen::Vector3d& query_point
	) const {
		// 先用 pseudonormal 得到最近点 & 无符号距离
		DistanceResult base = signed_distance_using_pseudonormal(query_point);
		double dist = std::abs(base.signedDistance);

		// 单点查询需要一个 1×3 的浮点矩阵
		Eigen::Matrix<float, 1, 3> Qf;
		Qf.row(0) = query_point.cast<float>();

		// 计算 fast winding-number
		Eigen::VectorXf W;
		igl::fast_winding_number(
			fwn_bvh,
			/* accuracy_scale = */ 2.0f,
			Qf,
			W
		);
		// W(0) > 0.5 → 内部，否则外部
		bool inside = (W(0) > 0.5f);

		DistanceResult res;
		res.signedDistance = inside ? -dist : dist;
		res.closestPoint = base.closestPoint;
		return res;
	}
	DistanceResult signed_distance_using_pseudonormal(const Eigen::Vector3d& query_point)const {
		DistanceResult res;
		Eigen::VectorXd S;
		Eigen::VectorXi I;
		Eigen::MatrixXd N, C;
		igl::signed_distance_pseudonormal(query_point.transpose(), V, F, tree, FN, VN, EN, EMAP, S, I, C, N);
		res.signedDistance = S(0);
		res.closestPoint = C.row(0); // 转置回 Vector3d
		res.faceID=I(0);
		return res;
	}
};



/**
 * @brief 线段到线段的最近点对和距离平方（这个需要手动实现）
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> segmentToSegment(
    const Eigen::Vector3d& p1, const Eigen::Vector3d& q1,
    const Eigen::Vector3d& p2, const Eigen::Vector3d& q2)
{
    Eigen::Vector3d d1 = q1 - p1;
    Eigen::Vector3d d2 = q2 - p2;
    Eigen::Vector3d r = p1 - p2;

    double a = d1.squaredNorm();
    double e = d2.squaredNorm();
    double f = d2.dot(r);

    const double EPSILON = 1e-10;
    double s, t;

    // 处理退化情况
    if (a <= EPSILON && e <= EPSILON) {
        s = t = 0.0;
        Eigen::Vector3d c1 = p1;
        Eigen::Vector3d c2 = p2;
        return {c1, c2, (c1 - c2).squaredNorm()};
    }

    if (a <= EPSILON) {
        s = 0.0;
        t = std::clamp(f / e, 0.0, 1.0);
    } else {
        double c = d1.dot(r);
        if (e <= EPSILON) {
            t = 0.0;
            s = std::clamp(-c / a, 0.0, 1.0);
        } else {
            double b = d1.dot(d2);
            double denom = a * e - b * b;

            if (denom != 0.0) {
                s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }

            t = (b * s + f) / e;

            if (t < 0.0) {
                t = 0.0;
                s = std::clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    Eigen::Vector3d c1 = p1 + s * d1;
    Eigen::Vector3d c2 = p2 + t * d2;

    return {c1, c2, (c1 - c2).squaredNorm()};
}

/**
 * @brief 线段-三角形相交检测（Möller-Trumbore）
 */
bool segmentTriangleIntersection(
    const Eigen::Vector3d& s0, const Eigen::Vector3d& s1,
    const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2,
    Eigen::Vector3d& intersection)
{
    const double EPSILON = 1e-10;

    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;
    Eigen::Vector3d dir = s1 - s0;

    Eigen::Vector3d h = dir.cross(edge2);
    double a = edge1.dot(h);

    if (std::abs(a) < EPSILON) return false;

    double f = 1.0 / a;
    Eigen::Vector3d s = s0 - v0;
    double u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) return false;

    Eigen::Vector3d q = s.cross(edge1);
    double v = f * dir.dot(q);

    if (v < 0.0 || u + v > 1.0) return false;

    double t = f * edge2.dot(q);

    if (t >= 0.0 && t <= 1.0) {
        intersection = s0 + t * dir;
        return true;
    }

    return false;
}

/**
 * @brief 计算3D线段和3D三角形之间的最近距离及最近点
 */
std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>
findClosestPointsSegmentTriangle(
    const std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>& triangle,
    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& segment)
{
    // --- 1. 解包输入 ---
    const auto& V0 = std::get<0>(triangle);
    const auto& V1 = std::get<1>(triangle);
    const auto& V2 = std::get<2>(triangle);
    const auto& S0 = segment.first;
    const auto& S1 = segment.second;

    // --- 2. 初始化最小距离和最近点 ---
    double min_sqr_dist = std::numeric_limits<double>::infinity();
    Eigen::Vector3d closest_point_on_triangle;
    Eigen::Vector3d closest_point_on_segment;

    auto update_min =
        [&](double sqr_dist, const Eigen::Vector3d& p_tri, const Eigen::Vector3d& p_seg)
    {
        if (sqr_dist < min_sqr_dist) {
            min_sqr_dist = sqr_dist;
            closest_point_on_triangle = p_tri;
            closest_point_on_segment = p_seg;
        }
    };

    // --- 3. 检查相交 ---
    Eigen::Vector3d intersection;
    if (segmentTriangleIntersection(S0, S1, V0, V1, V2, intersection)) {
        return std::make_tuple(0.0, intersection, intersection);
    }

    // --- 4. Case 1 & 2: 线段端点到三角形 (使用 AABB) ---
    {
        Eigen::MatrixXd V(3, 3);
        V.row(0) = V0;
        V.row(1) = V1;
        V.row(2) = V2;

        Eigen::MatrixXi F(1, 3);
        F << 0, 1, 2;

        Eigen::MatrixXd P(2, 3);
        P.row(0) = S0;
        P.row(1) = S1;

        igl::AABB<Eigen::MatrixXd, 3> tree;
        tree.init(V, F);

        Eigen::VectorXd sqrD;
        Eigen::VectorXi I;
        Eigen::MatrixXd C;

        tree.squared_distance(V, F, P, sqrD, I, C);

        update_min(sqrD(0), C.row(0), S0);
        update_min(sqrD(1), C.row(1), S1);
    }

    // --- 5. Case 3, 4, 5: 三角形顶点到线段 ---
    // 使用 igl::point_simplex_squared_distance
    {
        // 构建线段的顶点和元素矩阵
        Eigen::MatrixXd seg_V(2, 3);
        seg_V.row(0) = S0;
        seg_V.row(1) = S1;

        Eigen::MatrixXi seg_Ele(1, 2);
        seg_Ele << 0, 1;

        for (const auto& tri_vertex : {V0, V1, V2}) {
            Eigen::RowVector3d p = tri_vertex;
            double sqr_d;
            Eigen::RowVector3d c;

            igl::point_simplex_squared_distance<3>(
                p, seg_V, seg_Ele, 0, sqr_d, c
            );

            update_min(sqr_d, tri_vertex, c);
        }
    }

    // --- 6. Case 6, 7, 8: 线段到三角形三条边 (边-边距离) ---
    {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> tri_edges = {
            {V0, V1}, {V1, V2}, {V2, V0}
        };

        for (const auto& [E0, E1] : tri_edges) {
            auto [closest_on_seg, closest_on_edge, sqr_dist] =
                segmentToSegment(S0, S1, E0, E1);
            update_min(sqr_dist, closest_on_edge, closest_on_seg);
        }
    }

    // --- 7. 返回最终结果 ---
    return std::make_tuple(
        std::sqrt(min_sqr_dist),
        closest_point_on_triangle,
        closest_point_on_segment
    );
}
