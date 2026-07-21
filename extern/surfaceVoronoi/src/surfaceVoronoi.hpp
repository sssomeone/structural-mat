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
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/Exact_integer.h>
#include <CGAL/Extended_homogeneous.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
//#include <CGAL/Nef_polyhedron_S2.h>
#include <CGAL/IO/Nef_polyhedron_iostream_3.h>
#include <CGAL/Aff_transformation_3.h>
typedef CGAL::Exact_predicates_exact_constructions_kernel inexact_Kernel;
// typedef CGAL::Polyhedron_3<inexact_Kernel>  Polyhedron;
typedef CGAL::Polygon_2<inexact_Kernel> Polygon_2;
typedef CGAL::Nef_polyhedron_3<inexact_Kernel>  Nef_polyhedron;
typedef inexact_Kernel::Point_3 Point_3;
typedef inexact_Kernel::Point_2 Point_2;
typedef inexact_Kernel::Vector_3 Vector_3;
typedef inexact_Kernel::Plane_3 Plane_3;
typedef CGAL::Aff_transformation_3<inexact_Kernel> Aff_transformation_3;
// typedef Polyhedron::HalfedgeDS             HalfedgeDS;
using namespace std;
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include "PointToMesh.cpp"
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
#include "MeshTopology.cpp"


#include <Eigen/dense>
#include <unordered_set>

typedef double REAL;
// typedef Kernel::FT REAL;

namespace surface_voronoi
{
	struct pair_hash
	{
		std::size_t operator () (const std::pair<int, int> & pair) const
		{
			std::size_t h1 = std::hash<int>()(pair.first);
			std::size_t h2 = std::hash<int>()(pair.second);

			return h1 ^ h2;
		}
	};


	struct Triangle_Base_Convex_Top
	{
		typedef Eigen::Vector<REAL,2> Point2;
		typedef Eigen::Vector3d Point3;

		const REAL maxHeight=1e9;
		typedef tuple<Point2, Point2, Point2> Triangle;
		int faceID;
		Triangle tri;
		tuple<REAL,REAL,REAL> computeBarycentricCoordinates(const Point2& point, const Triangle& triangle) {
			// 获取三角形的三个顶点
			const Point2& A = std::get<0>(triangle);
			const Point2& B = std::get<1>(triangle);
			const Point2& C = std::get<2>(triangle);

			// 计算向量
			Point2 v0 = C - A;  // AC向量
			Point2 v1 = B - A;  // AB向量
			Point2 v2 = point - A;  // AP向量

			// 计算点积
			REAL dot00 = v0.dot(v0);
			REAL dot01 = v0.dot(v1);
			REAL dot02 = v0.dot(v2);
			REAL dot11 = v1.dot(v1);
			REAL dot12 = v1.dot(v2);

			// 计算重心坐标
			REAL invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
			REAL v = (dot11 * dot02 - dot01 * dot12) * invDenom;
			REAL w = (dot00 * dot12 - dot01 * dot02) * invDenom;
			REAL u = 1.0 - v - w;
			return make_tuple(u, v, w);
		}

	public:
		struct Vertex;
		struct Edge;
		struct Hyperplane;

		map<int, Vertex> vertexs;
		map<int, Edge> edges;
		map<int, Hyperplane> hyperPlanes;

		int hyperPlaneId, vertexId, edgeId;

		//点
		struct Vertex {
			int hplane[3];
			Eigen::Matrix<REAL, 3, 1> AinvW;

			Vertex() {}
			Vertex(const Vertex& rhs) {
				for (int i = 0; i < 3; ++i) {
					hplane[i] = rhs.hplane[i];
				}
				AinvW = rhs.AinvW;
			};
			Point2 get_point()const {
				return Point2(-AinvW(0, 0),-AinvW(1, 0));
			}
			REAL coord(const int i) {
				return -AinvW(i, 0);
			}
			void setHyperPlane(const int h1, const int h2, const int h3, map<int, Hyperplane>& hyperPlanes) {
				hplane[0] = h1;
				hplane[1] = h2;
				hplane[2] = h3;
				initGAndw(hyperPlanes);
			}
			void setHyperPlane(const int h1, const int h2, const int h3) {
				hplane[0] = h1;
				hplane[1] = h2;
				hplane[2] = h3;
			}
			Vertex& operator=(const Vertex& rhs) {
				for (int i = 0; i < 3; ++i) {
					hplane[i] = rhs.hplane[i];
				}
				AinvW = rhs.AinvW;
				return *this;
			}
			void initGAndw(map<int, Hyperplane>& hyperPlanes) {
				Eigen::Matrix<REAL, 3, 3> A;
				Eigen::Matrix<REAL, 3, 1> W;
				for (int i = 0; i < 3; ++i) {
					A(i, 0) = hyperPlanes[hplane[i]].g(0);
					A(i, 1) = hyperPlanes[hplane[i]].g(1);
					A(i, 2) = hyperPlanes[hplane[i]].g(2);
					W(i) = hyperPlanes[hplane[i]].w;
				}
				AinvW = A.inverse() * W;
			}
		};

		//边
		struct Edge {
			int p1, p2;
			int hplanes[2];

			Edge(const Edge& rhs) :p1(rhs.p1), p2(rhs.p2) {
				for (int i = 0; i < 2; ++i)
					hplanes[i] = rhs.hplanes[i];
			}
			Edge(const int _p1, const int _p2, const int _h1, const int _h2) :\
				p1(_p1), p2(_p2) {
				hplanes[0] = _h1;
				hplanes[1] = _h2;
			};
			Edge(const int _p1, const int _p2) :\
				p1(_p1), p2(_p2) {
			};
			Edge() {
			};
		};


	struct Hyperplane {
		int sourceIndex;
		Eigen::Matrix<REAL, 3, 1> g;
		REAL w;
		Hyperplane() {}
		Hyperplane& operator=(const Hyperplane& rhs) {
			sourceIndex = rhs.sourceIndex;
			g = rhs.g;
			w = rhs.w;
			return *this;
		}
		Hyperplane(const Hyperplane& rhs) {
			sourceIndex = rhs.sourceIndex;
			g = rhs.g;
			w = rhs.w;
		}
		Hyperplane(const Triangle& rhs, const REAL d1, const REAL d2, const REAL d3,const int soureceids) :
			Hyperplane(get<0>(rhs), get<1>(rhs), get<2>(rhs), d1, d2, d3, soureceids) {
		}
		Hyperplane(const Point2& p1, const Point2& p2, const Point2& p3, REAL d1, REAL d2, REAL d3, int soureceids) : sourceIndex(soureceids) {
			if (sourceIndex < 0) {
				switch (sourceIndex)
				{
				case -5:
				case -4:
					g = Eigen::Matrix<REAL, 3, 1>(0, 0, -1);
					w = d3;
					break;

				default:
					break;
				}
			}
			else {
				Eigen::Matrix<REAL, 3, 3> m;

				Eigen::Matrix<REAL, 2, 1> _p1(p1(0), p1(1));
				Eigen::Matrix<REAL, 2, 1> _p2(p2(0), p2(1));
				Eigen::Matrix<REAL, 2, 1> _p3(p3(0), p3(1));


				m.block(0, 0, 1, 2) = _p1.transpose();
				m.block(1, 0, 1, 2) = _p2.transpose();
				m.block(2, 0, 1, 2) = _p3.transpose();

				m.col(2) = Eigen::Matrix<REAL, 3, 1>::Ones();

				if (abs(m.determinant()) == REAL(0)) {
					cerr << "cant be inverse" << endl;
					return;
					exit(-1);
				}

				Eigen::Matrix<REAL, 3, 1> right(d1, d2, d3);
				auto res = m.inverse() * right;

				g = Eigen::Matrix<REAL, 3, 1>(res(0), res(1), -1);
				w = res(2);
			}
		}
		Hyperplane(int sourceIds) :sourceIndex(sourceIds) {}

		enum VERTEXOFHYPERPLANE {
			ONPLANE,
			ONUPPERSIDE,
			ONBOTTONSIDE
		};
		REAL zero = REAL();
		VERTEXOFHYPERPLANE onUpperSide(const Vertex& rhs)const {
			REAL value = -(g.transpose() * rhs.AinvW)(0, 0) + w;
			return value > zero ? ONBOTTONSIDE : ONUPPERSIDE;
		}
	};


	Triangle_Base_Convex_Top(const Triangle_Base_Convex_Top& rhs) {
		hyperPlanes = rhs.hyperPlanes;
		vertexs = rhs.vertexs;
		edges = rhs.edges;
		hyperPlaneId = rhs.hyperPlaneId;
		vertexId = rhs.vertexId;
		edgeId = rhs.edgeId;
	}
	Triangle_Base_Convex_Top() {
		//应该是处理到面的程度，切割到。然后利用产生的线，形成新的面。
		for (int i = 1; i <= 6; ++i) {
			vertexs.insert(make_pair(i, Vertex()));
		}
		//三维空间中四面体有四个面，将这四个面分别提升至高维，四维空间中表现为体。不会访问其中的信息。侧面的四个的体下标分别为 1 ~ 3
		//这些hyperPlane不会调用onUpperSide函数
		for (int i = -3; i <= -1; ++i) {
			hyperPlanes.insert(make_pair(i, Hyperplane(i)));
		}

		//三维空间中的四面体提升到高维，四维空间中表现为体，id 为 0
		hyperPlanes.insert(make_pair(-4, Hyperplane(Point2(), Point2(), Point2(),  maxHeight, maxHeight, maxHeight, -4)));

		//底下的四面体
		hyperPlanes.insert(make_pair(-5, Hyperplane(Point2(), Point2(), Point2(), -maxHeight, -maxHeight, -maxHeight, -5)));

		edges[1] = Edge(1, 2, -2, -5);
		edges[2] = Edge(2, 3, -3, -5);
		edges[3] = Edge(1, 3, -1, -5);
		edges[4] = Edge(4, 5, -2, -4);
		edges[5] = Edge(5, 6, -4, -3);
		edges[6] = Edge(4, 6, -4, -1);
		edges[7] = Edge(1, 4, -1, -2);
		edges[8] = Edge(2, 5, -2, -3);
		edges[9] = Edge(3, 6, -1, -3);

		hyperPlaneId = -1;
		vertexId = 6;
		edgeId = 9;

	}
	void init(const tuple<Point2,Point2,Point2> p) {
		{
			Point2 p0 = get<0>(p), p1 = get<1>(p);
			REAL A = p1.y() - p0.y();        // y2 - y1
			REAL B = p0.x() - p1.x();        // x1 - x2
			REAL C = -(A * p0(0) + B * p0(1));
			hyperPlanes[-2].g = Eigen::Matrix<REAL, 3, 1>(A, B, 0);
			hyperPlanes[-2].w = C;
		}
		{
			Point2 p0 = get<1>(p), p1 = get<2>(p);
			REAL A = p1.y() - p0.y();        // y2 - y1
			REAL B = p0.x() - p1.x();        // x1 - x2
			REAL C = -(A * p0(0) + B * p0(1));
			hyperPlanes[-3].g = Eigen::Matrix<REAL, 3, 1>(A, B, 0);
			hyperPlanes[-3].w = C;
		}
		{
			Point2 p0 = get<0>(p), p1 = get<2>(p);
			REAL A = p1.y() - p0.y();        // y2 - y1
			REAL B = p0.x() - p1.x();        // x1 - x2
			REAL C = -(A * p0(0) + B * p0(1));
			hyperPlanes[-1].g = Eigen::Matrix<REAL, 3, 1>(A, B, 0);
			hyperPlanes[-1].w = C;
		}

		vertexs[1].setHyperPlane(-5, -1, -2);
		vertexs[1].AinvW = Eigen::Matrix<REAL, 3, 1>(get<0>(p).x(), get<0>(p).y(), -maxHeight);

		vertexs[2].setHyperPlane(-5, -2, -3);
		vertexs[2].AinvW = Eigen::Matrix<REAL, 3, 1>(get<1>(p).x(), get<1>(p).y(),-maxHeight);

		vertexs[3].setHyperPlane(-1, -3, -5);
		vertexs[3].AinvW = Eigen::Matrix<REAL, 3, 1>(get<2>(p).x(), get<2>(p).y(), -maxHeight);


		vertexs[4].setHyperPlane(-1, -2, -4);
		vertexs[4].AinvW = Eigen::Matrix<REAL, 3, 1>(get<0>(p).x(), get<0>(p).y(), maxHeight);

		vertexs[5].setHyperPlane(-2, -3, -4);
		vertexs[5].AinvW = Eigen::Matrix<REAL, 3, 1>(get<1>(p).x(), get<1>(p).y(), maxHeight);

		vertexs[6].setHyperPlane(-1, -3, -4);
		vertexs[6].AinvW = Eigen::Matrix<REAL, 3, 1>(get<2>(p).x(), get<2>(p).y(),  maxHeight);

		get<0>(tri)=get<0>(p);
		get<1>(tri)=get<1>(p);
		get<2>(tri)=get<2>(p);

	}

	int addHyperPlane(const Hyperplane& plane) {
		set<int> uselessEdges;
		set<int> uselessVertexs;

		map<int, Hyperplane::VERTEXOFHYPERPLANE> vertexsOnPlaneState;
		for (const auto& vertex : vertexs) {
			auto state = plane.onUpperSide(vertex.second);
			if (state == Hyperplane::VERTEXOFHYPERPLANE::ONUPPERSIDE) {
				uselessVertexs.insert(vertex.first);
			}
			vertexsOnPlaneState.insert(make_pair(vertex.first, state));
		}

		int status_code = 0;
		if (uselessVertexs.empty()) {
			return -1;
		}
		hyperPlaneId = plane.sourceIndex;
		hyperPlanes.insert(make_pair(hyperPlaneId, Hyperplane(plane)));

		map<int, set<int>> verticesInFacet;

		for (auto& curEdge:edges) {
			auto& e=curEdge.second;
			auto useEnd1Useless = vertexsOnPlaneState[e.p1];
			auto useEnd2Useless = vertexsOnPlaneState[e.p2];

			if (useEnd1Useless != Triangle_Base_Convex_Top::Hyperplane::VERTEXOFHYPERPLANE::ONBOTTONSIDE && useEnd2Useless != Triangle_Base_Convex_Top::Hyperplane::VERTEXOFHYPERPLANE::ONBOTTONSIDE) {
				uselessEdges.insert(curEdge.first);
				continue;
			}

			if (useEnd1Useless != Triangle_Base_Convex_Top::Hyperplane::VERTEXOFHYPERPLANE::ONBOTTONSIDE) {
				Vertex vertex_new;
				vertex_new.hplane[0] = e.hplanes[0];
				vertex_new.hplane[1] = e.hplanes[1];
				vertex_new.hplane[2] = hyperPlaneId;
				vertex_new.initGAndw(hyperPlanes);
				vertexs[++vertexId] = vertex_new;
				e.p1 = vertexId;
				verticesInFacet[e.hplanes[0]].insert(vertexId);
				verticesInFacet[e.hplanes[1]].insert(vertexId);

			}
			else if (useEnd2Useless != Triangle_Base_Convex_Top::Hyperplane::VERTEXOFHYPERPLANE::ONBOTTONSIDE) {
				Vertex vertex_new;

				vertex_new.hplane[0] = e.hplanes[0];
				vertex_new.hplane[1] = e.hplanes[1];
				vertex_new.hplane[2] = hyperPlaneId;
				vertex_new.initGAndw(hyperPlanes);

				vertexs[++vertexId] = Vertex(vertex_new);
				e.p2 = vertexId;

				verticesInFacet[e.hplanes[0]].insert(vertexId);
				verticesInFacet[e.hplanes[1]].insert(vertexId);
			}
		}

		for (auto& edg : uselessEdges) {
			edges.erase(edg);
		}

		for (auto& usef : uselessVertexs) {
			vertexs.erase(usef);
		}

		map<int, set<int>> edgesInHyperPlane;

		for (const pair<int, set<int>>& edge_pair : verticesInFacet) {

			if (edge_pair.second.size() != 2) {
				status_code = -1024;
				std::cout << "edge vertex not 2" << endl;
				std::cout << edge_pair.first << endl;
				std::cout << edge_pair.second.size() << endl;
			}

			Edge newEdge(*edge_pair.second.begin(), *edge_pair.second.rbegin());

			newEdge.hplanes[0] = edge_pair.first;
			newEdge.hplanes[1] = hyperPlaneId;

			edges[++edgeId] = newEdge;
		}
		return status_code;
	}
		vector<pair<Point3,Point3>> GetSegments(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F)
		{
			vector<pair<Point3,Point3>> result;
			for (auto cur_edge : edges)
			{
				auto e = cur_edge.second;
				if (e.hplanes[0] < 0 || e.hplanes[1] < 0) {
					continue;
				}
				auto p1_BarycentricCoordinates = computeBarycentricCoordinates(vertexs[e.p1].get_point(),tri);
				auto p2_BarycentricCoordinates = computeBarycentricCoordinates(vertexs[e.p2].get_point(),tri);
				//123,213,132,
				Eigen::Vector3d tri_p1=V.row(F(faceID,0));
				Eigen::Vector3d tri_p3=V.row(F(faceID,1));
				Eigen::Vector3d tri_p2=V.row(F(faceID,2));
				Point3 p1=CGAL::to_double(get<0>(p1_BarycentricCoordinates))*tri_p1+CGAL::to_double(get<1>(p1_BarycentricCoordinates))*tri_p2+CGAL::to_double(get<2>(p1_BarycentricCoordinates))*tri_p3;
				Point3 p2=CGAL::to_double(get<0>(p2_BarycentricCoordinates))*tri_p1+CGAL::to_double(get<1>(p2_BarycentricCoordinates))*tri_p2+CGAL::to_double(get<2>(p2_BarycentricCoordinates))*tri_p3;
				result.push_back(make_pair(p1, p2));
			}
			return result;
		}


	vector<int> constructFace(vector<pair<int,int>>& edges) {
		if (edges.empty()) {
			return {};
		}

		// 构建邻接表，每个点记录其相邻的点
		unordered_map<int, vector<int>> graph;
		for (const auto& edge : edges) {
			graph[edge.first].push_back(edge.second);
			graph[edge.second].push_back(edge.first);
		}

		// 从第一条边的第一个点开始遍历
		vector<int> result;
		int start = edges[0].first;
		int current = start;
		int previous = -1; // 记录上一个访问的点，避免走回头路

		do {
			result.push_back(current);

			// 找到下一个要访问的点（不是上一个访问的点）
			int next = -1;
			for (int neighbor : graph[current]) {
				if (neighbor != previous) {
					next = neighbor;
					break;
				}
			}

			// 更新为下一轮循环做准备
			previous = current;
			current = next;

		} while (current != start && current != -1);

		return result;
	}

		vector<pair<int,vector<Eigen::Vector3d>>> GetVoronoiRegions(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F)
	{
		map<int,vector<pair<int,int>>> regions;
		for (auto cur_edge : edges)
		{
			auto e = cur_edge.second;
			if (e.hplanes[0] < 0 && e.hplanes[1] < 0) {
				continue;
			}
			if (e.hplanes[0]>=0) {
				regions[e.hplanes[0]].push_back(make_pair(e.p1,e.p2));
			}
			if (e.hplanes[1]>=0) {
				regions[e.hplanes[1]].push_back(make_pair(e.p2,e.p1));
			}
		}

		Eigen::Vector3d tri_p1=V.row(F(faceID,0));
		Eigen::Vector3d tri_p3=V.row(F(faceID,1));
		Eigen::Vector3d tri_p2=V.row(F(faceID,2));
		vector<pair<int,vector<Eigen::Vector3d>>> result;

		Eigen::Vector3d face_normal=(tri_p2-tri_p1).cross(tri_p3-tri_p2).normalized();
		for (auto& region : regions) {
			auto face=constructFace(region.second);
			vector<Eigen::Vector3d> face_with_points;
			for (auto pid:face) {
				auto BarycentricCoordinates = computeBarycentricCoordinates(vertexs[pid].get_point(),tri);
				Point3 p=CGAL::to_double(get<0>(BarycentricCoordinates))*tri_p1+CGAL::to_double(get<1>(BarycentricCoordinates))*tri_p2+CGAL::to_double(get<2>(BarycentricCoordinates))*tri_p3;
				face_with_points.push_back(p);
			}
			Eigen::Vector3d cur_face_normal=(face_with_points[0]-face_with_points[1]).cross(face_with_points[2]-face_with_points[1]).normalized();
			if (cur_face_normal.dot(face_normal)<0) {
				reverse(face_with_points.begin(),face_with_points.end());
			}
			result.emplace_back(make_pair(region.first,face_with_points));
		}
		return result;
	}

	};
	Eigen::MatrixXi transform_vectorf_to_eigenf(const vector<Eigen::Vector3i>& dual_fs) {
		Eigen::MatrixXi F(dual_fs.size(),3);
		for (int i=0;i<dual_fs.size();++i) {
			F(i,0) = dual_fs[i].x();
			F(i,1) = dual_fs[i].y();
			F(i,2) = dual_fs[i].z();
		}
		return F;
	}

	vector<vector<tuple<int, REAL, REAL, REAL>>> InferOverPropagatedDistancesForLRVD(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F, const vector<Eigen::Vector3d>& sources)
	{
		PointToMeshDistance distance_query(V,F,QUERY_TYPE_PSEUDONORMAL);
		vector<Eigen::Vector<REAL,3>> verts(V.rows());
		MeshTopology::FaceAdjacencyProcessor processor(V,F);
		for (int i=0;i<verts.size();++i) {
			verts[i](0)=V(i,0);
			verts[i](1)=V(i,1);
			verts[i](2)=V(i,2);
		}
		vector<Eigen::Vector<REAL,3>> exact_sources(sources.size());
		for (int i=0;i<exact_sources.size();++i) {
			exact_sources[i]=Eigen::Vector<REAL,3>(sources[i].x(), sources[i].y(), sources[i].z());
		}
		vector<vector<tuple<int, REAL, REAL, REAL>>> m_distances(F.rows());

		struct Evt
		{
			int whichSource;
			int toWhichFace;

			REAL d1, d2, d3;
			const bool operator>(const Evt& other) const
			{
				return d1 + d2 + d3 > other.d1 + other.d2 + other.d3;
			}
		};

		vector<tsl::robin_set<int>> considered(sources.size());
		queue<Evt> pending;

		Evt evt;
		for (int i = 0; i < sources.size(); ++i)
		{
			evt.whichSource = i;
			evt.toWhichFace = distance_query.signed_distance(Eigen::Vector3d(sources[i].x(), sources[i].y(), sources[i].z())).faceID;

			evt.d1 = (exact_sources[i] - verts[F(evt.toWhichFace,0)]).squaredNorm();
			evt.d2 = (exact_sources[i] - verts[F(evt.toWhichFace,1)]).squaredNorm();
			evt.d3 = (exact_sources[i] - verts[F(evt.toWhichFace,2)]).squaredNorm();

			pending.push(evt);
			considered[evt.whichSource].insert(evt.toWhichFace);
		}

		// int threeNeighboringFaces[3];
		while (!pending.empty())
		{
			Evt& evt = pending.front();

			bool contribute=true;
			for (auto& field: m_distances[evt.toWhichFace]) {
				if (evt.d1 > get<1>(field) && evt.d2 > get<2>(field) && evt.d3 > get<3>(field)) {
					contribute=false;
					break;
				}
			}
			if (!contribute) {
				pending.pop();
				continue;
			}
			m_distances[evt.toWhichFace].emplace_back(evt.whichSource, evt.d1, evt.d2, evt.d3);

			std::vector<int> NeighboringFaces = processor.getAdjacentFaceIds(evt.toWhichFace);
			for (auto current_neighboring_face : NeighboringFaces)
			{
				if (considered[evt.whichSource].find(current_neighboring_face) != considered[evt.whichSource].end() || current_neighboring_face == -1)
					continue;
				evt.toWhichFace = current_neighboring_face;
				evt.d1 = (exact_sources[evt.whichSource] - verts[F(evt.toWhichFace,0)]).squaredNorm();
				evt.d2 = (exact_sources[evt.whichSource] - verts[F(evt.toWhichFace,1)]).squaredNorm();
				evt.d3 = (exact_sources[evt.whichSource] - verts[F(evt.toWhichFace,2)]).squaredNorm();
				pending.push(evt);
				considered[evt.whichSource].insert(evt.toWhichFace);
			}
			pending.pop();
		}


		return m_distances;
	}

	pair<vector<pair<Eigen::Vector3d, Eigen::Vector3d>>,pair<Eigen::MatrixXd,Eigen::MatrixXi>> GetLRVD_Bisectors(const Eigen::MatrixXd& V,const Eigen::MatrixXi&F, const vector<Eigen::Vector3d>& sources)
	{
		vector<pair<Eigen::Vector3d,Eigen::Vector3d>> segs;
		Eigen::MatrixXd dual_V(sources.size(),3);
		for (int i=0;i<sources.size();++i) {
			dual_V(i,0)=sources[i].x();
			dual_V(i,1)=sources[i].y();
			dual_V(i,2)=sources[i].z();
		}
		vector<Eigen::Vector3i> dual_fs;
		std::cout<<"infer start"<<std::endl;
		auto resultingField = InferOverPropagatedDistancesForLRVD(V,F, sources);
		std::cout<<"infer end"<<std::endl;

		vector<tuple<Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>>> triangles(F.rows());
		for (int i=0;i<triangles.size();++i) {
			Eigen::Vector3d p1=V.row(F(i,0));
			Eigen::Vector3d p2=V.row(F(i,1));
			Eigen::Vector3d p3=V.row(F(i,2));
			get<0>(triangles[i]) = Eigen::Vector<REAL,2>(0, 0);
			get<1>(triangles[i]) = Eigen::Vector<REAL,2>((p2-p1).norm(), 0);
			get<2>(triangles[i]) = Eigen::Vector<REAL,2>((p3-p1).dot((p2-p1).normalized()), sqrt((p3-p1).squaredNorm()-pow((p3-p1).dot((p2-p1).normalized()),2)));
		}

#pragma omp parallel for num_threads(8)
		for (int faceID = 0; faceID < F.rows(); ++faceID)
		{
			if (resultingField[faceID].size() <= 1)
			{
				continue;
			}
			Triangle_Base_Convex_Top tble;

			tble.init(triangles[faceID]);
			tble.faceID = faceID;

			for (int j = 0; j < resultingField[faceID].size(); ++j)
			{
				tble.addHyperPlane(Triangle_Base_Convex_Top::Hyperplane(tble.tri,get<1>(resultingField[faceID][j]),
					get<2>(resultingField[faceID][j]),
					get<3>(resultingField[faceID][j]),
					get<0>(resultingField[faceID][j])));
			}
			auto segs_face = tble.GetSegments(V,F);
			vector<Eigen::Vector3i> current_dual_fs;
			for (auto& v:tble.vertexs) {
				if (v.second.hplane[0]>=0&&v.second.hplane[1]>=0&&v.second.hplane[2]>=0) {
					current_dual_fs.push_back(Eigen::Vector3i(v.second.hplane[0],v.second.hplane[1],v.second.hplane[2]));
				}
			}
#pragma omp critical
			{
				copy(segs_face.begin(), segs_face.end(), back_inserter(segs));
				copy(current_dual_fs.begin(), current_dual_fs.end(), back_inserter(dual_fs));
			}
		}
		return make_pair(segs,make_pair(dual_V,transform_vectorf_to_eigenf(dual_fs)));
	}



	vector<vector<vector<Eigen::Vector3d>>> GetLRVD_Regions(const Eigen::MatrixXd& V,const Eigen::MatrixXi&F, const vector<Eigen::Vector3d>& sources, vector<vector<int>>& sites_of_face)
	{
		vector<vector<vector<Eigen::Vector3d>>> regions(sources.size());
		sites_of_face.resize(F.rows());

		Eigen::MatrixXd dual_V(sources.size(),3);
		for (int i=0;i<sources.size();++i) {
			dual_V(i,0)=sources[i].x();
			dual_V(i,1)=sources[i].y();
			dual_V(i,2)=sources[i].z();
		}
		vector<Eigen::Vector3i> dual_fs;
		std::cout<<"infer start"<<std::endl;
		auto resultingField = InferOverPropagatedDistancesForLRVD(V,F, sources);
		std::cout<<"infer end"<<std::endl;

		vector<tuple<Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>>> triangles(F.rows());
		for (int i=0;i<triangles.size();++i) {
			Eigen::Vector3d p1=V.row(F(i,0));
			Eigen::Vector3d p2=V.row(F(i,1));
			Eigen::Vector3d p3=V.row(F(i,2));
			get<0>(triangles[i]) = Eigen::Vector<REAL,2>(0, 0);
			get<1>(triangles[i]) = Eigen::Vector<REAL,2>((p2-p1).norm(), 0);
			get<2>(triangles[i]) = Eigen::Vector<REAL,2>((p3-p1).dot((p2-p1).normalized()), sqrt((p3-p1).squaredNorm()-pow((p3-p1).dot((p2-p1).normalized()),2)));
		}

#pragma omp parallel for num_threads(8)
		for (int faceID = 0; faceID < F.rows(); ++faceID)
		{
			Triangle_Base_Convex_Top tble;

			tble.init(triangles[faceID]);
			tble.faceID = faceID;

			for (int j = 0; j < resultingField[faceID].size(); ++j)
			{
				tble.addHyperPlane(Triangle_Base_Convex_Top::Hyperplane(tble.tri,get<1>(resultingField[faceID][j]),
					get<2>(resultingField[faceID][j]),
					get<3>(resultingField[faceID][j]),
					get<0>(resultingField[faceID][j])));
			}
			vector<pair<int,vector<Eigen::Vector3d>>> current_face = tble.GetVoronoiRegions(V,F);
			for (auto &v:current_face) {
				sites_of_face[faceID].push_back(v.first);
			}
#pragma omp critical
			{
				for (auto& v:current_face) {
					regions[v.first].push_back(v.second);
				}
			}
		}
		return regions;
	}



	pair<vector<vector<vector<Eigen::Vector3d>>>,vector<set<int>>> GetLRVD_Regions_for_MedialAxis(const Eigen::MatrixXd& V,const Eigen::MatrixXi&F, const vector<Eigen::Vector3d>& sources, vector<vector<int>>& sites_of_face)
	{
		vector<set<int>> neighbors_of_sites(sources.size());

		vector<vector<vector<Eigen::Vector3d>>> regions(sources.size());
		sites_of_face.resize(F.rows());

		Eigen::MatrixXd dual_V(sources.size(),3);
		for (int i=0;i<sources.size();++i) {
			dual_V(i,0)=sources[i].x();
			dual_V(i,1)=sources[i].y();
			dual_V(i,2)=sources[i].z();
		}
		vector<Eigen::Vector3i> dual_fs;
		std::cout<<"infer start"<<std::endl;
		auto resultingField = InferOverPropagatedDistancesForLRVD(V,F, sources);
		std::cout<<"infer end"<<std::endl;

		vector<tuple<Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>,Eigen::Vector<REAL,2>>> triangles(F.rows());
		for (int i=0;i<triangles.size();++i) {
			Eigen::Vector3d p1=V.row(F(i,0));
			Eigen::Vector3d p2=V.row(F(i,1));
			Eigen::Vector3d p3=V.row(F(i,2));
			get<0>(triangles[i]) = Eigen::Vector<REAL,2>(0, 0);
			get<1>(triangles[i]) = Eigen::Vector<REAL,2>((p2-p1).norm(), 0);
			get<2>(triangles[i]) = Eigen::Vector<REAL,2>((p3-p1).dot((p2-p1).normalized()), sqrt((p3-p1).squaredNorm()-pow((p3-p1).dot((p2-p1).normalized()),2)));
		}

#pragma omp parallel for num_threads(8)
		for (int faceID = 0; faceID < F.rows(); ++faceID)
		{
			Triangle_Base_Convex_Top tble;

			tble.init(triangles[faceID]);
			tble.faceID = faceID;

			for (int j = 0; j < resultingField[faceID].size(); ++j)
			{
				tble.addHyperPlane(Triangle_Base_Convex_Top::Hyperplane(tble.tri,get<1>(resultingField[faceID][j]),
					get<2>(resultingField[faceID][j]),
					get<3>(resultingField[faceID][j]),
					get<0>(resultingField[faceID][j])));
			}
			vector<pair<int,vector<Eigen::Vector3d>>> current_face = tble.GetVoronoiRegions(V,F);
			for (auto &v:current_face) {
				sites_of_face[faceID].push_back(v.first);
			}
#pragma omp critical
			{
				for (auto& v:tble.vertexs) {
					if (v.second.hplane[0]>=0&&v.second.hplane[1]>=0&&v.second.hplane[2]>=0) {
						int hp1=v.second.hplane[0],hp2=v.second.hplane[1],hp3=v.second.hplane[2];
						// neighbors_of_sites.insert(make_pair(min(v.second.hplane[0],v.second.hplane[1]),max(v.second.hplane[0],v.second.hplane[1])));
						// neighbors_of_sites.insert(make_pair(min(v.second.hplane[0],v.second.hplane[2]),max(v.second.hplane[0],v.second.hplane[2])));
						// neighbors_of_sites.insert(make_pair(min(v.second.hplane[1],v.second.hplane[2]),max(v.second.hplane[1],v.second.hplane[2])));
						neighbors_of_sites[hp1].insert(hp2);
						neighbors_of_sites[hp1].insert(hp3);

						neighbors_of_sites[hp2].insert(hp3);
						neighbors_of_sites[hp2].insert(hp1);

						neighbors_of_sites[hp3].insert(hp1);
						neighbors_of_sites[hp3].insert(hp2);
					}
				}

				for (auto& v:current_face) {
					regions[v.first].push_back(v.second);
				}
			}
		}
		return {regions,neighbors_of_sites};
	}
}
