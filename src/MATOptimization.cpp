//
// Created by pengfei on 2025/12/23.
//
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <LBFGSB.h> // 使用 LBFGSpp 的带约束版本

// 定义简单的输入结构
struct SurfacePoint {
    Eigen::Vector3d p; // 坐标
    double weight;     // AS_i
};

struct Plane {
    Eigen::Vector3d normal; // a, b, c
    double d;               // d
    double weight;          // SP_i
};

// 优化参数包
struct OptimizationConfig {
    double lambda_sqem = 1.0;   // 拟合项权重
    double lambda_triangle = 0.1;   // 均匀项权重
    double epsilon = 1e-8;  // 防止根号下为0的平滑项
};

class MATOptimization {
private:
    const std::vector<SurfacePoint>& points_;
    const std::vector<Plane>& planes_;
    const std::vector<Eigen::Vector3d>& neighbors_;
    OptimizationConfig config_;

    double w_uniform_;

public:
    MATOptimization(
        const std::vector<SurfacePoint>& points,
        const std::vector<Plane>& planes,
        const std::vector<Eigen::Vector3d>& neighbors,
        const OptimizationConfig& config
    ) : points_(points), planes_(planes), neighbors_(neighbors), config_(config)
    {
        if (!neighbors_.empty()) {
            w_uniform_ = config_.lambda_triangle / static_cast<double>(neighbors_.size());
        } else {
            w_uniform_ = 0.0;
        }
    }

    // 核心优化函数 (保持不变，只计算总能量和梯度)
    double operator()(const Eigen::VectorXd& x_aug, Eigen::VectorXd& grad) {
        Eigen::Vector3d x = x_aug.head(3);
        double r = x_aug(3);

        double energy = 0.0;
        grad.setZero();

        // --- 1. Surface Points ---
        for (const auto& sp : points_) {
            Eigen::Vector3d diff = x - sp.p;
            double sq_dist = diff.squaredNorm();
            double smooth_dist = std::sqrt(sq_dist + config_.epsilon);
            double residual = smooth_dist - r;
            double weighted_coeff = config_.lambda_sqem * sp.weight;

            energy += weighted_coeff * residual * residual;

            double common_factor = 2.0 * weighted_coeff * residual;
            grad.head(3) += common_factor * (diff / smooth_dist);
            grad(3) -= common_factor;
        }

        // --- 2. Tangent Planes ---
        for (const auto& pl : planes_) {
            double signed_dist = pl.normal.dot(x) + pl.d;
            double residual = signed_dist - r;
            double weighted_coeff = config_.lambda_sqem * pl.weight;

            energy += weighted_coeff * residual * residual;

            double common_factor = 2.0 * weighted_coeff * residual;
            grad.head(3) += common_factor * pl.normal;
            grad(3) -= common_factor;
        }

        // --- 3. Uniformity ---
        for (const auto& v : neighbors_) {
            Eigen::Vector3d diff = x - v;
            energy += w_uniform_ * diff.squaredNorm();
            grad.head(3) += 2.0 * w_uniform_ * diff;
        }

        return energy;
    }

    // ==========================================
    // [新增] 专门用于计算分项能量的辅助函数
    // ==========================================
    std::pair<double, double> computeComponents(const Eigen::Vector3d& x, double r) {
        double e_fit = 0.0;
        double e_uni = 0.0;

        // 1. 计算拟合项 (点 + 面)
        for (const auto& sp : points_) {
            double smooth_dist = std::sqrt((x - sp.p).squaredNorm() + config_.epsilon);
            double residual = smooth_dist - r;
            e_fit += sp.weight * residual * residual;
        }
        for (const auto& pl : planes_) {
            double signed_dist = pl.normal.dot(x) + pl.d;
            double residual = signed_dist - r;
            e_fit += pl.weight * residual * residual;
        }

        // 2. 计算均匀项
        for (const auto& v : neighbors_) {
            e_uni += (x - v).squaredNorm();
        }
        e_uni/=neighbors_.size();
        return {e_fit, e_uni};
    }
};

struct OptimizationResult_unstable {
    Eigen::Vector3d opt_x;
    double opt_r;
    double final_energy;      // 总能量
    double energy_fitting;    // 第一部分：拟合能量 (点 + 面)
    double energy_uniform;    // 第二部分：均匀化能量
    bool success;
};

OptimizationResult_unstable optimize_medial_sphere(
    const std::vector<SurfacePoint>& points,
    const std::vector<Plane>& planes,
    const std::vector<Eigen::Vector3d>& neighbors,
    Eigen::Vector3d init_x,
    double init_r,
    OptimizationConfig config
) {
    LBFGSpp::LBFGSBParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    LBFGSpp::LBFGSBSolver<double> solver(param);


    Eigen::VectorXd x_aug(4);
    x_aug << init_x(0), init_x(1), init_x(2), init_r;

    Eigen::VectorXd lb(4), ub(4);
    lb.head(3).setConstant(-std::numeric_limits<double>::infinity());
    ub.head(3).setConstant(std::numeric_limits<double>::infinity());
    lb(3) = 1e-6; // r > 0
    ub(3) = std::numeric_limits<double>::infinity();

    MATOptimization fun(points, planes, neighbors, config);

    double final_energy;
    try {
        solver.minimize(fun, x_aug, final_energy, lb, ub);
    } catch (std::exception& e) {
        return {init_x, init_r, -1.0, 0.0, 0.0, false};
    }

    // ============ [新增] 获取最优解并拆分能量 ============
    Eigen::Vector3d opt_x = x_aug.head(3);
    double opt_r = x_aug(3);

    // 调用辅助函数计算分项
    std::pair<double, double> components = fun.computeComponents(opt_x, opt_r);

    return {
        opt_x,
        opt_r,
        final_energy,
        components.first,  // energy_fitting
        components.second, // energy_uniform
        true
    };
}
