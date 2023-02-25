#ifndef _SLERP_H
#define _SLERP_H

#include <utility>
#include <vector>
#include <iostream>
#include <math.h>
#include <numeric>
#include <iomanip>

#include <interpolation/quat.h>
#include <boost/math/interpolators/cubic_b_spline.hpp>
using boost::math::cubic_b_spline;

class interpolator6D{
public:
    interpolator6D(std::vector<std::vector<double>>& wps, double interval_length=0.1, std::string mode="spline");
    ~interpolator6D();

    quat slerp(quat qa, quat qb, double t);

    std::vector<double> get_spline_inter_point(int i, double t);
    std::vector<double> get_linear_inter_point(int i, double t);

    void interpolate();

    void to_rpy();
    void refine_path_interval();

    std::vector<std::vector<double>>& get_result();
    std::vector<std::vector<double>>& get_rpy_result();
    std::vector<int>& get_wps_idx();
    int get_total_n();
    double get_total_len();

    void print();
    void rpy_print();

private:
    std::string mode_;

    double interval_length_;
    double fineinterval_length_;
    std::vector<double> len_intervals_;
    std::vector<int> n_intervals_;

    std::vector<std::vector<double>>& wps_;
    int n_wps_;
    std::vector<int> idx_wps_;

    std::vector<quat> quats_;
    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> z_;


    cubic_b_spline<double> x_spline_;
    cubic_b_spline<double> y_spline_;
    cubic_b_spline<double> z_spline_;

    double total_len_;
    int total_n_;
    std::vector<std::vector<double>> interpolated_path_;
    std::vector<std::vector<double>> interpolated_rpy_path_;
};

#endif //_SLERP_H
