#include <interpolation/interpolator6D.h>

interpolator6D::interpolator6D(std::vector<std::vector<double>>& wps, double interval_length)
:wps_(wps), interval_length_(interval_length){

    n_wps_ = wps.size();
    idx_wps_.reserve(n_wps_);
    idx_wps_.push_back(0);

    // length between waypoints
    len_intervals_.reserve(n_wps_-1);
    n_intervals_.reserve(n_wps_-1);
    for(int i=0; i<n_wps_-1; i++) {
        std::vector<double> dif;
        dif.reserve(3);
        dif.push_back(wps[i+1][0] - wps[i][0]);
        dif.push_back(wps[i+1][1] - wps[i][1]);
        dif.push_back(wps[i+1][2] - wps[i][2]);
        double rot_len = fabs(wps[i+1][3]*wps[i][3] + wps[i+1][4]*wps[i][4] + wps[i+1][5]*wps[i][5] + wps[i+1][6]*wps[i][6]);
        rot_len = acos(rot_len)*2;
        double len = 0.;
        for (auto df_i: dif) {
            len += df_i * df_i;
        }
        len = sqrt(len);
        // std::cout << "len: " << len << ", rot_len: " << rot_len << std::endl;
        len = len + rot_len;
        len_intervals_.push_back(len);

        if(floor(len/interval_length)*interval_length == len){
            n_intervals_.push_back(floor(len/interval_length)-1);
        }else{
            n_intervals_.push_back(floor(len/interval_length));
        }

        int idx = i+1;
        for(int j=0; j<=i; j++){
            idx += n_intervals_[j];
        }
        idx_wps_.push_back(idx);
    }

    // prepare ingredients for interpolation
    t_.reserve(n_wps_);
    tx_.reserve(n_wps_);
    ty_.reserve(n_wps_);
    tz_.reserve(n_wps_);
    x_.reserve(n_wps_);
    y_.reserve(n_wps_);
    z_.reserve(n_wps_);
    quats_.reserve(n_wps_);
    for(int i=0; i<n_wps_; i++){
        t_.push_back(i);
        tx_.push_back(i);
        ty_.push_back(i);
        tz_.push_back(i);
        x_.push_back(wps[i][0]);
        y_.push_back(wps[i][1]);
        z_.push_back(wps[i][2]);
        quat q(wps[i][3], wps[i][4], wps[i][5], wps[i][6]);
        if(i>0){
            double cosHalfTheta = quats_[i-1].x*q.x + quats_[i-1].y*q.y + quats_[i-1].z*q.z + quats_[i-1].w*q.w;
            if(cosHalfTheta<0){
                q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w;
            }
        }
        quats_.push_back(q);
    }

    x_spline_ = new makima<std::vector<double>>(std::move(tx_), std::move(x_));
    y_spline_ = new makima<std::vector<double>>(std::move(ty_), std::move(y_));
    z_spline_ = new makima<std::vector<double>>(std::move(tz_), std::move(z_));

    // for output
    total_len_ = std::accumulate(len_intervals_.begin(), len_intervals_.end(), 0.0);
    total_n_ = std::accumulate(n_intervals_.begin(), n_intervals_.end(), 0) + n_wps_;
    interpolated_path_.reserve(total_n_);
    interpolated_rpy_path_.reserve(total_n_);

    interpolate();
    to_rpy();
}
interpolator6D::~interpolator6D(){
}

std::vector<double> interpolator6D::get_inter_point(int i, double t){
    std::vector<double> p;
    p.reserve(7);
    p.push_back((*x_spline_)(t_[i]+t));
    p.push_back((*y_spline_)(t_[i]+t));
    p.push_back((*z_spline_)(t_[i]+t));
    quat q = slerp(quats_[i], quats_[i+1], t);
    p.push_back(q.x);
    p.push_back(q.y);
    p.push_back(q.z);
    p.push_back(q.w);
    return p;
}

void interpolator6D::interpolate(){
    for(int i=0; i<n_wps_-1; i++){
        for(int j=0; j<=n_intervals_[i]; j++){
            interpolated_path_.push_back( get_inter_point(i,(static_cast<double>(j)*interval_length_)/len_intervals_[i]) );
        }
    }
    std::vector<double> p;
    p.reserve(7);
    p.push_back((*x_spline_)(t_[n_wps_-1]));
    p.push_back((*y_spline_)(t_[n_wps_-1]));
    p.push_back((*z_spline_)(t_[n_wps_-1]));
    p.push_back(quats_[n_wps_-1].x);
    p.push_back(quats_[n_wps_-1].y);
    p.push_back(quats_[n_wps_-1].z);
    p.push_back(quats_[n_wps_-1].w);
    interpolated_path_.push_back(p);
}

std::vector<std::vector<double>>& interpolator6D::get_result(){
    return interpolated_path_;
}
std::vector<std::vector<double>>& interpolator6D::get_rpy_result(){
    return interpolated_rpy_path_;
}
std::vector<int>& interpolator6D::get_wps_idx(){
    return idx_wps_;
}
int interpolator6D::get_total_n(){
    return total_n_;
}
double interpolator6D::get_total_len(){
    return total_len_;
}


quat interpolator6D::slerp(quat qa, quat qb, double t) {
    // quaternion to return
    quat* qm = new quat();
    // Calculate angle between them.
    double cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
    // if qa=qb or qa=-qb then theta = 0 and we can return qa
    if (cosHalfTheta < 0) {
        qb.w = -qb.w; qb.x = -qb.x; qb.y = -qb.y; qb.z = qb.z;
        cosHalfTheta = -cosHalfTheta;
    }
    if (abs(cosHalfTheta) >= 1.0){
        qm->w = qa.w; qm->x = qa.x; qm->y = qa.y; qm->z = qa.z;
        return *qm;
    }
    // Calculate temporary values.
    double halfTheta = acos(cosHalfTheta);
    double sinHalfTheta = sqrt(1.0 - cosHalfTheta*cosHalfTheta);
    // if theta = 180 degrees then result is not fully defined
    // we could rotate around any axis normal to qa or qb
    if (fabs(sinHalfTheta) < 0.001){ // fabs is floating point absolute
        qm->w = (qa.w * 0.5 + qb.w * 0.5);
        qm->x = (qa.x * 0.5 + qb.x * 0.5);
        qm->y = (qa.y * 0.5 + qb.y * 0.5);
        qm->z = (qa.z * 0.5 + qb.z * 0.5);
        return *qm;
    }
    double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
    double ratioB = sin(t * halfTheta) / sinHalfTheta;
    //calculate Quaternion.
    qm->w = (qa.w * ratioA + qb.w * ratioB);
    qm->x = (qa.x * ratioA + qb.x * ratioB);
    qm->y = (qa.y * ratioA + qb.y * ratioB);
    qm->z = (qa.z * ratioA + qb.z * ratioB);
    return *qm;
}

void interpolator6D::print(){
    std::cout << "[ " << std::setw(11) << "px, " << std::setw(11) << "py, " << std::setw(11) << "pz, "
    << std::setw(11) << "qx, " << std::setw(11) << "qy, "
    << std::setw(11) << "qz, " << std::setw(11) << "qw, ]" << std::endl;
    for(int i=0; i<total_n_; i++){
        std::cout << "[ ";
        for(int j=0; j<7; j++){
            std::cout << std::setw(11) << interpolated_path_[i][j] << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

void interpolator6D::rpy_print(){
    std::cout << "[ " << std::setw(11) << "px, " << std::setw(11) << "py, " << std::setw(11) << "pz, "
              << std::setw(11) << "r, " << std::setw(11) << "p, "
              << std::setw(11) << "y, ]" << std::endl;
    for(int i=0; i<total_n_; i++){
        std::cout << "[ ";
        for(int j=0; j<6; j++){
            std::cout << std::setw(11) << interpolated_rpy_path_[i][j] << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}

void interpolator6D::to_rpy(){
    std::vector<double> prev_rpy;
    for(int i=0; i<total_n_; i++){
        std::vector<double> p;
        p.reserve(6);
        p.push_back(interpolated_path_[i][0]);
        p.push_back(interpolated_path_[i][1]);
        p.push_back(interpolated_path_[i][2]);
        quat q(interpolated_path_[i][3], interpolated_path_[i][4], interpolated_path_[i][5], interpolated_path_[i][6]);
        std::vector<double> rpy = q.euler();
        if(q.test >= 1.0){
            rpy[1] = M_PI/2;
            rpy[0] = prev_rpy[0];
            rpy[2] = prev_rpy[2];
        }
        if(q.test <= -1.0){
            rpy[1] = -M_PI/2;
            rpy[0] = prev_rpy[0];
            rpy[2] = prev_rpy[2];
        }
        p.push_back(rpy[0]);
        p.push_back(rpy[1]);
        p.push_back(rpy[2]);
        interpolated_rpy_path_.push_back(p);
        prev_rpy = std::move(rpy);
    }
}