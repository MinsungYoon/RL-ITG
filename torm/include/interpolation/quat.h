#ifndef _QUAT_H
#define _QUAT_H

#include <math.h>
#include <iostream>
#include <random>

class quat{
public:
    quat(){
        x=0;
        y=0;
        z=0;
        w=1;
    }
    quat(double x, double y, double z, double w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
    ~quat(){}

    void conjugate();
    void negative();
    void setRandomQuaternion();

    std::vector<double> euler();
    void print();

    double x;
    double y;
    double z;
    double w;
    double test{0.0};
};


inline std::vector<double> quat::euler(){
    std::vector<double> euler;
    euler.resize(3,0.0);
    const static double PI_OVER_2 = M_PI * 0.5;
    const static double EPSILON = 1e-10;
    double sqw, sqx, sqy, sqz;

    // quick conversion to Euler angles to give tilt to user
    sqw = w*w;
    sqx = x*x;
    sqy = y*y;
    sqz = z*z;

    test = 2.0 * (w*y - x*z);

    euler[1] = asin(test);
    if (PI_OVER_2 - fabs(euler[1]) > EPSILON) {
        euler[2] = atan2(2.0 * (x*y + w*z),
                         sqx - sqy - sqz + sqw);
        euler[0] = atan2(2.0 * (w*x + y*z),
                         sqw - sqx - sqy + sqz);
    } else {
        // compute heading from local 'down' vector
        euler[2] = atan2(2*y*z - 2*x*w,
                         2*x*z + 2*y*w);
        euler[0] = 0.0;

        // If facing down, reverse yaw
        if (euler[1] < 0)
            euler[2] = M_PI - euler[2];
    }
    return euler;
}

//inline std::vector<double> quat::euler(){
//    std::vector<double> euler;
//    euler.resize(3,0.0);
//    double sqw = w*w;
//    double sqx = x*x;
//    double sqy = y*y;
//    double sqz = z*z;
//    double unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
//    double test = x*y + z*w;
//
//    if (test > 0.4999999999*unit) { // singularity at north pole
//        euler[1] = 2 * atan2(x,w);
//        euler[2] = M_PI/2;
//        euler[0] = 0;
//        std::cout<< "111111111111111111111111111111111111111111111111111111111111111111111111111test: " << test<< std::endl;
//        return euler;
//    }
//    if (test < -0.4999999999*unit) { // singularity at south pole
//        euler[1] = -2 * atan2(x,w);
//        euler[2] = -M_PI/2;
//        euler[0] = 0;
//        std::cout<< "2222222222222222222222111122222222222222222222222222222222222222222222222222222222test: " << test<< std::endl;
//        return euler;
//    }
//    euler[1] = atan2(2*y*w-2*x*z , sqx - sqy - sqz + sqw);
//    euler[2] = asin(2*test/unit);
//    euler[0] = atan2(2*x*w-2*y*z , -sqx + sqy - sqz + sqw);
////    euler[1] = atan2(2*y*w-2*x*z , 1 - 2*sqy - 2*sqz);
////    euler[2] = asin(2*test/unit);
////    euler[0] = atan2(2*x*w-2*y*z , 1 - 2*sqx - 2*sqz);
//    return euler;
//}

inline void quat::conjugate(){
    x = -1*x;
    y = -1*y;
    z = -1*z;
}

inline void quat::negative(){
    x = -1*x;
    y = -1*y;
    z = -1*z;
    w = -1*w;
}

inline void quat::setRandomQuaternion() {
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<double> die(-1, 1);

    double x,y,z, u,v,w, s;
    do { x = die(mersenne); y = die(mersenne); z = x*x + y*y; } while (z > 1);
    do { u = die(mersenne); v = die(mersenne); w = u*u + v*v; } while (w > 1);
    s = sqrt((1-z) / w);
    this->x = x;
    this->y = y;
    this->z = s*u;
    this->w = s*v;
}

inline void quat::print(){
    std::cout<< "q: " << x << ", " << y << ", " << z << ", " << w << std::endl;
}

//// [euler(q) == euler(-q) test]: result: equal!
//while(true) {
//quat aa;
//aa.setRandomQuaternion();
//auto rpy1 = aa.euler();
//aa.negative();
//auto rpy2 = aa.euler();
//if(rpy1[0] != rpy2[0] || rpy1[1] != rpy2[1] || rpy1[2] != rpy2[2]){
//aa.print();
//std::cout << "[ERROR!] RPY1" << rpy1[0] << ", " << rpy1[1] << ", " << rpy1[2] << std::endl;
//std::cout << "[ERROR!] RPY2" << rpy2[0] << ", " << rpy2[1] << ", " << rpy2[2] << std::endl;
//}
//}

#endif


