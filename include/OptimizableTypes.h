/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORB_SLAM3_OPTIMIZABLETYPES_H
#define ORB_SLAM3_OPTIMIZABLETYPES_H

#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>
#include <Thirdparty/g2o/g2o/types/sim3.h>

#include <Eigen/Geometry>
#include <include/CameraModels/GeometricCamera.h>


namespace ORB_SLAM3 {
class  EdgeSE3ProjectXYZOnlyPose: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZOnlyPose(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project(v1->estimate().map(Xw));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return (v1->estimate().map(Xw))(2)>0.0;
    }


    virtual void linearizeOplus();

    Eigen::Vector3d Xw;
    GeometricCamera* pCamera;
};

class  EdgeSE3ProjectXYZOnlyPoseToBody: public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZOnlyPoseToBody(){}

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project((mTrl * v1->estimate()).map(Xw));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(Xw))(2)>0.0;
    }


    virtual void linearizeOplus();

    Eigen::Vector3d Xw;
    GeometricCamera* pCamera;

    g2o::SE3Quat mTrl;
};

/**
 * @brief 单目边
 * VertexSBAPointXYZ 3D地图点 SBA是 Sparse Bundle Adjustment 稀疏捆绑调整的缩写 PointXYZ表征是一个xyz坐标点
 * VertexSE3Expmap 关键帧相机姿态 SE3表征是一个器流形特性 Expmap表示采用李代数的指数映射（Exponential Map）来进行优化（更新位姿）
 * Eigen::Vector2d 2D像素观测点的误差 对应的是地图点在相机上投影对应像素坐标的误差
 */
class  EdgeSE3ProjectXYZ: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZ();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project(v1->estimate().map(v2->estimate()));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        return ((v1->estimate().map(v2->estimate()))(2)>0.0);
    }

    virtual void linearizeOplus();

    GeometricCamera* pCamera;
};

class  EdgeSE3ProjectXYZToBody: public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZToBody();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    // v2->estimate() 地图点的三维坐标
    // v1->estimate() 当前关键帧的相机位姿
    // mTrl 是相机 body frame 到右目（或次相机）frame 的相对位姿
    // mTrl * v1->estimate() 将地图点变换到右目的坐标系下
    // .map(v2->estimate()) 将三维点投影到当前相机坐标系中
    // pCamera->project() 将三维点投影到像素坐标系中
    void computeError()  {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs-pCamera->project((mTrl * v1->estimate()).map(v2->estimate()));
    }

    bool isDepthPositive() {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        return ((mTrl * v1->estimate()).map(v2->estimate()))(2)>0.0;
    }

    virtual void linearizeOplus();

    GeometricCamera* pCamera;
    g2o::SE3Quat mTrl;
};

class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexSim3Expmap();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        _estimate = g2o::Sim3();
    }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<g2o::Vector7d> update(const_cast<double*>(update_));

        if (_fix_scale)
            update[6] = 0;

        g2o::Sim3 s(update);
        setEstimate(s*estimate());
    }

    GeometricCamera* pCamera1, *pCamera2;

    bool _fix_scale;
};


class EdgeSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, ORB_SLAM3::VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3ProjectXYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap* v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs-v1->pCamera1->project(v1->estimate().map(v2->estimate()));
    }

    // virtual void linearizeOplus();

};

class EdgeInverseSim3ProjectXYZ : public  g2o::BaseBinaryEdge<2, Eigen::Vector2d,  g2o::VertexSBAPointXYZ, VertexSim3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeInverseSim3ProjectXYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    void computeError()
    {
        const ORB_SLAM3::VertexSim3Expmap* v1 = static_cast<const ORB_SLAM3::VertexSim3Expmap*>(_vertices[1]);
        const g2o::VertexSBAPointXYZ* v2 = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);

        Eigen::Vector2d obs(_measurement);
        _error = obs-v1->pCamera2->project((v1->estimate().inverse().map(v2->estimate())));
    }

    // virtual void linearizeOplus();

};

}

#endif //ORB_SLAM3_OPTIMIZABLETYPES_H
