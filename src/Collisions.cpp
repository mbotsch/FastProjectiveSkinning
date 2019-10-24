//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Collisions.h"
#include <cfloat>

namespace Projective_Skinning
{

void Collision_Detection::init(const Mat3X &mesh_vertices, const IndexVector &tets, const Mat3X &shrinkV, unsigned int nV)
{
    nV_ = nV;
    tet_volume_signs_.resize(tets.size()/4);

    // determine average tetrahedron edge and set this to cellsize
    float avgL = 0.0f;
    unsigned int anz = 0;
    for(unsigned int t = 0; t < tets.size(); t+= 4)
    {
        Vec3 p[4];
        for(int i = 0; i < 4; i++)
        {
            p[i] = (tets[t + i] < nV) ? mesh_vertices.col(tets[t + i]) : shrinkV.col(tets[t + i] - nV);
        }

        anz += 6;
        avgL += (p[0] - p[1]).norm();
        avgL += (p[0] - p[2]).norm();
        avgL += (p[0] - p[3]).norm();
        avgL += (p[1] - p[2]).norm();
        avgL += (p[1] - p[3]).norm();
        avgL += (p[2] - p[3]).norm();

        Mat33 T33;
        for(int i = 1; i < 4; i++)
        {
            T33.col(i - 1) = p[i] - p[0];
        }

        tet_volume_signs_[t/4] = (T33.determinant() >= 0.0f) ? 1.0f : -1.0f;
    }

    // 0.9 of avg edgelength heuristically performs better than 1.0
    cellsize_ = 0.9*avgL/(float)anz;
}


void Collision_Detection::test_for_collsions(const Mat3X& mesh_vertices, const IndexVector& tets, const Mat3X& shrinkV)
{
    // hash vertices
    hash_grid_.clear();
    for(unsigned int i = 0; i < nV_; i++)
    {
        Vec3 v = mesh_vertices.col(i)/cellsize_;
        hash_grid_[hash(static_cast<int>(floor(v(0))),static_cast<int>(floor(v(1))), static_cast<int>(floor(v(2))))].push_back(i);
    }

    // test tets for collisions

    colliding_vertices_.clear();
    colliding_tets_.clear();
    colliding_vertices_.resize(0);
    colliding_tets_.resize(0);

    std::vector<IndexVector> cv(tets.size()/4);

    #pragma omp parallel for
    for(unsigned int t = 0; t < tets.size(); t+=4)
    {
        usint ti[4];

        Mat33 T33;
        ti[0] = tets[t + 0];
        Vec3 A = (ti[0] >= nV_) ? shrinkV.col(ti[0] - nV_) : mesh_vertices.col(ti[0]);
        for(int i = 1; i < 4; i++)
        {
            ti[i] = tets[t + i];
            T33.col(i - 1) = (ti[i] >= nV_) ? shrinkV.col(ti[i] - nV_) : mesh_vertices.col(ti[i]);
        }

        Vec3 bb_min, bb_max;
        int bb_minI[3], bb_maxI[3];
        for(unsigned int i = 0; i < 3 ; i++)
        {
            bb_min(i) = std::min(A(i),(T33.row(i)).minCoeff());
            bb_max(i) = std::max(A(i),(T33.row(i)).maxCoeff());

            bb_minI[i] = static_cast<int>(floor(bb_min(i)/cellsize_));
            bb_maxI[i] = static_cast<int>(floor(bb_max(i)/cellsize_));
        }
        T33.col(0) -= A;
        T33.col(1) -= A;
        T33.col(2) -= A;

        // do not test degenerated tetrahedra
        if(T33.determinant()*tet_volume_signs_[t/4] <= 0.0f)
            continue;

        // make sure that you do qr decomposition just once
        bool first = true;
        Eigen::PartialPivLU<Mat33> lu;

        for(int x = bb_minI[0] ; x <= bb_maxI[0];x++)
            for(int y = bb_minI[1] ; y <= bb_maxI[1];y++)
                for(int z = bb_minI[2] ; z <= bb_maxI[2];z++)
                {
                    auto it = hash_grid_.find(hash(x,y,z));
                    if(it != hash_grid_.end())
                    {
                        for(unsigned int i = 0; i < (it->second).size(); i++)
                        {
                            unsigned int vi = (it->second)[i];

                            if(vi != ti[0] && vi != ti[1] && vi != ti[2] && vi != ti[3]) //avoid selfcollisions
                            {
                                if(first)
                                {
                                    lu.compute(T33);
                                    first = false;
                                }
                                Vec3 v = mesh_vertices.col(vi);
                                Vec3 x = v - A;

                                if(vertex_in_tet(lu, v, x, bb_min, bb_max))
                                {
                                    cv[t/4].push_back(vi);
                                }
                            }
                        }
                    }
                }

    }

    // reduce map
    for(usint i = 0; i < cv.size(); i++)
    {
        for(usint j = 0; j < cv[i].size(); j++)
        {
            colliding_vertices_.push_back(cv[i][j]);
            colliding_tets_.push_back(i);
        }
    }
}

bool Collision_Detection::vertex_in_tet(Mat44& T, Vec4& v)
{
    Vec4 barycenter;
    barycenter = T.colPivHouseholderQr().solve(v);
    if(barycenter.minCoeff() > 0)
        return true;
    else
        return false;
}

bool Collision_Detection::vertex_in_tet(Mat44& T, Vec4& v, Vec3 &min, Vec3 &max)
{
	
	//std::cout << v.transpose() << "\n" << T.transpose() << std::endl << std::endl; 
	if(v(0) < min(0) || v(1) < min(1) || v(2) < min(2) || v(0) > max(0) || v(1) > max(1) || v(2) > max(2)) 
		return false;
	Vec4 barycenter;
	barycenter = T.colPivHouseholderQr().solve(v);
	if(barycenter.minCoeff() > 0) 
		return true;
	else
		return false;
}

bool Collision_Detection::vertex_in_tet(Eigen::HouseholderQR<Mat44> &qr, Vec4& v, Vec3 &min, Vec3 &max)
{

    //std::cout << v.transpose() << "\n" << T.transpose() << std::endl << std::endl;
    if(v(0) < min(0) || v(1) < min(1) || v(2) < min(2) || v(0) > max(0) || v(1) > max(1) || v(2) > max(2))
        return false;
    Vec4 barycenter;
    barycenter = qr.solve(v);
    if(barycenter.minCoeff() > 0)
        return true;
    else
        return false;
}

bool Collision_Detection::vertex_in_tet(Eigen::PartialPivLU<Mat33> &lu, Vec3& v, Vec3& x, Vec3 &min, Vec3 &max)
{

    //std::cout << v.transpose() << "\n" << T.transpose() << std::endl << std::endl;
    if(v(0) < min(0) || v(1) < min(1) || v(2) < min(2) || v(0) > max(0) || v(1) > max(1) || v(2) > max(2))
        return false;
    Vec3 barycenter;
    barycenter = lu.solve(x);
    float delta = 1.0 - barycenter(0) - barycenter(1) - barycenter(2);
    if(barycenter.minCoeff() > 0 && delta > 0)
        return true;
    else
        return false;
}

}

