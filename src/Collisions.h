//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include <unordered_map>
#include <tuple>
#include <iostream>

#include "mesh/Helper.h"

namespace Projective_Skinning
{

class Collision_Detection
{
public:
    // initializer
    void init(const Mat3X& mesh_vertices, const IndexVector &tets, const Mat3X& shrinkV, unsigned int nV);

    // performs collision test of mesh
    void test_for_collsions(const Mat3X& mesh_vertices, const IndexVector &tets, const Mat3X& shrinkV);

    // setter and getter
    void set_cellsize(float newCellsize){cellsize_ = newCellsize;}
    float get_cellsize(){return cellsize_;}
private:

    // different versions of testing vertex in tetrahedron
    inline bool vertex_in_tet(Mat44 &T, Vec4& v, Vec3 &min, Vec3 &max);
    inline bool vertex_in_tet(Eigen::HouseholderQR<Mat44> &qr, Vec4& v, Vec3 &min, Vec3 &max);
    inline bool vertex_in_tet(Eigen::PartialPivLU<Mat33> &lu, Vec3& v, Vec3& x, Vec3 &min, Vec3 &max);
    inline bool vertex_in_tet(Mat44 &T, Vec4& v);

    // hash function
	inline long int hash(int x, int y, int z)
	{
		return (x * 18397) + (y * 20483) + (z * 29303);
	}
	
public:
    // stores colliding vertex and tet indices
    IndexVector colliding_vertices_;
    IndexVector colliding_tets_;
private:
    std::unordered_map<long int, IndexVector> hash_grid_;
	float cellsize_;
	unsigned int nV_;

    // used to prevent collisions with degenerated tets (opposite sign of volume)
    std::vector<float> tet_volume_signs_;
};

}
