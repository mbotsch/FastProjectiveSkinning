//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <set>

namespace Projective_Skinning
{
template < int Rows, int Cols, int Options = (Eigen::ColMajor) >
using Matrixf = Eigen::Matrix<float, Rows, Cols, Options>; ///< A typedef of the dense matrix of Eigen.
typedef Matrixf<2, 1> Vec2;								///< A 2d column vector.
typedef Matrixf<2, 2> Mat22;								///< A 2 by 2 matrix.
typedef Matrixf<2, 3> Mat23;								///< A 2 by 3 matrix.
typedef Matrixf<3, 1> Vec3;								///< A 3d column vector.
typedef Matrixf<3, 2> Mat32;								///< A 3 by 2 matrix.
typedef Matrixf<3, 3> Mat33;								///< A 3 by 3 matrix.
typedef Matrixf<3, 4> Mat34;								///< A 3 by 4 matrix.
typedef Matrixf<4, 1> Vec4;								///< A 4d column vector.
typedef Matrixf<4, 4> Mat44;								///< A 4 by 4 matrix.
typedef Matrixf<3, Eigen::Dynamic> Mat3X;				///< A 3 by n matrix.
typedef Matrixf<Eigen::Dynamic, 3> MatX3;				///< A n by 3 matrix.
typedef Matrixf<Eigen::Dynamic, 1> VecX;					///< A nd column vector.
typedef Matrixf<Eigen::Dynamic, Eigen::Dynamic> MatXX;	///< A n by m matrix.

//Sparse
template<int Options = Eigen::ColMajor>
using SparseMatrixT = Eigen::SparseMatrix<float, Options>;	///< A typedef of the sparse matrix of Eigen.
typedef SparseMatrixT<> SparseMatrix;						///< The default sparse matrix of Eigen.
typedef Eigen::Triplet<float> Triplet;						///< A triplet, used in the sparse triplet representation for matrices.

template < int Rows, int Cols, int Options = (Eigen::ColMajor) >
using Matrixd = Eigen::Matrix<double, Rows, Cols, Options>; ///< A typedef of the dense matrix of Eigen.
typedef Matrixd<2, 1> Vec2d;								///< A 2d column vector.
typedef Matrixd<2, 2> Mat22d;								///< A 2 by 2 matrix.
typedef Matrixd<2, 3> Mat23d;								///< A 2 by 3 matrix.
typedef Matrixd<3, 1> Vec3d;								///< A 3d column vector.
typedef Matrixd<3, 2> Mat32d;								///< A 3 by 2 matrix.
typedef Matrixd<3, 3> Mat33d;								///< A 3 by 3 matrix.
typedef Matrixd<3, 4> Mat34d;								///< A 3 by 4 matrix.
typedef Matrixd<4, 1> Vec4d;								///< A 4d column vector.
typedef Matrixd<4, 4> Mat44d;								///< A 4 by 4 matrix.
typedef Matrixd<3, Eigen::Dynamic> Mat3Xd;				///< A 3 by n matrix.
typedef Matrixd<Eigen::Dynamic, 3> MatX3d;				///< A n by 3 matrix.
typedef Matrixd<Eigen::Dynamic, 1> VecXd;					///< A nd column vector.
typedef Matrixd<Eigen::Dynamic, Eigen::Dynamic> MatXXd;	///< A n by m matrix.

typedef std::vector<unsigned int> IndexVector;
typedef std::set<unsigned int> IndexSet;

typedef unsigned int usint;

inline Vec3 getNearestLinePoint(const Vec3& p, const Vec3& v0, const Vec3& v1)
{
    Vec3 a = (v1 - v0);
    float L = a.norm();
    a/=L;
    float Lnew = a.dot(p - v0);
    Lnew = (Lnew > L)? L : Lnew;
    Lnew = (Lnew < 0)? 0 : Lnew;
    return v0 + Lnew*a;
}

}
