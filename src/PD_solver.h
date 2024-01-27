//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include "mesh/Mesh.h"
#include "svd/svd3x3.h" // todo: move this to vectypes

namespace Projective_Skinning
{
// base class of Projective Dynamics Solvers
class PD_Solver
{
public:
    PD_Solver();
    virtual ~PD_Solver();

    // initializes matrices and vectors
    virtual void init(Mesh *mesh,
                      const float tetweight, const float anchorweight, const float,
                      const float timestep, const float damping, const bool soft_bc);

    // does one projective dynamics iteration with `iteration` local/global iterations
    virtual void update_skin(int iterations);
    virtual void reset();

    virtual void update_anchors();
    virtual void update_HR(float*, float*); // todo: move upsampling variables to this class
    virtual void update_normals(bool);

    virtual void update_ogl_sim_mesh_buffers(float*, float*, size_t){}
    virtual void reinit(const std::vector<int>&, const int){}

protected:

    // local global update functions
    virtual void update_local();
    virtual void update_global();

    Mesh *mesh_;

    std::vector<Eigen::Triplet<float>> A_triplets_;
    SparseMatrix S_, At_, A_bc_;

    Mat3X old_points_, projections_, velocity_, momentum_, boundary_, rest_edge_inverse_;
    Mat3X* p_points_;

    VecX masses_;

    Eigen::SimplicialLDLT<SparseMatrix> solver_;

    int num_non_boundary_;
    int num_points_;
    int num_tets_;
    int num_projections_;

    IndexVector tets_;

    std::vector<float> tetweights_;
    std::vector<float> per_vertex_damping_;

    // store squareroot of weights for different constraints
    float weight_tet_;
    float weight_anc_;

    float dt_;
    float damping_;
};


// PD Solver including the possibility to use collision constraints
class PD_Solver_Collisions : public PD_Solver
{
public:

    virtual void init(Mesh *mesh,
                      const float tetweight, const float anchorweight, const float collisionweight,
                      const float timestep, const float damping, const bool soft_bc) override;

    virtual void reinit(const std::vector<int> &coltrigs, const int num_ti) override;

    virtual void update_anchors() override;

private:

    // local global update functions
    virtual void update_local();

    // additional members
    std::vector<int> collisions_indices_;
    int num_non_collision_projections_;
    int num_ti_;
    float weight_col_;

};

}
