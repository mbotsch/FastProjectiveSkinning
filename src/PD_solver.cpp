//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "PD_solver.h"

namespace Projective_Skinning
{

PD_Solver::PD_Solver()
    :mesh_(nullptr)
{

}

PD_Solver::~PD_Solver()
{
    mesh_ = nullptr;
}

void PD_Solver::init(Mesh *mesh,
                     const float tetweight, const float anchorweight, const float,
                     const float timestep, const float damping, const bool soft_bc)
{
    mesh_ = mesh;
    dt_ = timestep;
    damping_ = damping/timestep;
    p_points_ = &mesh_->vertices_;
    old_points_ = *p_points_;

    tets_ = mesh_->tets_.indices;

    weight_tet_ = sqrt(tetweight);
    weight_anc_ = sqrt(anchorweight);

    num_non_boundary_   = soft_bc ? mesh_->num_non_rigid_ + mesh_->num_rigid_ : mesh_->num_non_rigid_;
    num_tets_           = tets_.size()/4;
    num_projections_    = soft_bc ? num_tets_*3 + mesh_->num_rigid_ : num_tets_*3;
    num_points_         = mesh_->num_non_rigid_ + mesh_->num_rigid_;

    velocity_.resize(3,num_non_boundary_);
    momentum_.resize(3,num_non_boundary_);

    projections_.resize(3, num_projections_);
    boundary_.resize(3,num_projections_);

    velocity_.setZero();
    momentum_.setZero();
    projections_.setZero();
    boundary_.setZero();

    per_vertex_damping_.resize(mesh_->num_simulated_skin_, damping_);
    for(auto i : mesh_->additional_anchors_) per_vertex_damping_[i] = 0;

    rest_edge_inverse_.resize(3,3*num_tets_);
    tetweights_.resize(num_tets_);
    std::vector<Eigen::Triplet<float>> triplets;

    std::vector<float> weightfactors;
    mesh_->adjust_tet_weights_(weightfactors);
    for(int t = 0; t < num_tets_; t++)
    {
        Mat33 edges;
        for(int i = 1; i < 4; i++)
        {
            edges.col(i - 1) = mesh_->orig_vertices_.col(tets_[4*t + i]) - mesh_->orig_vertices_.col(tets_[4*t]);
        }
        tetweights_[t] = weight_tet_*sqrt(weightfactors[tets_[4*t]%mesh_->num_simulated_skin_]*fabs(edges.determinant()/6.0));
        rest_edge_inverse_.block<3,3>(0,3*t) = edges.inverse();

        Projective_Skinning::Mat34 P0Substract;
        P0Substract <<
        -1,1,0,0,
        -1,0,1,0,
        -1,0,0,1;

        P0Substract = tetweights_[t]*rest_edge_inverse_.block<3,3>(0,3*t).transpose()*P0Substract;

        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                triplets.push_back(Eigen::Triplet<float>(3*t + i, tets_[4*t + j], P0Substract(i,j)));
            }
        }
    }
    for(int i = mesh_->num_non_rigid_; i < num_non_boundary_; i++)
    {
        triplets.push_back(Eigen::Triplet<float>(3*num_tets_ + i - mesh_->num_non_rigid_, i, weight_anc_));
    }


    SparseMatrix A;
    std::vector<Eigen::Triplet<float>> tripletsBC;

    A.resize(num_projections_,num_non_boundary_);
    for(unsigned int i = 0; i < triplets.size(); i++)
    {
        if(triplets[i].col() < num_non_boundary_ &&
                (mesh_->additional_anchors_.find(triplets[i].col()) == mesh_->additional_anchors_.end() || soft_bc))
            A_triplets_.push_back(triplets[i]);
        else if(num_non_boundary_ < num_points_)
        {
            tripletsBC.push_back(triplets[i]);
        }
    }
    A.setFromTriplets(A_triplets_.begin(), A_triplets_.end());

    A_bc_.resize(num_projections_, num_points_);
    A_bc_.setFromTriplets(tripletsBC.begin(), tripletsBC.end());

    At_ = A.transpose();

    masses_.resize(num_non_boundary_);
    masses_.setZero();

    for(int i = 0; i < mesh_->num_simulated_skin_; i++)
    {
        masses_(i) = mesh_->vertex_masses_[i]/(dt_*dt_);
    }
    Projective_Skinning::SparseMatrix M(masses_.asDiagonal());

    S_ = At_*A + M;

    solver_.compute(S_);
}

void PD_Solver::update_skin(int iterations)
{
    #pragma omp parallel for
    for(int i = 0; i < mesh_->num_simulated_skin_; i++)
    {
        old_points_.col(i) = p_points_->col(i);
        p_points_->col(i) += dt_*velocity_.col(i);
        momentum_.col(i) = masses_(i)*p_points_->col(i);
    }

    if(num_non_boundary_ != num_points_)
    {
        #pragma omp parallel for
        for(int i = 0; i < 3; i++)
            boundary_.row(i) = (A_bc_* p_points_->block(i,0,1,num_points_).transpose()).transpose();
    }
    else
    {
        #pragma omp parallel for
        for(int i = mesh_->base_rigid_; i < num_non_boundary_; i++)
        {
            projections_.col(3*num_tets_ + i - mesh_->base_rigid_) = weight_anc_*mesh_->vertices_.col(i);
        }
    }

    for(int i = 0; i < iterations; i++)
    {
        update_local();
        update_global();
    }

    #pragma omp parallel for
    for(int i = 0; i < mesh_->num_simulated_skin_; i++)
    {
        velocity_.col(i) = (p_points_->col(i) - old_points_.col(i))*per_vertex_damping_[i];
    }
}

void PD_Solver::reset()
{
    mesh_->reset(true);
    velocity_.setZero();
}

void PD_Solver::update_anchors()
{
    mesh_->update_anchors();
}

void PD_Solver::update_HR(float *, float *)
{
    mesh_->upsample();
}

void PD_Solver::update_normals(bool)
{
    mesh_->compute_normals();
}

void PD_Solver::update_local()
{
    #pragma omp parallel for
    for(int t = 0; t < num_tets_; t++)
    {
        Mat33 edges;
        for(int i = 1; i < 4; i++)
        {
            edges.col(i - 1) = p_points_->col(tets_[4*t + i]) - p_points_->col(tets_[4*t]);
        }

        Mat33 F = edges * rest_edge_inverse_.block<3,3>(0,3*t);

        // rotation extraction
        if(F.determinant() < 1e-6)
        {
            Mat33 U, V;
            Vec3 S2;
            svd3x3(F, U, S2, V);
            F = U*V.transpose();
        }
        else
        {
            Mat33 Rlast(F);
            int k = - 1;
            do
            {
                k++;
                Mat33 Y(F.inverse());
                Rlast = F;
                F = 0.5*(F + Y.transpose());
            }while((F - Rlast).squaredNorm() > 1e-5 && k<20);
        }

        projections_.block<3,3>(0,3*t) = tetweights_[t]*F - boundary_.block<3,3>(0,3*t);
    }
}

void PD_Solver::update_global()
{    
    #pragma omp parallel for
    for(int i = 0; i < 3; i++)
    {
        p_points_->block(i,0,1,mesh_->num_non_rigid_) =
                solver_.solve(At_*(projections_.row(i).transpose()) + momentum_.row(i).transpose()).topRows(mesh_->num_non_rigid_).transpose();
    }
}


//--------------------------------------------------------------------------------------------------------------------------------------------


void PD_Solver_Collisions::init(Mesh *mesh,
                                const float tetweight, const float anchorweight, const float collisionweight,
                                const float timestep, const float damping, const bool soft_bc)
{
    PD_Solver::init(mesh,tetweight,anchorweight, collisionweight,timestep,damping,soft_bc);

    weight_col_ = sqrt(collisionweight);
    num_non_collision_projections_ = num_projections_;
}


void PD_Solver_Collisions::reinit(const std::vector<int> &collisions_indices, const int num_ti)
{
    collisions_indices_ = collisions_indices;

    // add collision triplets
    int row = num_non_collision_projections_;
    std::vector<Eigen::Triplet<float>> triplets = A_triplets_;
    for(int i = 0; i < (int)collisions_indices_.size(); i+= 4)
    {
        for(int j = 0; j < 3; j++)
        {
            triplets.push_back(Eigen::Triplet<float>(row,collisions_indices_[i + j + 1], weight_col_));
            if(i < 4*num_ti)
            {
                // translation invariant constraints
                triplets.push_back(Eigen::Triplet<float>(row,collisions_indices_[i], -weight_col_));
            }
            row++;
        }
    }

    // resize
    num_projections_ = row;
    num_ti_ = num_ti;
    projections_.resize(3,num_projections_);

    // reinit A
    SparseMatrix A(num_projections_,num_non_boundary_);
    A.setFromTriplets(triplets.begin(), triplets.end());
    At_ = A.transpose();

    // reinit S and its factorization
    SparseMatrix M(masses_.asDiagonal());
    S_ = At_*A + M;
    solver_.compute(S_);
}

void PD_Solver_Collisions::update_anchors()
{
    PD_Solver::update_anchors();
    mesh_->transform_collision_tet_basepoints_();
}

void PD_Solver_Collisions::update_local()
{
    PD_Solver::update_local();

    #pragma omp parallel for
    for(int i = 0; i < (int)collisions_indices_.size(); i+= 4)
    {
        Vec3 p = p_points_->col(collisions_indices_[i]);
        Vec3 p1 = p_points_->col(collisions_indices_[i + 1]);
        Vec3 p2 = p_points_->col(collisions_indices_[i + 2]);
        Vec3 p3 = p_points_->col(collisions_indices_[i + 3]);

        // check on which side of triangle normal the vertex is
        Vec3 tp = p - p1;
        Vec3 n = (p2 - p1).cross(p3 - p1);
        float dot = tp.dot(n);
        if(dot < 0)
        {
            // shift vertices if it is on wrong side
            Vec3 shift = dot*n/n.dot(n);
            p1 += shift; p2 += shift; p3 += shift;
        }
        p*= weight_col_; p1*= weight_col_; p2*=weight_col_; p3*=weight_col_;

        // store position or vector if translation invariance is desired
        projections_.col(num_non_collision_projections_ + 3*(i/4) + 0) = (i < 4*num_ti_) ? p1 - p : p1;
        projections_.col(num_non_collision_projections_ + 3*(i/4) + 1) = (i < 4*num_ti_) ? p2 - p : p2;
        projections_.col(num_non_collision_projections_ + 3*(i/4) + 2) = (i < 4*num_ti_) ? p3 - p : p3;
    }
}

}
