//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

//== INCLUDES =================================================================

#include "Inverse_kinematics.h"
#include <Eigen/QR>


//== IMPLEMENTATION ==========================================================
using namespace Projective_Skinning;

Inverse_kinematics::Inverse_kinematics(Skeleton& skeleton, float damping)
    : skeleton_(skeleton), lambda_(damping)
{
}


//----------------------------------------------------------------------------


void Inverse_kinematics::set_effector(int _joint_idx, const Vec3& _target)
{
    // set effector status for specified joint
    Joint *joint = skeleton_.joints_[_joint_idx];
    if (joint->is_root_)
    {
        std::cout << "Cannot compute IK for root joint.\n";
        return;
    }
    else
    {
        joint->is_effector_ = true;
        joint->target_      = _target;
    }
}


//----------------------------------------------------------------------------


void Inverse_kinematics::solve_IK(unsigned int _iterations)
{
    timer_.start();


    //--- Step 1: collect effectors ---
    if (collect_effectors())
    {
        //--- Step 2: collect degrees of freedom ---
        if (collect_dofs())
        {
            //--- Step 3: perform some Gauss-Newton iterations ---
            for (unsigned int i=0; i < _iterations; ++i)
            {
                // setup Jacobian matrix and vector of effector errors
                if (setup())
                {
                    // solve linear system for the update dtheta
                    compute_update();

                    // apply update dtheta to skeleton matrices
                    apply_update();
                }
            }
        }
    }


    timer_.stop();
}


//----------------------------------------------------------------------------


unsigned int Inverse_kinematics::collect_effectors()
{
    effectors_.clear();

    for (Joint* joint : skeleton_.joints_)
    {
        if (joint->is_effector_)
        {
            effectors_.push_back( Effector(joint->target_, joint) );
        }
    }

    return effectors_.size();
}


//----------------------------------------------------------------------------


unsigned int Inverse_kinematics::collect_dofs()
{
    dofs_.clear();

    int dof_idx = 0;

    // start from effectors
    for (auto effector: effectors_)
    {
        // get effector joint
        Joint *joint = effector.joint;

        // go up kinematic chain to collect DoF that joint depends on
        // stop at root joint
        for (joint=joint->parent_; !joint->is_root_; joint=joint->parent_)
        {
            // was this joint added to DoFs alrady?
            if (dofs_.find(joint->name_) != dofs_.end())
                continue;

            // if not, insert vector of DoFs for this joint
            std::vector<DoF>& joint_dofs = dofs_[joint->name_];

            // knees have one axis only
            if (joint->name_.find("knee") != std::string::npos)
            {
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(1,0,0)));
            }
            // elbows cannot rotate with z axis
            else if (joint->name_.find("elbow") != std::string::npos)
            {
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(1,0,0)));
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(0,1,0)));
            }
            // general joints have three axes
            else
            {
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(1,0,0)));
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(0,1,0)));
                joint_dofs.push_back(DoF(dof_idx++, joint, Vec3(0,0,1)));
            }
        }
    }

    return dof_idx;
}


//----------------------------------------------------------------------------


bool Inverse_kinematics::setup()
{
    // how many constraints? three per effector (x,y,z)
    int n_targets = (int) effectors_.size() * 3;


    // count number of degrees of freedom
    int n_dofs = 0;
    for (auto jit = dofs_.begin(); jit != dofs_.end(); ++jit)
    {
        //add number of dofs of joint to overall number of dofs.
        n_dofs += jit->second.size();
    }

    // DEBUG information
    //std::cout << effectors_.size() << " effectors, " << n_targets << " constraints\n";
    //std::cout << dofs_.size() << " involved joints, " << n_dofs << " dofs\n";


    // if #constraints or #dofs is zero -> stop
    if (n_targets * n_dofs == 0)
    {
        std::cerr << "IK: matrix is degenerate!\n";
        return false;
    }


    // Jacobian matrix and vector of errors
    errors_   = VectorXd::Zero(n_targets);
    jacobian_ = MatrixXd::Zero(n_targets, n_dofs);

    for (int i=0; i < (int)effectors_.size(); ++i)
    {
        const Effector &e  = effectors_[i];
        const Vec3 s = skeleton_.joint_positions_.col(e.joint->index_);

        errors_(i*3 + 0) = e.target(0) - s(0);
        errors_(i*3 + 1) = e.target(1) - s(1);
        errors_(i*3 + 2) = e.target(2) - s(2);


        // go up kinematic chain, add entries to Jacobian
        for (Joint* joint=e.joint->parent_; !joint->is_root_; joint=joint->parent_)
        {
            // get dofs of current joint
            const std::vector<DoF> &joint_dofs = dofs_[joint->name_];

            // get rotation center of current joint
            const Vec3 c = skeleton_.joint_positions_.col(joint->index_);

            for (auto& dof : joint_dofs)
            {
                // rotate rotation axis of joint by its global transformation
                const Vec3 axis =  joint->transform_.linear()*dof.axis;

                // gradient of s w.r.t. dof
                const Vec3 grad = axis.cross(s-c);

                // update Jacobian matrix
                jacobian_(3*i + 0, dof.index) = grad(0);
                jacobian_(3*i + 1, dof.index) = grad(1);
                jacobian_(3*i + 2, dof.index) = grad(2);
            }
        }
    }

    return true;
}


//----------------------------------------------------------------------------


void Inverse_kinematics::compute_update()
{
    dtheta_.setZero();

    unsigned int n     = jacobian_.cols();
    MatrixXd jacobianT = jacobian_.transpose();
    MatrixXd DLS       = jacobianT * jacobian_ + lambda_ * MatrixXd::Identity(n,n);
    dtheta_            = DLS.ldlt().solve( jacobianT * errors_ );
}


//----------------------------------------------------------------------------


void Inverse_kinematics::apply_update()
{
    // go through map of all DoFs
    for (auto const& jdofs : dofs_)
    {
        // go through all DoFs of one joint
        for (auto const& dof: jdofs.second)
        {
            // damp angle
            float angle = (float)dtheta_[dof.index] * 0.01f;

            // apply it to local bone transformation
            dof.joint->local_.rotate(Eigen::AngleAxisf(angle, dof.axis));

            // joint rotation
            dof.joint->localJ_.rotate(Eigen::AngleAxisf(angle*0.5f, dof.axis));
        }
    }

    // update forward kinematics
    skeleton_.transform();
}


//=============================================================================
