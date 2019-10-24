//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================
#pragma once
//=============================================================================

#include "mesh/Skeleton.h"

#include <Eigen/Dense>
using namespace Eigen;

#include <pmp/Timer.h>

#include <vector>
#include <map>


//== CLASS DEFINITION =========================================================


/** The class Inverse_kinematics is used for computing inverse kinematics (IK)
    on a specified skeleton (or character). After setting target positions for
    a set of effector joints (see set_effector()) a call of solve_IK()
    performs the IK computations.
    The constraints and degrees of freedom of the IK problem are represented
    by the types Effector and DoF and stored in effectors_ and dofs_ .
*/

class Inverse_kinematics
{
public: // public functions

    /// construct with a skeleton or character and the damping factor for
    /// damped least squares IK
    Inverse_kinematics(Projective_Skinning::Skeleton& skeleton, float damping=0.05);

    /// set the damping value for damped least squares IK
    void set_damping(float d) { lambda_=d; }
    /// get the damping value for damped least squares IK
    float damping() const { return lambda_; }

    /// use joint as effector and set its target position. call solve_IK() afterwards
    void set_effector(int joint_idx, const Projective_Skinning::Vec3& target_position);

    /// solve inverse kinematics
    void solve_IK(unsigned int iterations=10);

    /// get timing of last ik
    double get_ik_time(){return timer_.elapsed();}


private: // private types

    /// The Effector class stores the effector joint and its target position
    struct Effector
    {
        /// construct with joint and target position
        Effector(const Projective_Skinning::Vec3& p, Projective_Skinning::Joint* j) : target(p), joint(j) {}

        /// the target position of this effector
        Projective_Skinning::Vec3 target;
        /// the joint that acts as an effector
        Projective_Skinning::Joint* joint;
    };


    /** The DoF structure represents degrees of freedom for the inverse kinematics
      computation. Each DoF corresponds to a joint and its rotation axis, which
      is either the x-, y-, or z-axis. The index is computed by just enumerating
      all DoFs, and corresponds to the column index in the Jacobian matrix
      */
    struct DoF
    {
        /// construct with DoF index, joint, and axis
        DoF(int idx=-1,Projective_Skinning::Joint* j=NULL, Projective_Skinning::Vec3 a=Projective_Skinning::Vec3(0,0,0)) :
            index(idx), joint(j), axis(a)
        {}

        /// the index of this DoF (column index for Jacobian matrix)
        int index;

        /// the joint
        Projective_Skinning::Joint* joint;

        /// the rotation axis (x, y, or z) of this DoF
        Projective_Skinning::Vec3 axis;
    };


private: // private functions

    /// collect all effectors in effectors_. returns number of effectors.
    unsigned int collect_effectors();

    /// collect all degrees of freedom in dofs_. returns number of dofs.
    unsigned int collect_dofs();

    /// compute Jacobian matrix and vector of errors
    bool setup();

    /// solve linear system to get update dtheta_
    void compute_update();

    /// update local transformation by dtheta_, trigger forward kinematics
    void apply_update();


private: // private data

    /// the skeleton used for inverse kinematics
    Projective_Skinning::Skeleton& skeleton_;

    /// the damping factor for damped least squares IK
    float    lambda_;

    /// vector of errors (difference of effector positions to target positions)
    VectorXd errors_;

    /// Jacobian matrix
    MatrixXd jacobian_;

    /// update to joint angles
    VectorXd dtheta_;

    /// vector of effectors, each storing a joint and its target position
    /// \sa Effector
    std::vector<Effector> effectors_;

    /// The degrees of freedom (DoFs) in the current IK problem. DoFs are
    /// stored in a map, such that you can access the DoFs of a joint by
    /// specifying its name. For each joint we store a vector of DoFs.
    /// \sa DoF
    std::map< std::string, std::vector<DoF> > dofs_;

    /// Timer to measure ik-time
    pmp::Timer timer_;
};


//=============================================================================
