//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include "defines.h"
#include "PD_solver.h"
#include "Collisions.h"

#ifdef WITH_CUDA
#include "PD_solver_cuda.h"
#endif

namespace Projective_Skinning
{

class Animator
{
public:
    Animator();
    ~Animator();

    void init(const char* skin_lr_filename, const char *skel_filename, const char *skin_hr_filename, const char *us_filename);
    void init_from_ini(const std::string ini_filename);
    void rotate_current_joint(int axis, float angle);
    void update_skeleton(bool animate, float dt);
    void reset();
    void reset_collisions();

    void update_mesh(bool animate, float dt, bool update_high_res, bool update_collisions, bool timing_updates, float* d_vbo, float* d_nbo, size_t num_bytes);
    void set_animation_mode(int mode);
    void increase_bone();
    void decrease_bone();

    inline int get_active_bone(){return active_joint_;}
    inline void set_active_bone(int b){ active_joint_ = b;}

    void update_collisions();

    bool load_animation(const std::string filename, const std::string base);
    void load_pose_from_animation(float time);

    void animate_current_joint(const float time);


    Mesh mesh_;
private:

    class Timestamp
    {
    public:
        Timestamp(int n_joints, float _time)
            :time(_time)
        {
            angles.resize(n_joints,0);
            axiss.resize(3,n_joints);
            axiss.setZero();
            axiss.row(0).setConstant(1.0);
            quats.resize(n_joints, Eigen::Quaternionf(1,0,0,0));
        }

        std::vector<float> angles;
        Mat3X axiss;
        std::vector<Eigen::Quaternionf> quats;
        float time;
    };
    std::vector<Eigen::Quaternionf> quats_;

    std::vector<Timestamp> timestamps_;


    int active_joint_;
    unsigned int animation_mode_;

public:

    //std::vector<float> weight_factors_;

    bool ignore_last_frame_collisions_;

    std::string basename_;

    Collision_Detection hash_;

    PD_Solver* solver_;

    pmp::Timer timer_;
    float time_;
    float avg_simulation_time_;
    float avg_upsampling_time_;
    float avg_skeleton_time_;
    float avg_normal_time_;
    float avg_collision_time_;
    float avg_draw_time_;

    std::vector<int> collision_indices_;

    std::string animation_base_;
    int num_available_animations_;
    int current_animation_;
    bool animate_current_joint_;

    IndexSet lastFrameIgnored_;
    std::vector<bool> is_not_TI_;

};

}
