//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Animator.h"

namespace Projective_Skinning
{

Animator::Animator()
    :mesh_(D_USE_SLIDING_JOINTS),
    active_joint_(1),
    animation_mode_(2),
    ignore_last_frame_collisions_(true),
    solver_(nullptr),
    time_(0),
    avg_simulation_time_(0), avg_upsampling_time_(0), avg_skeleton_time_(0), avg_normal_time_(0), avg_collision_time_(0), avg_draw_time_(0),
    num_available_animations_(0),
    current_animation_(0),
    animate_current_joint_(false)

{
    std::cout << "\n\n\tFAST PROJECTIVE SKINNING ";
    if(D_GPU)
        std::cout << "(GPU)";
    else
        std::cout << "(CPU)";
    std::cout << "\n___________________________________________________\n\n";

}

Animator::~Animator()
{
    delete solver_;
}

void Animator::init(const char *skin_lr_filename, const char* skel_filename, const char* skin_hr_filename, const char* us_filename)
{
    //load from skin and skeleton file or from saved .psmesh file and skeleton
    if(!mesh_.init(skin_lr_filename, skel_filename, D_SHRINKSKIN, D_USE_DIFF_MASSES, D_MASS, skin_hr_filename, us_filename))
    {
        std::cerr << "Failed to initialize mesh." << std::endl;
        exit(1);
    }


    if(D_SIMULATE)
    {

#ifdef WITH_CUDA
        if(D_GPU)
        {
            solver_ = D_GLOBAL_COLLISIONS ? new PDsolverCudaCollisions() : new PDsolverCuda();
        }
#endif

        if(!D_GPU || solver_ == nullptr)
        {
            if(D_GPU) std::cout << "CUDA not supportet! Building CPU version." << std::endl;
            solver_ = D_GLOBAL_COLLISIONS ? new PD_Solver_Collisions() : new PD_Solver();
        }

        solver_->init(&mesh_, D_TETSTRAIN, D_ANCHOR, D_COLLISION, D_TIMESTEP, D_DAMPING, !D_USE_BC);


        if(D_GLOBAL_COLLISIONS)
        {
            hash_.init(mesh_.vertices_,mesh_.tets_.collision_indices,mesh_.collision_tet_basepoints_,mesh_.num_simulated_skin_);
        }
    }

	// define D_GPU to false if cuda is not suportet
#ifndef WITH_CUDA

#undef D_GPU
#define D_GPU false

#endif

}




void Animator::update_skeleton(bool _animate, float dt)
{
    if(_animate)
    {
        time_+=dt;

        if(!timestamps_.empty() && !animate_current_joint_)
        {
            // sine animation curve
            //float norm_time = 0.5f - 0.5f*cos(0.002*time_/timestamps_.size());

            int slowdown = 500*timestamps_.size();
            int lower_int = (int)time_/slowdown;
            float slope = 1.0/(float)slowdown;

            // saw animation curve
            float norm_time = (lower_int%2 == 0) ?  slope*time_ - (float)lower_int : -slope*time_ + lower_int + 1;

            load_pose_from_animation(norm_time);
        }
        else
        {
            animate_current_joint(time_);
        }

        solver_->update_anchors();
    }
}

void Animator::update_mesh(bool animate, float dt, bool update_high_res, bool _update_collisions, bool timing_updates, float *d_vbo, float *d_nbo)
{
    static int its = 0;
    static float skeltime = 0;
    static float normaltime = 0;
    static float simtime = 0;
    static float coltime = 0;
    static float ustime = 0;
    static float drawtime = 0;

    if(timing_updates)
    {
        drawtime += timer_.stop().elapsed();

        its ++;
        timer_.start();
        if(_update_collisions)
            update_collisions();
        coltime += timer_.stop().elapsed();

        // forward kinematics
        timer_.start();
        update_skeleton(animate, dt);
        skeltime += timer_.stop().elapsed();

        // update mesh (PD simulation)
        timer_.start();
        solver_->update_skin(D_ITERATIONS);
        simtime += timer_.stop().elapsed();

        // update normals
        timer_.start();
        solver_->update_normals(true);
        normaltime += timer_.stop().elapsed();

        // upsampling
        if(mesh_.use_high_res_ && update_high_res)
        {
            timer_.start();
            solver_->update_HR(d_vbo, d_nbo);
            ustime += timer_.stop().elapsed();
        }
        else
            solver_->update_ogl_sim_mesh_buffers(d_vbo, d_nbo);

        // update time variables
        if(its % 50 == 0)
        {
            avg_simulation_time_ = simtime/50.0;
            simtime = 0.0f;

            avg_normal_time_ = normaltime/50.0;
            normaltime = 0.0f;

            avg_upsampling_time_ = ustime/50.0;
            ustime = 0.0f;

            avg_skeleton_time_ = skeltime/50.0;
            skeltime = 0.0f;

            avg_collision_time_ = coltime/50.0;
            coltime = 0.0f;

            avg_draw_time_ = drawtime/50.0;
            drawtime = 0.0f;
        }

        timer_.start();
    }
    else
    {
        if(_update_collisions)
            update_collisions();

        // forward kinematics
        update_skeleton(animate, dt);

        // update mesh (PD simulation)
        solver_->update_skin(D_ITERATIONS);

        // update normals
        solver_->update_normals(true);

        // upsampling
        if(mesh_.use_high_res_ && update_high_res)
            solver_->update_HR(d_vbo, d_nbo);
        else
            solver_->update_ogl_sim_mesh_buffers(d_vbo, d_nbo);
    }

}


void Animator::reset()
{
    reset_collisions();

    for(auto jo : mesh_.skeleton_.joints_) jo->is_effector_ = false;

    solver_->reset();

    time_ = 0;
}

void Animator::reset_collisions()
{
    if(D_GLOBAL_COLLISIONS)
    {
        collision_indices_.clear();
        lastFrameIgnored_.clear();
        is_not_TI_.clear();
        solver_->reinit(collision_indices_, 0);
    }
}

void Animator::rotate_current_joint(int axis, float angle)
{
    if(active_joint_ >= 0)
    {
        //mesh_.skeleton_.angles_(axis,active_Bone_) += angle;
        Vec3 ax(0,0,0); ax(axis) = 1.0;
        mesh_.skeleton_.joints_[active_joint_]->local_ *= Eigen::AngleAxisf(angle,ax);
        mesh_.skeleton_.joints_[active_joint_]->localJ_ *= Eigen::AngleAxisf(angle/2.0,ax);

        mesh_.skeleton_.transform();
        solver_->update_anchors();
    }
}

void Animator::set_animation_mode(int mode)
{
    reset();
    time_ = 0;
    if(mode < num_available_animations_)
    {
        std::string f, empty;
        f = animation_base_ + std::to_string(mode) + std::string(".anim");
        animate_current_joint_ = !load_animation(f, empty);
        current_animation_ = mode;

        if(animate_current_joint_)
        {
            std::cout << "Animating selected joint..." << std::endl;
        }
    }
    else
    {
        animate_current_joint_ = true;
        current_animation_ = -1;
        std::cout << "Animating selected joint..." << std::endl;
    }

}

void Animator::increase_bone()
{
    active_joint_ ++;

    if(active_joint_ >= (int)mesh_.skeleton_.vol_bones_.size())
        active_joint_ = -1;

}

void Animator::decrease_bone()
{
    active_joint_ --;

    if(active_joint_ < -1)
        active_joint_ = mesh_.skeleton_.vol_bones_.size() -1;

}

void Animator::update_collisions()
{
    std::vector<bool> isHandCollision;
    std::set<int> alreadyIn;
    std::vector<int> colinds;
    std::vector<int> colindsH;

    for(usint i = 0; i < collision_indices_.size()/4; i++)
    {
        int ip = collision_indices_[4*i];
        int i1 = collision_indices_[4*i + 1];
        int i2 = collision_indices_[4*i + 2];
        int i3 = collision_indices_[4*i + 3];

        Vec3 p0 = mesh_.vertices_.col(i1);
        Vec3 p1 = mesh_.vertices_.col(i2);
        Vec3 p2 = mesh_.vertices_.col(i3);
        Vec3 p = mesh_.vertices_.col(ip);

//        float maxesq = std::max((p0-p1).dot(p0-p1), std::max((p1-p2).dot(p1-p2),(p2-p0).dot(p2-p0)));
//        float minpesq = std::min((p0-p).dot(p0-p), std::min((p1-p).dot(p1-p),(p2-p).dot(p2-p)));


        Vec3 bary = (p0 + p1 + p2)/3.0;
        float radiusTrigSQ = (p0 - bary).squaredNorm();
        float radiusPSQ = (p - bary).squaredNorm();


        Vec3 n = ((p1 - p0).cross(p2 - p0)).normalized();
        float ndist = n.dot(p - p0);

        if(radiusPSQ < radiusTrigSQ
                //minpesq < maxesq
                &&  ndist < 0.25f*mesh_.skin_.avg_edge_length && (mesh_.vertex_normals_.col(i1).dot(mesh_.vertex_normals_.col(ip)) < 0.25) )
        {
            colinds.push_back(ip);
            colinds.push_back(i1);
            colinds.push_back(i2);
            colinds.push_back(i3);

            alreadyIn.insert(ip);

            isHandCollision.push_back(is_not_TI_[i]);
        }
    }


    hash_.test_for_collsions(mesh_.vertices_, mesh_.tets_.collision_indices, mesh_.collision_tet_basepoints_);

    if(ignore_last_frame_collisions_)
    {
        for(auto i : hash_.colliding_vertices_)
        {
            if(lastFrameIgnored_.find(i) != lastFrameIgnored_.end() && ! mesh_.skeleton_.is_hand_bone_[mesh_.skin_.assoc_bone[i]])
            {
                alreadyIn.insert(i);
            }
        }
    }
    lastFrameIgnored_.clear();

    for(usint i = 0; i < hash_.colliding_vertices_.size(); i++)
    {

        int id = hash_.colliding_vertices_[i];

        if(alreadyIn.find(id) != alreadyIn.end())
        {
            lastFrameIgnored_.insert(id);
            continue;
        }

        usint trig = mesh_.tets_.collision_tet_to_trig[hash_.colliding_tets_[i]];

        int i0 = mesh_.skin_.sim_indices[3*trig + 0];
        int i1 = mesh_.skin_.sim_indices[3*trig + 1];
        int i2 = mesh_.skin_.sim_indices[3*trig + 2];

        IndexSet &nb = mesh_.skin_.neighbors_2_ring[id];
        if(nb.find(i0) != nb.end() || nb.find(i1) != nb.end() || nb.find(i2) != nb.end())
        {
            lastFrameIgnored_.insert(id);
            continue;
        }

        // ignore vertex to hand triangle collisions
        if(mesh_.skeleton_.is_character_mesh_)
        {
            if(mesh_.skeleton_.is_hand_bone_[mesh_.skin_.assoc_bone[i0]] ||
               mesh_.skeleton_.is_hand_bone_[mesh_.skin_.assoc_bone[i1]] ||
               mesh_.skeleton_.is_hand_bone_[mesh_.skin_.assoc_bone[i2]] ||
               mesh_.additional_anchors_.find(id) != mesh_.additional_anchors_.end() ||
               mesh_.additional_anchors_.find(i0) != mesh_.additional_anchors_.end() ||
               mesh_.additional_anchors_.find(i1) != mesh_.additional_anchors_.end() ||
               mesh_.additional_anchors_.find(i2) != mesh_.additional_anchors_.end())
            {
                continue;
            }
        }

        if(mesh_.vertex_normals_.col(i0).dot(mesh_.vertex_normals_.col(id)) < 0.0)
        {
            colinds.push_back(id);
            colinds.push_back(i0);
            colinds.push_back(i1);
            colinds.push_back(i2);

            isHandCollision.push_back(mesh_.skeleton_.is_hand_bone_[mesh_.skin_.assoc_bone[id]]);
        }
    }

    // new GPU friendly version todo: make this better
    collision_indices_.clear();
    is_not_TI_.clear();

    int num_Hand = 0;
    is_not_TI_.resize(isHandCollision.size(), false);
    for(usint i = 0; i < colinds.size()/4; i++)
    {
        if(!isHandCollision[i])
        {
            collision_indices_.push_back(colinds[4*i + 0]);
            collision_indices_.push_back(colinds[4*i + 1]);
            collision_indices_.push_back(colinds[4*i + 2]);
            collision_indices_.push_back(colinds[4*i + 3]);
        }
        else
        {
            colindsH.push_back(colinds[4*i + 0]);
            colindsH.push_back(colinds[4*i + 1]);
            colindsH.push_back(colinds[4*i + 2]);
            colindsH.push_back(colinds[4*i + 3]);

            num_Hand++;
            is_not_TI_[isHandCollision.size() - num_Hand] = true;
        }
    }
    collision_indices_.insert(collision_indices_.end(), colindsH.begin(), colindsH.end());

    solver_->reinit(collision_indices_, collision_indices_.size()/4 - num_Hand);
}


bool Animator::load_animation(const std::string file, const std::string base)
{
    std::ifstream ifs;
    ifs.open(file);
    if(!ifs)
    {
        return false;
    }
    else
    {
        std::cout << "\nOpened " << file << std::endl;
    }

    int nTS,nJ;
    ifs >> nTS >> nJ;

    int nJ2 = mesh_.skeleton_.joint_positions_.cols();
    if(nJ > nJ2)
    {
        std::cerr << "Problem: Incorrect joint size" << std::endl;
        return false;
    }

    quats_.resize(nJ, Eigen::Quaternionf(1,0,0,0));

    timestamps_.clear();
    for(int i = 0; i < nTS; i++)
    {
        timestamps_.emplace_back(nJ2,(float)i);
        Timestamp &TS = timestamps_.back();

        ifs >> TS.time;
        for(int j = 0; j < nJ; j++)
        {
            ifs >> TS.axiss(0,j) >> TS.axiss(1,j) >> TS.axiss(2,j);
        }
        for(int j = nJ; j < nJ2; j++)
        {
            TS.axiss.col(j) = Vec3(1,0,0);
        }
        for(int j = 0; j < nJ2; j++)
        {
            if(j < nJ)
                ifs >> TS.angles[j];
            else
                TS.angles[j] = 0;
            TS.quats[j] = Eigen::AngleAxisf(TS.angles[j], TS.axiss.col(j));
        }
    }
    ifs.close();

    if(!base.empty())
    {
        animation_base_ = file.substr(0,file.rfind(base)) + std::string(base);

        std::string f = animation_base_ + std::to_string(0) + std::string(".anim"); ifs.open(f);
        int ctr = 1;
        while(ifs)
        {
            ifs.close();
            f = animation_base_ + std::to_string(ctr) + std::string(".anim"); ifs.open(f);
            ctr++;
        }
        num_available_animations_ = ctr - 1;

        std::cout << "available animations: " << num_available_animations_ << std::endl;
    }

    std::cout << "animation " << file << " successfully loaded. " << timestamps_.size() << " Timestamps" << std::endl;
    return true;
}

void Animator::load_pose_from_animation(float time)
{
    // avoid overflow
    time = std::min(0.999999f,time);

    // get the timestamp indices before and after time
    int id0 = (int)(time*(timestamps_.size() - 1));
    int id1 = id0 + 1;
    if(id1 >= (int)timestamps_.size())
    {
        std::cerr << "invalid timestamp";
        return;
    }

    // timefraction between both timestamps
    float dt = time*(timestamps_.size() - 1) - (float)id0;

    // set quaternions to interpolated rotation
    Timestamp &TS0 = timestamps_[id0];
    Timestamp &TS1 = timestamps_[id1];
    for(usint i = 0; i < quats_.size(); i++)
    {
        quats_[i] = TS0.quats[i].slerp(dt,TS1.quats[i]);
    }

    mesh_.skeleton_.transform(quats_);
}

void Animator::animate_current_joint(const float time)
{
    if(active_joint_ >= 0 && active_joint_ < (int)mesh_.skeleton_.joints_.size())
    {
        int axis = ((int)(time/(2.0*M_PI*500.0)))%3;
        Vec3 ax(0,0,0); ax(axis) = 1.0;
        float angle = M_PI*0.1*sin(0.002*time);
        mesh_.skeleton_.joints_[active_joint_]->local_.linear() = Mat33(Eigen::AngleAxisf(angle,ax));
        mesh_.skeleton_.joints_[active_joint_]->localJ_.linear() = Mat33(Eigen::AngleAxisf(angle/2.0,ax));

        mesh_.skeleton_.transform();
    }
}

}


