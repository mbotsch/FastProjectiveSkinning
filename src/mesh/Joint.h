//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include "Helper.h"
#include <iostream>

namespace Projective_Skinning
{

class Joint
{
public:
    Joint(const std::string name, const Vec3 &p, Eigen::Affine3f& T, Eigen::Affine3f& TJ,  const int ind, const bool root = false)
        :name_(name), position_(p),parent_(nullptr), is_root_(root), index_(ind), volumetric_jointindex_(-1), parent_bone_(-1), transform_(T), transformJ_(TJ), is_effector_(false)
    {
        transform_.setIdentity(); transformJ_.setIdentity(); local_.setIdentity(); localJ_.setIdentity(); global_.setIdentity(); global_inv_.setIdentity();
    }

    void init()
    {
        if(!is_root_)
        {
            local_.translate(position_ - parent_->position_);
            global_ = parent_->global_ * local_;
        }
        else
        {
            local_.translate(position_);
            global_ = local_;
        }

        localJ_ = local_;
        global_inv_ = global_.inverse();

        for(auto joint : childreen_)
            joint->init();
    }

    void forward_kinematics()
    {
        Eigen::Affine3f globalJ = localJ_;
        if(!is_root_)
        {
            global_ = parent_->global_ * local_;
            globalJ = parent_->global_ * localJ_;
        }
        else
        {
            global_ = local_;
        }

        transform_ = global_*global_inv_;
        transformJ_ = globalJ*global_inv_;

        for(auto joint : childreen_)
            joint->forward_kinematics();
    }

    bool is_leaf(){return childreen_.empty();}

public:
    std::string name_;
    Vec3 position_;
    std::vector<Joint*> childreen_;
    Joint* parent_;
    bool is_root_;
    int index_;
    int volumetric_jointindex_;
    int parent_bone_;
    Eigen::Affine3f local_, localJ_, global_, global_inv_;
    Eigen::Affine3f &transform_, &transformJ_;

    // ik variables
    bool is_effector_;
    Vec3 target_;
};


class VolJoint
{
public:
    VolJoint(const Vec3 &c, float r, unsigned int index, int nslice = 6, int ndisk = 3)
    :c_(c),radius_(r), stickIndex_(index)
    {
	
	vertices_.resize(3, 2 + nslice*(2*ndisk - 1));
	int vi = 0;
	
	//vertices for cap1
	vertices_.col(vi) << 0,0,r; vi++;
	float phi = 0;
	for(int it = 1; it < 2*ndisk; it++)
	{
	    float theta = it*M_PI/((float)ndisk*2.0);
	    for(int ip = 0; ip < nslice; ip++)
	    {
		vertices_.col(vi) << r*sin(theta)*cos(phi),
				     r*sin(theta)*sin(phi),
				     r*cos(theta); 
		vi++;
		phi += 2.0*M_PI/(float)nslice;
	    }
	    //phi += M_PI/6.0;
	}
	
	
	vertices_.col(vi) << 0,0,-r; vi++;
	nVertices_ = vi;
	
	//translate	
	for(unsigned int i = 0; i < nVertices_; i++)
	{
	    vertices_.col(i) += c;
	}
	
	//tesselate
	//caps
	for(int i = 1; i< nslice + 1; i++)
	{
	    indices_.push_back(0);
	    indices_.push_back(i);
	    indices_.push_back(i%nslice + 1);
	    
	    indices_.push_back(nVertices_ - 1);
	    indices_.push_back(nVertices_ - 1 - i);
	    indices_.push_back(nVertices_ - 1 - i%nslice - 1);
	}
	//cylinder
	for(unsigned int i = 1; i < nVertices_ - nslice - 1; i++)
	{
	    int j = (i-1)/nslice;
	    indices_.push_back(i);
	    indices_.push_back(i + (j + 1 - (i + nslice - 2)/nslice)*nslice + nslice - 1);
	    indices_.push_back(i + nslice);
	    
	    indices_.push_back(i);
	    indices_.push_back(i + nslice);
	    indices_.push_back(j*nslice + i%nslice + 1);
	}
	
    }
    VolJoint();
    ~VolJoint(){}
    bool vIsInJoint(const Vec3& v, float rFactor = 1.0)const
	{
        Vec3 p = v - c_;
		float r = radius_*rFactor;
		return (p.dot(p) < r*r);
	}
	
    bool pushVOut(const Vec3 &v, Vec3& dest)
	{
        Vec3 p = v - c_;
		if(p.dot(p) < radius_*radius_)
		{
			dest = c_ + radius_*p.normalized();
			return true;
		}
		else
		{
			dest = v;
			return false;
		}
	}
    
    Vec3 c_;
    float radius_;
    Mat3X vertices_;
    IndexVector indices_;
    unsigned int nVertices_;
	unsigned int stickIndex_;
};

}
