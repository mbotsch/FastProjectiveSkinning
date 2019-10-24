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

class VolBone
{
public:
    VolBone(const Vec3 &v0,const Vec3 &v1, float r, int nslice = 6, int ndisk = 3)
	:v0_(v0),v1_(v1),radius_(r)
	{
        Vec3 d = v1 - v0;
        Vec3 c = 0.5*(v1 + v0);
		float height = (d).norm();
		int steps = (int)(height*5.0/(18.0/(float)ndisk*r)) + 1;
		
		vertices_.resize(3, 2 + nslice*(2*ndisk + steps - 1));
		int vi = 0;
		
		//vertices for cap1
		vertices_.col(vi) << 0,0,r + 0.5*height; vi++;
		float phi = 0;
		for(int it = 1; it < ndisk + 1 ; it++)
		{
			float theta = it*M_PI/((float)ndisk*2.0);
			for(int ip = 0; ip < nslice; ip++)
			{
				vertices_.col(vi) << r*sin(theta)*cos(phi),
				r*sin(theta)*sin(phi),
				r*cos(theta) + 0.5*height; 
				vi++;
				phi += 2.0*M_PI/(float)nslice;
			}
			//phi += M_PI/6.0;
		}
		
		//vertices for cylinder
		for(int i = 1; i<steps; i++)
		{
			float h = i*height/((float)steps);
			for(int ip = 0; ip < nslice; ip++)
			{    
				vertices_.col(vi) << r*cos(phi),
				r*sin(phi),
				0.5*height - h; 
				vi++;
				phi += 2.0*M_PI/(float)nslice;
			}
			//phi += M_PI/6.0;
		}
		
		//vertices for cap2
		for(int it = ndisk; it < ndisk*2 ; it++)
		{
			float theta = it*M_PI/((float)ndisk*2.0);
			for(int ip = 0; ip < nslice; ip++)
			{
				vertices_.col(vi) << r*sin(theta)*cos(phi),
				r*sin(theta)*sin(phi),
				r*cos(theta) - 0.5*height; 
				vi++;
				phi += 2.0*M_PI/(float)nslice;
			}
			//phi += M_PI/6.0;
		}
		vertices_.col(vi) << 0,0,-0.5*height - r; vi++;
		nVertices_ = vi;
		
		//rotate & translate
		d.normalize();
		float angle = acos(d(2));
		if(fabs(angle) > 1e-3)
		{
            Vec3 axis(-d(1),d(0),0);
            if(axis.squaredNorm() < 1e-4)
                axis = Vec3(1,0,0);

			axis.normalize(); 
            Mat33 R;
            R = Eigen::AngleAxisf(angle, axis);
			vertices_ = R*vertices_;
		}
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
    ~VolBone(){}
    
    void rotate(float rotX, float rotY, float rotZ)
    {
    Mat33 rx, ry, rz, r;
    rx = Eigen::AngleAxisf(rotX, Vec3::UnitX());
    ry = Eigen::AngleAxisf(rotY, Vec3::UnitY());
    rz = Eigen::AngleAxisf(rotZ, Vec3::UnitZ());
	r = rz*ry*rx;
	vertices_ = r*vertices_;
    }
    
    bool vIsInBone(const Vec3& v) const
	{
        Vec3 x = v - v0_;
        Vec3 a = v1_ - v0_; //precompute
		float rsq = radius_*radius_; //precompute
		float dot = x.dot(a);
		if(dot <= 0)
		{
			//test sphere1
			float distsq = x.dot(x);
			if(distsq < rsq)
			{
				return true;
			}
		}
		else 
		{
			float asq = a.dot(a); //precompute
			if(dot >= asq)
			{
				//test sphere2
                Vec3 x2 = v - v1_;
				float distsq = (x2).dot(x2);
				if(distsq < rsq)
				{
					return true;
				}
			}
			else
			{
				//test cylinder
				float distsq = x.dot(x) - dot*dot/asq;
				if(distsq < rsq)
				{
					return true;
				}
			}
		}
		return false;
	}
	
	
    bool vIsInBone(const Vec3 &v, float factor)
	{
        Vec3 a = (v1_ - v0_);
		float L = a.norm();
		a/=L;
		float Lnew = a.dot(v - v0_);
		Lnew = (Lnew > L)? L : Lnew;
		Lnew = (Lnew < 0)? 0 : Lnew;
        Vec3 nearest = v0_ + Lnew*a;
		
        Vec3 p = v - nearest;
		if(p.dot(p) < radius_*radius_*factor*factor)
		{
			return true;
		}
		else
		{
			return false;
		}
		
		
	}
	
    bool pushVOut(const Vec3 &v, Vec3& dest)
	{
        Vec3 a = (v1_ - v0_);
		float L = a.norm();
		a/=L;
		float Lnew = a.dot(v - v0_);
		Lnew = (Lnew > L)? L : Lnew;
		Lnew = (Lnew < 0)? 0 : Lnew;
        Vec3 nearest = v0_ + Lnew*a;
		
        Vec3 p = v - nearest;
		if(p.dot(p) < radius_*radius_)
		{
			dest = nearest + radius_*p.normalized();
			return true;
		}
		else
		{
			dest = v;
			return false;
		}
		
		
	}

    float getDist(const Vec3& v)
    {
        Vec3 a = (v1_ - v0_);
        float L = a.norm();
        a/=L;
        float Lnew = a.dot(v - v0_);
        Lnew = (Lnew > L)? L : Lnew;
        Lnew = (Lnew < 0)? 0 : Lnew;
        Vec3 nearest = v0_ + Lnew*a;

        Vec3 p = v - nearest;
        return p.norm();
    }
    
    Vec3 v0_,v1_;
    float radius_;
    Mat3X vertices_;
    IndexVector indices_;
    unsigned int nVertices_;
};

}
