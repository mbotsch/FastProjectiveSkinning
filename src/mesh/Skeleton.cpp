//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Skeleton.h"

namespace Projective_Skinning
{

const float Skeleton::bone_dist_factor_ = 0.75f;

Skeleton::~Skeleton()
{
    for(auto jo : joints_) delete jo;

    joints_.clear();
}

bool Skeleton::init(const Mat3X & positions, const IndexVector &indices, const Mat3X &vertices, const int moved)
{
    bone_indices_.clear();
    joint_by_name_.clear();
    name_by_joint_.clear();

    unsigned int   nV = positions.cols();

    joint_positions_.resize(3,nV);
    orig_joint_positions_.resize(3,nV);

    // read vertices
   joint_positions_ = positions;
   orig_joint_positions_ = positions;

    is_character_mesh_ = false;

    bone_indices_ = indices;

    is_hand_bone_.clear();
    is_hand_bone_.resize(bone_indices_.size()/2,false);

    std::vector<float> bone_radii(nV, 0);
    //compute boneradii from minimal distance to vertices
    if(orig_bone_radii_.size() == bone_radii.size())
        bone_radii = orig_bone_radii_;
    else
        bone_radii[0] = 10;

    for(usint l = 0; l < bone_indices_.size(); l+=2)
    {
        Vec3 b0 = joint_positions_.col(bone_indices_[l]);
        Vec3 b1 = joint_positions_.col(bone_indices_[l + 1]);

        if(moved == -1 || orig_bone_radii_.empty() ||
          (int)bone_indices_[l] == moved || (int)bone_indices_[l + 1] == moved)
        {
            float mindist = 1e8;

            for(usint i = 0; i < vertices.cols(); i++)
            {
                Vec3 v = vertices.col(i);
                Vec3 nearest = get_nearest_point_on_line(v,b0,b1);
                float dist = (nearest - v).squaredNorm();
                if(dist < mindist)
                {
                    mindist = dist;
                }

            }

            bone_radii[l/2 + 1] = sqrt(mindist)*bone_dist_factor_;
        }
    }

    orig_bone_radii_ = bone_radii;

    //create volumetric bones and joints
    create_volumetric_skeleton(bone_radii);

    joints_[0]->init();

    //optional, if needed
    compute_normals();

    return true;
}

void Skeleton::reset()
{
    for(auto jo : joints_)
    {
        jo->local_.linear() = Mat33::Identity();
        jo->localJ_.linear() = Mat33::Identity();
    }

    transform();
}

bool Skeleton::init(const char *filename, const Mat3X &vertices)
{
    // parse file
    if(std::ifstream(filename))
    {
        read_skeleton(filename, vertices);
    }
    else
    {
        std::cerr << "Error: cannot read skeleton from " << filename << std::endl; return false;
    }


    joints_[0]->init();


    //print out some skeleton information
    std::cout
    << "read "
    << filename << ": "
    << vol_joints_.size() << " Joints, "
    << vol_bones_.size() << " Bones" << std::endl;

    //optional, if needed
    compute_normals();

    return true;
}

void Skeleton::store_skeleton(const char *filename)
{
    std::cout << "storing skeleton to file: " << filename << std::endl;

    std::ofstream ofs(filename);

    ofs << joint_positions_.cols() << "\n";

    for(int i = 0; i < joint_positions_.cols(); i++)
    {
        ofs << joint_positions_(0,i) << "\t" << joint_positions_(1,i) << "\t" << joint_positions_(2,i) << "\t";
        if(joints_[i]->is_root_)
            ofs << joints_[i]->name_ << "\t" << "root" << "\n";
        else
            ofs << joints_[i]->name_ << "\t" << joints_[i]->parent_->name_ << "\n";
    }

    ofs.close();
}

void Skeleton::read_skeleton(const char* filename, const Mat3X& vertices)
{
	// parse the file
	std::ifstream ifs(filename);
	if ( !ifs )
	{
		std::cerr << "Opening " << filename << " failed. Maybe wrong filename?" << std::endl;
	}
	else
    {
		unsigned int   nV;
		std::string name;
        float          x, y, z;
		
		ifs >> nV;
		joint_positions_.resize(3,nV);
        orig_joint_positions_.resize(3,nV);
		
        std::vector<float> bone_radii(nV, 0);
		
		std::vector<std::string> jointnames;
		std::vector<std::string> parentnames;
		
		// read vertices
		for ( unsigned int i=0; i < nV && !ifs.eof(); ++i )
		{
			ifs >> x >> y >> z;
            joint_positions_.col(i) << x,y,z;
			ifs >> name;
			jointnames.push_back(name);
			joint_by_name_[name] = i;
			name_by_joint_[i] = name;
			ifs >> name;
			parentnames.push_back(name);
        }

        // determine if mesh is agcg character mesh
        is_character_mesh_ = joint_by_name_.find("r_hip") != joint_by_name_.end();

		for(usint i = 0; i < parentnames.size(); i++)
		{
            std::string pname = parentnames[i];
            for(usint j = 0; j < jointnames.size(); j++)
            {
                if(pname == jointnames[j])
                {
                    bone_indices_.push_back(j);
                    bone_indices_.push_back(i);
                    break;
				}
			}
		}

        is_hand_bone_.clear();
        is_hand_bone_.resize(bone_indices_.size()/2,false);

        // move leafjoints of agcg charcters into the mesh to get better radii and store all handjoints
        for(usint i = 0; i < bone_indices_.size(); i+=2)
		{
            usint ichild = bone_indices_[i+1];

            if( name_by_joint_[ichild].find("thumb") != name_by_joint_[ichild].npos ||
                name_by_joint_[ichild].find("index") != name_by_joint_[ichild].npos ||
                name_by_joint_[ichild].find("middle") != name_by_joint_[ichild].npos ||
                name_by_joint_[ichild].find("ring") != name_by_joint_[ichild].npos ||
                name_by_joint_[ichild].find("pinky") != name_by_joint_[ichild].npos)
            {
                is_hand_bone_[i/2] = true;
            }

		}
        orig_joint_positions_ = joint_positions_;


		//compute boneradii from minimal distance to vertices
		bone_radii[0] = 10;
        for(usint l = 0; l < bone_indices_.size(); l+=2)
		{
            Vec3 b0 = joint_positions_.col(bone_indices_[l]);
            Vec3 b1 = joint_positions_.col(bone_indices_[l +1]);
			
			float mindist = 1e8;
			
			for(usint i = 0; i < vertices.cols(); i++)
			{
				Vec3 v = vertices.col(i);
                Vec3 nearest = get_nearest_point_on_line(v,b0,b1);
				float dist = (nearest - v).squaredNorm();
				if(dist < mindist)
				{
					mindist = dist;
				}
				
			}

            bone_radii[l/2 + 1] = sqrt(mindist)*bone_dist_factor_;
        }

        //create volumetric bones and joints
        create_volumetric_skeleton(bone_radii);


    } // todo: use function for duplicated code
	
	ifs.close();
	
}

void Skeleton::create_volumetric_skeleton(std::vector<float> &bone_radii)
{
    vol_joints_.clear();
    for(usint j = 0; j < vol_joints_.size(); j++)
    {
        delete joints_[j];
    }
    joints_.clear();
    indices_.clear();
    vol_bones_.clear();
    transformations_.clear();
    bone_v_start_.clear();
    joint_v_start_.clear();

    usint nV = joint_positions_.cols();
    std::vector<float> joint_radii(nV, 0);

    //jointradii set to be maximum of neighboring boneradii
    joint_radii[0] = bone_radii[1];
    for(usint i = 1; i < nV; i++)
    {
        joint_radii[i] = 1e-10;
        for(usint l = 0; l < bone_indices_.size(); l+=2)
        {
            if(bone_indices_[l] == i || bone_indices_[l+1] == i)
            {
                joint_radii[i] = std::max(joint_radii[i], bone_radii[l/2 + 1]);
            }
        }
    }

    //find minimal radius
    minimal_r_ = 1e10;
    for(auto jr : joint_radii)
    {
        if(jr < minimal_r_)
            minimal_r_ = jr;
    }
    for(auto br : bone_radii)
    {
        if(br < minimal_r_)
            minimal_r_ = br;
    }

    //initialize transforms
    transformations_.resize(2*joint_positions_.cols());// + joints_.size());
    for(auto &tj : transformations_)
    {
        tj = Eigen::Affine3f::Identity();
    }

    // create joint pointer vector
    for(int i = 0; i < joint_positions_.cols(); i++)
    {
        std::string joint_name = ((int)name_by_joint_.size() == joint_positions_.cols()) ?
                                        name_by_joint_[i] :
                                        std::string("joint") + std::to_string(i);
        joints_.emplace_back(new Joint(joint_name,joint_positions_.col(i), transformations_[i] , transformations_[i + joint_positions_.cols()], i ,i == 0));
    }

    for(int i = 0; i < (int)bone_indices_.size(); i+=2)
    {
        int i0 = bone_indices_[i];
        int i1 = bone_indices_[i + 1];

        joints_[i0]->childreen_.push_back(joints_[i1]);
        joints_[i1]->parent_ = joints_[i0];
        joints_[i1]->parent_bone_ = i/2;
    }
	//resize radii if they not fit sticklength
    for(unsigned int l = 0; l < bone_indices_.size(); l+=2)
	{
		unsigned int l0,l1;
        l0 = bone_indices_[l];
        l1 = bone_indices_[l + 1];
		Vec3 s0, s1;
		s0 = joint_positions_.col(l0);
		s1 = joint_positions_.col(l1);
		
		//compute bonelength
		Vec3 a = s1 - s0;
		const float L = a.norm();
		a/=L;
		
		//compute minimum bonelength
		float L1 = 2*bone_radii[l/2 + 1];
        if(!joints_[l0]->is_leaf())
		{
			L1 += joint_radii[l0];
		}
        if(!joints_[l1]->is_leaf())
		{
			L1 += joint_radii[l1];
		}
		
		//resize joint and boneradii if neccessary
		if(L < L1)
		{
			joint_radii[l0]*= 0.99*L/L1;
			joint_radii[l1]*= 0.99*L/L1;
			bone_radii[l/2 + 1]*= 0.99*L/L1;
		}
	}
	
	
    //create Bones
    for(unsigned int l = 0; l < bone_indices_.size(); l+=2)
	{
		unsigned int l0,l1;
        l0 = bone_indices_[l];
        l1 = bone_indices_[l + 1];
		Vec3 s0, s1;
		s0 = joint_positions_.col(l0);
		s1 = joint_positions_.col(l1);
		
		Vec3 a = s1 - s0;
		float L = a.norm();
		a/=L;
		
		
		//create Bone
		const float boner = bone_radii[l/2 + 1];
		Vec3 v0,v1;
        if(!joints_[l0]->is_leaf())
			v0 = s0 + (joint_radii[l0] + boner)*a;
		else
			v0 = s0;
        if(!joints_[l1]->is_leaf())
			v1 = s1 - (joint_radii[l1] + boner)*a;
		else
			v1 = s1;

        vol_bones_.push_back(VolBone(v0,v1,boner));
	} 
	
	//create Joints
	for(unsigned int i = 0; i<joint_positions_.cols(); i++)
	{
		Vec3 v = joint_positions_.col(i);
		//create joint
        if(!joints_[i]->is_leaf())
            vol_joints_.push_back(VolJoint(v,joint_radii[i], i));
		
	}
	
    int nV_sum = 0;
	//determine bone and jointvertexstartingindices to transform bone and jointvertices
	for(auto b: vol_bones_)
	{
        bone_v_start_.push_back(nV_sum);
        nV_sum += b.vertices_.cols();
	}
	for(auto j: vol_joints_)
	{
        joint_v_start_.push_back(nV_sum);
        nV_sum += j.vertices_.cols();
	}
    
    //join vertices
    vertices_.resize(3,nV_sum);
	unsigned int vi = 0;
	for(auto b: vol_bones_)
	{
        vertices_.block(0,vi,3,b.vertices_.cols()) = b.vertices_;
		vi += b.vertices_.cols();
	}
	for(auto j: vol_joints_)
	{
        vertices_.block(0,vi,3,j.vertices_.cols()) = j.vertices_;
		vi += j.vertices_.cols();
	}
	
	//join indices
	vi = 0;
	for(auto b: vol_bones_)
	{
		for(unsigned int i = 0; i < b.indices_.size(); i++)
		{
			indices_.push_back(b.indices_[i] + vi);   
		}
		vi += b.vertices_.cols();
	}
	for(auto j: vol_joints_)
	{
		for(unsigned int i = 0; i < j.indices_.size(); i++)
		{
			indices_.push_back(j.indices_[i] + vi);
		}
		vi += j.vertices_.cols();
    }

    for(int i = 0; i < (int)vol_joints_.size(); i++)
    {
        joints_[vol_joints_[i].stickIndex_]->volumetric_jointindex_ = i;
    }
}

void Skeleton::compute_normals()
{
    normals_.resize(3,vertices_.cols());
    normals_.setZero();

    // calculate face normals, accumulate vertex normals
    for (auto idx_it=indices_.begin(); idx_it!=indices_.end();)
    {
        unsigned int i0 = *idx_it++;
        unsigned int i1 = *idx_it++;
        unsigned int i2 = *idx_it++;

        const Vec3& p0 = vertices_.col(i0);
        const Vec3& p1 = vertices_.col(i1);
        const Vec3& p2 = vertices_.col(i2);

        Vec3 facenormal = ((p1-p0).cross(p2-p0)).normalized();

        normals_.col(i0) += facenormal;
        normals_.col(i1) += facenormal;
        normals_.col(i2) += facenormal;
    }


    // normalize vertex normals
    normals_.colwise().normalize();
}

void Skeleton::transform()
{
    // do forward kinematics (root joint will call all its childreen)
    joints_[0]->forward_kinematics();

    // apply transformations to bone/joint-vertices and sticks
    update();
}

void Skeleton::transform(const std::vector<Eigen::Quaternionf> &quats)
{
    for(usint i = 0; i < quats.size(); i++)
    {
        joints_[i]->local_.linear() = quats[i].matrix();

        Eigen::Quaternionf qI(1,0,0,0);
        Eigen::Quaternionf q = qI.slerp(0.5,quats[i]);

        joints_[i]->localJ_.linear() = q.matrix();
    }

    // update transformations
    joints_[0]->forward_kinematics();

    // apply transformations to bone/joint-vertices and sticks
    update();
}

void Skeleton::transform(VecX& angles, Mat3X& axis, const Vec3& translation)
{	
    int base = joint_positions_.cols();

    for(unsigned int i = 0; i < joint_positions_.cols(); i++)
    {
        transformations_[i] = Eigen::Affine3f::Identity();
        if(i == 0)
        {
            //global translation/rotation
            transformations_[i].pretranslate(translation);
        }
        else
        {
            unsigned int p = joints_[i]->parent_->index_;
            int ji = joints_[i]->parent_->volumetric_jointindex_;
            Vec3 pV = orig_joint_positions_.col(p);
            //translate back that parent is origin
            transformations_[i].pretranslate(-pV);

            Mat33 R;
            R = Eigen::AngleAxisf(angles(i), axis.col(i));

            Mat33 Rj;
            Rj = Eigen::AngleAxisf(angles(i)/2, axis.col(i));

            //apply half rotation to joint, translate back and apply parent bone trafo (does not work i parant is computed afterwards!)
            if(ji != -1)
            {
                transformations_[base + ji] = Rj * transformations_[i];
                transformations_[base + ji].pretranslate(pV); //same translation
                transformations_[base + ji] = transformations_[p] * transformations_[base + ji]; //bonetranslation parent
            }

            // apply full rotation to bone, translate back and apply parent trafo
            transformations_[i].prerotate(R);
            transformations_[i].pretranslate(pV);
            transformations_[i] = transformations_[p] * transformations_[i];
        }
    }

    // apply transformations to bone/joint-vertices and sticks
    update();
}

float Skeleton::get_joint_radius_from_stickindex(usint si)
{
    if(joints_[si]->volumetric_jointindex_ >= 0)
        return vol_joints_[joints_[si]->volumetric_jointindex_].radius_;
    else
        return minimal_r_;
}

Vec3 Skeleton::get_projection_on_boneline(const Vec3& p, usint & bone)
{
	Vec3 bestProj(0,0,0);
	float mindist = 1e6;
    for(usint b = 0; b < bone_indices_.size(); b+=2)
	{
        usint b0 = bone_indices_[b];
        usint b1 = bone_indices_[b + 1];
		
		Vec3 vb0 = joint_positions_.col(b0);
		Vec3 vb1 = joint_positions_.col(b1);
		
        Vec3 pnear = get_nearest_point_on_line(p,vb0,vb1);
		
        float dist = (pnear - p).norm();
        if(dist < mindist)
		{
            mindist = dist;
			bestProj = pnear;
			bone = b/2;
		}
	}
	return bestProj;
}

void Skeleton::find_sliding_joints()
{
    IndexSet sj_candidates;
    if(is_character_mesh_)
    {
        if(joint_by_name_.find("vl5") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["vl5"]);
        if(joint_by_name_.find("vl2") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["vl2"]);
        if(joint_by_name_.find("vt10") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["vt10"]);
        if(joint_by_name_.find("r_knee") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["r_knee"]);
        if(joint_by_name_.find("l_knee") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["l_knee"]);
        if(joint_by_name_.find("r_hip") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["r_hip"]);
        if(joint_by_name_.find("l_hip") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["l_hip"]);
        if(joint_by_name_.find("l_shoulder") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["l_shoulder"]);
        if(joint_by_name_.find("r_shoulder") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["r_shoulder"]);
        if(joint_by_name_.find("l_elbow") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["l_elbow"]);
        if(joint_by_name_.find("r_elbow") != joint_by_name_.end()) sj_candidates.insert(joint_by_name_["r_elbow"]);
    }
    else
    {
        for(auto jo : joints_)
        {
            if(jo->is_root_ || jo->childreen_.size() != 1)
            {
                continue;
            }

            Vec3 dir0 = (jo->parent_->position_ - jo->position_).normalized();
            Vec3 dir1 = (jo->childreen_[0]->position_ - jo->position_).normalized();

            if(fabs(dir0.dot(dir1)) > 0.9)
            {
                sj_candidates.insert(jo->index_);
            }
        }
    }

    for(usint i : sj_candidates)
    {
        Joint *jo = joints_[i];
        if(jo->is_root_ || jo->childreen_.size() != 1)
        {
            std::cerr << "Problem! Wrong slidingjoint choosen: " << joints_[i]->name_ << std::endl; continue;
        }

        Sliding_Joint sj;
        sj.j = i;
        sj.b0 = jo->parent_bone_;
        sj.b1 = jo->childreen_[0]->parent_bone_;
        sj.j0 = jo->parent_->index_;
        sj.j1 = jo->childreen_[0]->index_;
        sj.jointindex = jo->volumetric_jointindex_;
        sliding_joints_.push_back(sj);
    }
}

int Skeleton::pick_joint(const Vec2& coord2d, const Mat44& mvp)
{
    float closest_dist = 1e10;

    int joint = 0;
    for (int i = 0; i < joint_positions_.cols(); ++i)
    {
        const Vec4 p(joint_positions_(3*i) , joint_positions_(3*i + 1) , joint_positions_(3*i + 2) , 1.0);
        Vec4 ndc = mvp * p;
        ndc /= ndc(3);

        const float d = (Vec2(ndc(0),ndc(1)) - coord2d).norm();
        if (d < closest_dist)
        {
            closest_dist = d;
            joint = (int) i;
        }
    }

    return joint;
}

Vec3 Skeleton::get_nearest_point_on_line(const Vec3& p, const Vec3& v0, const Vec3& v1)
{
	Vec3 a = (v1 - v0);
	float L = a.norm();
	a/=L;
	float Lnew = a.dot(p - v0);
	Lnew = (Lnew > L)? L : Lnew;
	Lnew = (Lnew < 0)? 0 : Lnew;
    return v0 + Lnew*a;
}

Vec3 Skeleton::get_skeleton_line_intersection(const Vec3 &start, const Vec3 &end, std::pair<Correspondence, int> &correspondence)
{
    float tmin = 1.5;
    float t;
    int bone = -1000;
    for(int i = 0; i < (int)bone_indices_.size(); i+=2)
    {
        bool can_be_on_cone = true;
        const Vec3& c0 = vol_bones_[i/2].v0_;
        const Vec3& c1 = vol_bones_[i/2].v1_;
        float r = vol_bones_[i/2].radius_;

        if(line_intersects_cylinder(c0,c1,start,end,r,t))
        {
            if(t < tmin && t > 0)
            {
                tmin = t;
                correspondence.first = BONE;
                correspondence.second = i/2;
                bone = i/2;
                can_be_on_cone = false;
            }
        }
        if(line_intersects_sphere(c0,start,end,r,t))
        {
           if(t < tmin && t > 0)
            {
                tmin = t;
                correspondence.first = BONE;
                correspondence.second = i/2;
                bone = i/2;
            }
        }
        if(line_intersects_sphere(c1,start,end,r,t))
        {
            if(t < tmin && t > 0)
            {
                tmin = t;
                correspondence.first = BONE;
                correspondence.second = i/2;
                bone = i/2;
            }
        }

        if(can_be_on_cone)
        {
            // test conic sections
            for(int k = 0; k < 2; k++)
            {
                if(!joints_[bone_indices_[i + k]]->is_leaf())
                {
                    const Vec3 &cb = (k == 0) ? c0 : c1;
                    float rj = get_joint_radius_from_stickindex(bone_indices_[i + k]);
                    float factor = 0.80;
                    if(line_intersects_conic(cb,joint_positions_.col(bone_indices_[i + k]),start, end,factor*r, factor*rj,t))
                    {
                        if(t < tmin && t > 0)
                        {
                            tmin = t;
                            correspondence.first = INTER;
                            correspondence.second = joints_[bone_indices_[i + k]]->volumetric_jointindex_;
                            bone = -joints_[bone_indices_[i + k]]->volumetric_jointindex_ - 500;
                        }
                    }
                }
            }
        }

    }

    for(int i = 0; i < (int)vol_joints_.size(); i++)
    {
        if(line_intersects_sphere(vol_joints_[i].c_,start, end,vol_joints_[i].radius_,t))
        {
            if(t < tmin && t > 0)
            {
                tmin = t;
                correspondence.first = JOINT;
                correspondence.second = i;
                bone = -i - 1;
            }
        }
    }

    if(bone == -1000)
    {
        // this should never happen but try to use 80% of correspondance line if it does
        std::cerr << "Problem with skeleton intersection!" << std::endl;
        correspondence.first = NONE;
        correspondence.second = -1;
        tmin = 0.8;
    }

    return start + tmin*(end - start);
}

bool Skeleton::line_intersects_sphere(const Vec3 &center, const Vec3 &start, const Vec3 &end, const float r, float &_t)
{
    Vec3 d = end - start;
    Vec3 sc = start - center;

    float a = d.dot(d);
    float b = d.dot(sc);
    float c = sc.dot(sc) - r*r;

    float diskriminant = b*b - a*c;
    if(diskriminant < 0)
        return false;

    float t = (-b - sqrt(diskriminant))/a;

    if(t > 0)
    {
        _t = t;
        return true;
    }

    return false;
}

bool Skeleton::line_intersects_cylinder(const Vec3 &c0, const Vec3 &c1, const Vec3 &start, const Vec3 &end, const float r, float &_t)
{
    Vec3 d = end - start;
    Vec3 bo = (c1 - c0).normalized();
    Vec3 sc = start - c0;

    float sc_dot_bo = sc.dot(bo);

    float a = d.dot(d) - d.dot(bo)*d.dot(bo);
    float b = d.dot(sc) - (d.dot(bo))*(sc_dot_bo);
    float c = sc.dot(sc) - sc_dot_bo*sc_dot_bo - r*r;

    float diskriminant = b*b - a*c;
    if(diskriminant < 0)
        return false;

    float t = (-b - sqrt(diskriminant))/a;

    if(t > 0)
    {
        Vec3 intersection = start + t*d;

        // check if in between caps
        if((intersection - c0).dot(bo) > 0 && (intersection - c1).dot(bo) < 0)
        {
            _t = t;
            return true;
        }
    }

    return false;
}

bool Skeleton::line_intersects_conic(const Vec3 &c0, const Vec3 &c1, const Vec3 &start, const Vec3 &end, const float r0, const float r1, float &_t)
{
    Vec3 d = end - start;
    Vec3 bo = (c1 - c0);
    float h = bo.norm();
    bo/=h;
    Vec3 sc = start - c0;
    float rh = (r1- r0)/h;

    float sc_dot_bo = sc.dot(bo);

    float a = d.dot(d) - d.dot(bo)*d.dot(bo)*(1 + rh*rh);
    float b = d.dot(sc) - (d.dot(bo))*(sc_dot_bo)*(1 + rh*rh) - (d.dot(bo))*rh*r0;
    float c = sc.dot(sc) - sc_dot_bo*sc_dot_bo*(1 + rh*rh) - r0*r0 - 2*rh*r0*sc_dot_bo;

    float diskriminant = b*b - a*c;
    if(diskriminant < 0)
        return false;

    float t = (-b - sqrt(diskriminant))/a;

    if(t > 0)
    {
        Vec3 intersection = start + t*d;

        // check if in between caps
        if((intersection - c0).dot(bo) > 0 && (intersection - c1).dot(bo) < 0)
        {
            _t = t;
            return true;
        }
    }

    return false;
}

void Skeleton::update()
{
    //apply transformations on bonelines
    for(unsigned int i = 0; i<joint_positions_.cols(); i++)
    {
        Vec3 v = orig_joint_positions_.col(i);
        joint_positions_.col(i) = transformations_[i] * v;
    }
    //apply transformations on bonevertices
    for(unsigned int i = 0; i< vol_bones_.size(); i++)
    {
        vertices_.block(0,bone_v_start_[i],3,vol_bones_[i].vertices_.cols()) = transformations_[bone_indices_[2*i]]*vol_bones_[i].vertices_;
    }
    //apply transformations on jointvertices
    for(unsigned int i = 0; i< vol_joints_.size(); i++)
    {
        vertices_.block(0,joint_v_start_[i],3,vol_joints_[i].vertices_.cols()) = transformations_[joint_positions_.cols() + vol_joints_[i].stickIndex_]*vol_joints_[i].vertices_;
    }
}

}
