//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include <fstream>
#include <map>
#include <utility> //pair

#include "Bone.h"
#include "Joint.h"

namespace Projective_Skinning
{

class Skeleton
{

public:    

    enum Correspondence {BONE, JOINT, INTER, IGNORED, NONE};

	//empty constructor
    Skeleton(){}
	
	//destructor
    ~Skeleton();
    
    // initializer
    bool init(const char* filename, const Mat3X &vertices);
    bool init(const Mat3X & positions, const IndexVector &indices, const Mat3X &vertices, const int moved = -1);

    // reset skeleton to restpose
    void reset();

    void move_joint(const int joint, const Vec3 &target);

    // stores current skeleton to a .skel2 file
    void store_skeleton(const char* filename);

    // transforms skeleton and joint-/bonevertices
    void transform();
    // transformation function using quaternions
    void transform(const std::vector<Eigen::Quaternionf> &quats);
	//transforms skeleton and joint-/bonevertices according to angle-axis-rotations
    void transform(VecX &angles, Mat3X &axis, const Vec3& translation = Vec3(0,0,0));

	//computes normals for bones and joints
    void compute_normals();

    // finds potential slidingjoints
    void find_sliding_joints();

    // picks joint that is closest to the 2d screen coordinate
    int pick_joint(const Vec2& coord2d, const Mat44& mvp);

    //returns the joint radius of the joint corresponding to the stickindex si returns minimal_r if no joint was found
    float get_joint_radius_from_stickindex(usint si);
    // returns intersection of a line from start to end and the volumetric skeleton and whether this intersection is on a bone/joint/in between is stored in vbj
    Vec3 get_skeleton_line_intersection(const Vec3 &start, const Vec3 &end, std::pair<Correspondence, int> &correspondence);
    //get bonelineprojection on skeleton used for shrinkpairlaplace returns projection and the bone on which the projection lies in bone \todo better shrinkpairlaplace would get rid of this
    Vec3 get_projection_on_boneline(const Vec3 &p, usint &bone);

private:

    //.skel2 parser
    void read_skeleton(const char* filename, const Mat3X &vertices);

    // helper functions that compute line/sphere, line/cylinder or line/conic intersections
    bool line_intersects_sphere(const Vec3 &center, const Vec3 &start, const Vec3 &end, const float r, float &_t);
    bool line_intersects_cylinder(const Vec3& c0, const Vec3& c1, const Vec3 &start, const Vec3 &end, const float r, float &_t);
    bool line_intersects_conic(const Vec3& c0, const Vec3& c1, const Vec3 &start, const Vec3 &end, const float r0, const float r1, float &_t);

    // helper function that finds nearest mesh point for a given Point, and start and EndPoint of a LineSegment
    Vec3 get_nearest_point_on_line(const Vec3& p, const Vec3& v0, const Vec3& v1);

    //creates volumetric bones and joints and their vertices
    void create_volumetric_skeleton(std::vector<float> &bone_radii);

    // updates bone and jont vertices
    void update();
    
public:

    struct Sliding_Joint
	{
		//. _____ . _____ .
		//j0 b0   j  b1  j1
		
        int j,b0,b1,j0,j1,jointindex;
	};
	
    /// bones radii are set from bonedistfacor*(minimum distance to mesh)
    static const float bone_dist_factor_;
	
    /// volumetric bones and joints
    std::vector<VolJoint> vol_joints_;
    std::vector<VolBone> vol_bones_;
    
    Mat3X joint_positions_;// todo: get rid of?
    Mat3X orig_joint_positions_;// todo: get rid of?
    IndexVector bone_indices_;

    std::vector<Eigen::Affine3f> transformations_;
	
    Mat3X vertices_;
	Mat3X normals_;
    IndexVector indices_;
	IndexVector bone_v_start_;
    IndexVector joint_v_start_;
	float minimal_r_;
	
    std::vector<Sliding_Joint> sliding_joints_;
	
    std::map<std::string, usint> joint_by_name_;
    std::map<usint, std::string> name_by_joint_;
    std::vector<bool> is_hand_bone_;

    bool is_character_mesh_;

    std::vector<Joint*> joints_;

    std::vector<float> orig_bone_radii_;
};

}
