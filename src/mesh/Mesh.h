//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include <cfloat>
#include <algorithm>
#include <pmp/Timer.h>

#include "Skeleton.h"

namespace Projective_Skinning
{

class Mesh
{
    struct Skin
    {
        IndexVector sim_indices;
        IndexVector all_indices;
        unsigned int num_vertices;
        IndexVector assoc_bone;
        IndexVector edges;
        std::vector<IndexSet> neighbors_;
        std::vector<IndexSet> neighbors_2_ring;
        float avg_edge_length;
    };

    struct Tets
    {
        IndexVector indices;
        IndexVector collision_indices;
        IndexVector collision_tet_to_trig;
    };

    struct SJ_Interpolation_Parameters
    {
        usint i00,i01,i10,i11;
        float p_angle,p_h;
    };


public:
    Mesh(bool sj);

    // initializer functions from .off file and skeleton
    bool init(const char* skinfilename, const char *skel_filename,
              const bool shrink, const bool diffmasses, const float mass, const char* skin_hr_filename, const char* upsampling_filename);


    // store mesh to .off file
    void store_OFF(const char* off_filename, bool high_res = false);

    // resets mesh to initial configuration
    void reset(bool withshrinkV = true);

    // transformation for collision tetrahedra
    void transform_collision_tet_basepoints_();

    // updates vertices anchored to bones and ignored ones
    void update_anchors();

    // visualization Mesh lading and transformation (upsampling)
    bool load_upsampling_file(const char* filename);
    void upsample();

    // update face and vertex normals
    void compute_normals();
    void compute_normals_HR();

    // compute adjusted weightfactors for humanoid meshes (eg. make fingers and hands harder) and de-dimensionalize weights
    void adjust_tet_weights_(std::vector<float> &weight_factors);

private:

    // .off file loader
    bool load_OFF(const char* filename, Mat3X &verts, IndexVector &inds);

    // loads relevant upsampling data
    bool load_HR(const char *skin_hr_filename, const char* us_parameter_filename);


    // shrink skin to bones per laplace smoothing
    void shrink_skin_to_bones(IndexVector &vertexIsIgnored);

    // skin shrinking helper functions
    // transform mesh such that parent and child bones are parallel
    void stretch_out(Mat3X &shrinkV, VecX &angles, Mat3X &axis);
    // initializes shrinking correspondences with nearest neighbors
    void set_shrink_pairs_to_nearest(Mat3X& shrinkpairs, IndexVector &vertexIsIgnored);
    // smooth the shrinking correspondences such that neighboring skin vertices have neighboring correspondences
    void shrink_pair_smoothing(Mat3X &shrinkV, IndexVector &vertexIsIgnored);
    // shrinks the skin onto the volumetric skeleton using the correnspondences
    void move_to_shrinkpairs(Mat3X &shrinkV, IndexVector &vertexIsIgnored);

    // tetrahedralize with shrinked skin
    void tetrahedralize();
    void init_collision_tets();
    void init_simulation_tets();
    // solves tet problem arising when spitting non planar quadrangle by splitting neighboring quadrangels on the same diagonal
    void find_edgedirections(std::map<std::pair<usint, usint>, bool> &edge_directions);

    // skin sliding
    void init_sliding_joints(Mat3X &sj_refs, std::map<usint, IndexVector> &sjIndices);
    void skin_sliding();

    // find meshs edges and one- and two-ring neighbors
    void find_edges_and_neighbors();

    // finds ignored vertices (face for example if this should be animated with blendshapes) based on ignored bones
    void find_ignored_vertices(IndexVector &vertexIsIgnored);
    void add_non_ignored_neighbors_to_cluster(IndexSet &cluster, IndexSet &remaining, usint vstart, IndexVector &vertexIsIgnored);
    void set_ignored_bones();

    // computes AABB of entire mesh
    void computeBB();

    // creates the typical vertex order: 1. skin, 2. nonrigid shrunken skin, 3. simple rigid shrunken skin, 4. sliding shrunken, 5. ignored vertices
    void create_sorted_vertices(IndexVector &vertexIsIgnored, Mat3X &sj_refs, std::map<usint, IndexVector> &sjIndices);
    void resort_data(IndexVector &vertexOrder, IndexVector &vertexIsIgnored);

    // computes vertexmasses based on adjoining tetrahedra volumes
    void compute_masses(float totalmass);

    // initializes neighborfaces witch is used in comute normals
    void setup_normals(const int nV, const IndexVector & indices);

    // prints some mesh statistics
    void print_statistics();

public:
    // basic structs
    Skin skin_;
    Tets tets_;

    // mesh's skeleton
    Skeleton skeleton_;

    // mesh normals
    Mat3X vertex_normals_;
    Mat3X face_normals_;
    std::vector<std::vector<unsigned int>> neighbor_faces_;

    // all mesh vertices
    Mat3X vertices_;
    Mat3X orig_vertices_;

    // bounding box
    Vec3 bbMax_, bbMin_;

    // boneline correspondences building the collision tetraheda bases
    Mat3X collision_tet_basepoints_;
    Mat3X orig_collision_tet_basepoints_;
    std::vector<int> coltet_transform_indices_;

    // ignored vertices variables
    IndexSet ignored_bones_;
    IndexSet ignored_neighbors_;

    std::vector<int> transform_indices_;
    std::vector<std::pair<Skeleton::Correspondence,int>> shrinking_correspondences_;

    // HR variables
    Mat3X high_res_vertices_;
    IndexVector high_res_indices_;
    Mat3X high_res_vertex_normals_;
    Mat3X high_res_face_normals_;
    bool use_high_res_;

    //sliding joint variables
    std::vector<SJ_Interpolation_Parameters> sj_parameters_; // use this in future
    const bool use_sliding_joints_;

    // upsampling
    std::vector<std::vector<unsigned int>> us_neighors_;
    std::vector<std::vector<float>> us_Nij_;
    std::vector<std::vector<float>> us_normal_Nij_;
    int num_us_neighbors_;
    bool use_new_us_normals_ = true;
    IndexVector old_to_new_;

    // offsets
    int num_simulated_skin_;
    int num_non_rigid_;
    int num_rigid_;
    int num_simple_rigid_;
    int num_ignored_;
    int num_sliding_;
    int num_sliding_references_;

    // base indices
    int base_rigid_;
    int base_simple_rigid_;
    int base_ignored_;
    int base_sliding_;
    int base_sliding_references_;

    // additional anchors used for anchoring the boundary of ignored regions (face)
    IndexSet additional_anchors_;
    IndexVector additional_anchor_bones_;

    // mass of each vertex
    std::vector<float> vertex_masses_;
};

}

