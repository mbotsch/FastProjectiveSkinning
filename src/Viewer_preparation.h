//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================
#pragma once
//=============================================================================

//#define PMP_Scalar_TYPE_64

typedef double PSScalar;

#include <pmp/visualization/MeshViewer.h>
#include <pmp/visualization/Shader.h>
#include <Eigen/Dense>
#include "mesh/Skeleton.h"

//=============================================================================

class Preparation_Viewer : public pmp::MeshViewer
{
public:

    struct Mesh_OGL_Data
    {
        GLuint vertex_array;
        GLuint vertex_buffer;
        GLuint normal_buffer;
        GLuint index_buffer;

        Projective_Skinning::Mat3X *vertices;
        Projective_Skinning::Mat3X *normals;
        Projective_Skinning::IndexVector *indices;

        Mesh_OGL_Data():
            vertex_array(0),vertex_buffer(0),normal_buffer(0),index_buffer(0)
        {}
    };

    //! constructor
    Preparation_Viewer(const char* title, int width, int height);

    ~Preparation_Viewer();

    bool load_mesh(const char *filename) override;

    void build_from_ini(const char* ini_filename);

    std::string ini_filename_, skel_filename_, us_filename_, mesh_LR_filename_;
    bool ready_for_skinning_;

protected:
    //! this function handles keyboard events
    void keyboard(int key, int code, int action, int mod) override;

    void draw(const std::string &draw_mode) override;

    //! this function handles mouse button events
    virtual void mouse(int _button, int _action, int _mods) override;

    //! this function handles mouse motion (passive/active position)
    virtual void motion(double xpos, double ypos) override;

    //! this function renders the ImGUI elements and handles their events
    virtual void process_imgui() override;

private:
    int pick_joint(const Eigen::Vector2f &coord2d, const Eigen::Matrix4f &mvp);

    void init_ogl_buffers(Mesh_OGL_Data &ogl_data, Projective_Skinning::Mat3X& vertices, Projective_Skinning::Mat3X& normals, Projective_Skinning::IndexVector& indices);

    void find_low_to_high();

    bool computeGeodesics(int numNeighbors, float dist_factor);

    void fillLRLR();

    void compute_upsampling_weights(int num, int order, PSScalar (*wfunc)(PSScalar));

    void print_upsampling_weights(const char* filename);

    pmp::Shader skeleton_shader_, mesh_shader_;

    // skeleton buffer arrays
    GLuint skeleton_vertex_array_;
    GLuint skeleton_vertex_buffer_;
    GLuint skeleton_color_buffer_;

    // vol skeleton buffer arrays
    Mesh_OGL_Data ogl_skeleton_mesh_;

    int selected_joint_;

    // skeleton's joint positions
    std::vector<Eigen::Vector3f> joint_positions_;

    // skeleton's hirarchie (two indices define a joint connection)
    std::vector<unsigned int> joint_indices_;

    Projective_Skinning::Mat3X skin_vertices_;
    Projective_Skinning::Skeleton volumetric_skeleton_;

    pmp::SurfaceMeshGL mesh_HR_, mesh_LR_;

    bool decimated_;

    std::vector<int> low_to_high_;
    std::vector<int> high_to_low_;
    std::set<int> vertices_in_both_hr_lr_;

    std::vector<std::vector<int>> LR_neighbors_in_HR_;
    std::vector<std::vector<int>> LR_neighbors_;
    std::vector<std::vector<float>> neighbor_weights_;
    std::vector<std::vector<float>> neighborNi_;
    std::vector<std::vector<float>> neighborNNi_;

    int quadratic_;
    int decimation_percentage_;
    int num_us_neighbors_;
};

//=============================================================================
