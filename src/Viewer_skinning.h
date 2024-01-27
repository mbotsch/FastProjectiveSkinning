//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================
#pragma once
//=============================================================================


#include <pmp/visualization/TrackballViewer.h>
#include <pmp/visualization/Shader.h>
#include <pmp/Timer.h>
using namespace pmp;

#include "Animator.h"
#include "Inverse_kinematics.h"

#ifdef WITH_CUDA
#include <cuda_gl_interop.h>
#endif


//== CLASS DEFINITION =========================================================


/** 3D-Viewer for skeleton-based animation.
 The viewer class has a Character,
 which is controlled by manually selecting and manipulating joint angles,
 or through a pre-loaded animation, or through inverse kinematics
 \sa Character
 \sa Animation
 \sa Inverse_kinematics
**/

class Skinning_Viewer : public pmp::TrackballViewer
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

#ifdef WITH_CUDA
        cudaGraphicsResource *cuda_vbo_;
        cudaGraphicsResource *cuda_nbo_;
#endif
        Mesh_OGL_Data():
            vertex_array(0),vertex_buffer(0),normal_buffer(0),index_buffer(0)
#ifdef WITH_CUDA
           ,cuda_vbo_(0),cuda_nbo_(0)
#endif
        {}
    };

    /// constructor
    Skinning_Viewer(const char* _title, int _width, int _height, bool _showgui=true);
    ~Skinning_Viewer();

    void init(const std::string ini_filename, const char* skin_filename, const char* skel_filename, const char *skin_hr_filename, const char *us_filename);

private: // GUI functions

    /// render the scene
    virtual void draw(const std::string&) override;

    /// handle keyboard events
	virtual void keyboard(int key, int code, int action, int mods) override;

    /// this function handles mouse button events
    virtual void mouse(int button, int action, int mods) override;

    /// this function handles mouse motion (passive/active position)
    virtual void motion(double x, double y) override;

    /// this function triggers the time_integration() function
	virtual void do_processing() override;

    /// render/handle GUI
    virtual void process_imgui() override;

    /// draws the skeleton lines on the screen
    void draw_skeleton(const mat4& modelviewprojection);

    /// inits the opengl buffers
    void init_ogl_buffers(Mesh_OGL_Data &ogl_data, Projective_Skinning::Mat3X& vertices, Projective_Skinning::Mat3X& normals, Projective_Skinning::IndexVector& indices);

    /// updates the opengl buffers
    void upload_ogl_data();

    /// toggles drawmode of shrunken skin
    void toggle_shrunken(bool shrunken);

private: // simulation data and settings


    /// the loaded character to visualize
    Projective_Skinning::Animator animator_;

	/// is animation on/off?
	bool animate_;

    /// are we doing ik or animation?
    bool manipulate_;

    /// draw the skeleton?
    bool draw_skeleton_;

    /// sync to monitors framerate?
    bool vsync_;

    /// what to draw?
    enum Draw_options{DRAW_LR, DRAW_HR, DRAW_VOL_SKEL, DRAW_SHRUNKEN} draw_options_;

    /// draw wireframe
    bool wireframe_;

    /// do the simulation in each frame
    bool update_;

    /// print detailed timings to screen
    bool update_timings_;

    /// enable/disable collision detection
    bool collisions_;

    // timing helpers
    pmp::Timer timer_;
    float frame_time_;
    int fps_;
    float animation_speed_;

    // OpenGL stuff
    Shader mesh_shader_;
    Shader skeleton_shader_;

    Mesh_OGL_Data ogl_lr_mesh_;
    Mesh_OGL_Data ogl_hr_mesh_;
    Mesh_OGL_Data ogl_skeleton_mesh_;

    Mesh_OGL_Data *ogl_current_mesh_;

    // skeleton arrays
    GLuint skeleton_vertex_array_;
    GLuint skeleton_vertex_buffer_;
    GLuint skeleton_color_buffer_;

    Inverse_kinematics* ik_;

};


//=============================================================================
