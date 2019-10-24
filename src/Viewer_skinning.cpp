//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

//== INCLUDES =================================================================


#include <pmp/visualization/GL.h>
#include <imgui.h>
#include "Shaders.h"
#include "Viewer_skinning.h"


//== IMPLEMENTATION ==========================================================


Skinning_Viewer::Skinning_Viewer(const char* _title, int _width, int _height, bool _showgui)
: TrackballViewer(_title, _width, _height, _showgui),
  draw_options_(DRAW_LR),
  wireframe_(false),
  frame_time_(0.0),
  animation_speed_(0.8),
  ogl_current_mesh_(nullptr)
{
    clear_help_items();
    add_help_item("Space", "Start/stop simulation");
    add_help_item("A", "Start/stop animation");
    add_help_item("S", "Toggle skeleton rendering");
    add_help_item("H", "Toggle HR/LR rendering");
    add_help_item("W", "Toggle edge rendering");
    add_help_item("1-9", "Select animation mode");
    add_help_item("Up/Down", "Select next/prev joint");
    add_help_item("R", "Unset selected effector");
    add_help_item("Backspace", "Reset posture and effectors");
    add_help_item("Left mouse", "Rotate character");
    add_help_item("Right mouse", "Zoom character");
    add_help_item("Mouse wheel", "Zoom character");
    add_help_item("Ctrl + Left mouse", "Select/drag joint");
    add_help_item("G", "Toggle GUI dialog");
    add_help_item("F", "Toggle fullscreen");
    add_help_item("PageUp/Down", "Scale GUI dialogs");
    add_help_item("Esc/Q", "Quit application");
}


Skinning_Viewer::~Skinning_Viewer()
{
#ifdef WITH_CUDA
    if(ogl_lr_mesh_.cuda_vbo_)
        checkCudaErrors(cudaGraphicsUnregisterResource(ogl_lr_mesh_.cuda_vbo_));
    if(ogl_lr_mesh_.cuda_nbo_)
        checkCudaErrors(cudaGraphicsUnregisterResource(ogl_lr_mesh_.cuda_nbo_));
#endif

    if (ogl_lr_mesh_.vertex_buffer)    glDeleteBuffers(1, &ogl_lr_mesh_.vertex_buffer);
    if (ogl_lr_mesh_.normal_buffer)    glDeleteBuffers(1, &ogl_lr_mesh_.normal_buffer);
    if (ogl_lr_mesh_.index_buffer)    glDeleteBuffers(1, &ogl_lr_mesh_.index_buffer);
    if (ogl_lr_mesh_.vertex_array)     glDeleteVertexArrays(1, &ogl_lr_mesh_.vertex_array);

#ifdef WITH_CUDA
    if(ogl_hr_mesh_.cuda_vbo_)
        checkCudaErrors(cudaGraphicsUnregisterResource(ogl_hr_mesh_.cuda_vbo_));
    if(ogl_hr_mesh_.cuda_nbo_)
        checkCudaErrors(cudaGraphicsUnregisterResource(ogl_hr_mesh_.cuda_nbo_));
#endif

    if (ogl_hr_mesh_.vertex_buffer)    glDeleteBuffers(1, &ogl_hr_mesh_.vertex_buffer);
    if (ogl_hr_mesh_.normal_buffer)    glDeleteBuffers(1, &ogl_hr_mesh_.normal_buffer);
    if (ogl_hr_mesh_.index_buffer)      glDeleteBuffers(1, &ogl_hr_mesh_.index_buffer);
    if (ogl_hr_mesh_.vertex_array)     glDeleteVertexArrays(1, &ogl_hr_mesh_.vertex_array);


    if (skeleton_vertex_buffer_)      glDeleteBuffers(1, &skeleton_vertex_buffer_);
    if (skeleton_color_buffer_)      glDeleteBuffers(1, &skeleton_color_buffer_);
    if (skeleton_vertex_array_)     glDeleteVertexArrays(1, &skeleton_vertex_array_);

    delete ik_;
}

void Skinning_Viewer::init(const char *skin_filename, const char *skel_filename, const char* skin_hr_filename, const char* us_filename, const std::string anim_filename, const std::string anim_base)
{

    animator_.init(skin_filename, skel_filename, skin_hr_filename, us_filename);

#ifndef WITH_CUDA

#undef D_GPU
#define D_GPU false

#endif

    ik_ = new Inverse_kinematics(animator_.mesh_.skeleton_);

    init_ogl_buffers(ogl_lr_mesh_,animator_.mesh_.vertices_, animator_.mesh_.vertex_normals_, animator_.mesh_.skin_.all_indices);
    init_ogl_buffers(ogl_skeleton_mesh_,animator_.mesh_.skeleton_.vertices_, animator_.mesh_.skeleton_.normals_, animator_.mesh_.skeleton_.indices_);
    //init_ogl_buffers(ogl_skeleton_mesh_,bskel_.vertices_, bskel_.normals_, bskel_.indices_);

    if(animator_.mesh_.use_high_res_)
        init_ogl_buffers(ogl_hr_mesh_,animator_.mesh_.high_res_vertices_, animator_.mesh_.high_res_vertex_normals_, animator_.mesh_.high_res_indices_);

    ogl_current_mesh_ = &ogl_lr_mesh_;

    skeleton_vertex_array_ = 0;
    skeleton_vertex_buffer_ = 0;
    skeleton_color_buffer_ = 0;

    fps_ = 0;

    animate_            = false;
    manipulate_         = false;
    draw_skeleton_      = false;
    vsync_              = false;
    update_             = false;
    update_timings_     = false;
    collisions_         = false;

    glfwSwapInterval(0);

    // load shaders
    if (!mesh_shader_.source(character_vshader, character_fshader))
    {
        std::cerr << "Error: Could not load shaders.\n";
        exit(EXIT_FAILURE);
    }

    if (!skeleton_shader_.source(skeleton_vshader, skeleton_fshader))
    {
        std::cerr << "Error: Could not load skeleton shaders.\n";
        exit(EXIT_FAILURE);
    }

    // suppress unused variable warnings
    (void)mesh_vshader;
    (void)mesh_fshader;

    upload_ogl_data();
    Projective_Skinning::Vec3 c(0.5*(animator_.mesh_.bbMin_ + animator_.mesh_.bbMax_));
    float r = 0.5*(animator_.mesh_.bbMax_ - animator_.mesh_.bbMin_).norm();
    set_scene(vec3(c(0),c(1),c(2)), r);

    // if there exists a HR mesh --> draw it
    if(animator_.mesh_.use_high_res_)
    {
        draw_options_ = DRAW_HR;
        ogl_current_mesh_ = &ogl_hr_mesh_;
        upload_ogl_data();
    }

    if(!anim_filename.empty())
        animator_.load_animation(anim_filename, anim_base);
}


//-----------------------------------------------------------------------------


void Skinning_Viewer::keyboard(int key, int code, int action, int mods)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT)
        return;

    switch (key)
    {
        // toggle pause
        case GLFW_KEY_SPACE:
        {
            update_ = !update_;
            break;
        }

        // toggle animation
        case GLFW_KEY_A:
        {
            animate_ = !animate_;
            if(animate_) update_ = true;
            break;
        }

        // toggle wireframe
        case GLFW_KEY_W:
        {
            wireframe_ = !wireframe_;
            break;
        }

        // toggle high res
        case GLFW_KEY_H:
        {
            if(animator_.mesh_.use_high_res_)
            {
                if(draw_options_ == DRAW_HR)
                {
                    draw_options_ = DRAW_LR;
                    ogl_current_mesh_ = &ogl_lr_mesh_;
                    upload_ogl_data();
                }
                else if(draw_options_ == DRAW_LR)
                {
                    draw_options_ = DRAW_HR;
                    ogl_current_mesh_ = &ogl_hr_mesh_;
                    upload_ogl_data();
                }
            }
            break;
        }

        //toggle skeleton
        case GLFW_KEY_S:
        {
            draw_skeleton_ = !draw_skeleton_;
            break;
        }

        //reset posture
        case GLFW_KEY_R:
        {
            // todo: move to ik function
            if(animator_.get_active_bone() >= 0)
            {
                animator_.mesh_.skeleton_.joints_[animator_.get_active_bone()]->is_effector_ = false;
            }
            break;
        }
        case GLFW_KEY_BACKSPACE:
        {
            animator_.reset();
            view_all();
            break;
        }

        case GLFW_KEY_UP:
        {
            animator_.increase_bone();
            break;
        }

        case GLFW_KEY_DOWN:
        {
            animator_.decrease_bone();
            break;
        }

        case GLFW_KEY_1:case GLFW_KEY_2:case GLFW_KEY_3:case GLFW_KEY_4:case GLFW_KEY_5:case GLFW_KEY_6:case GLFW_KEY_7:case GLFW_KEY_8:case GLFW_KEY_9:
        {
            int pressed_number = static_cast<int>(key - GLFW_KEY_0);
            animator_.set_animation_mode(pressed_number - 1);
            animate_ = true;
            break;
        }

        default:
        {
            TrackballViewer::keyboard(key, code, action, mods);
            break;
        }
    }
}


//-----------------------------------------------------------------------------


void Skinning_Viewer::do_processing()
{
    if(!update_ || !D_SIMULATE || !D_SHRINKSKIN) return;
    static float time = 0.0;
    static int its = 0;
    float dt = 0;

    // update frametime
    if(its != 0)
        dt = timer_.stop().elapsed();

    its++;

    time += dt;
    if(its%100 == 0)
    {
        frame_time_ = time/100.0;
        fps_ = static_cast<int>(1000.0/frame_time_ + 0.5);
        time = 0;
    }

    timer_.start();

    float *dptr = nullptr, *nptr = nullptr;
#ifdef WITH_CUDA
    // get cuda pointer from ogl buffer
    checkCudaErrors(cudaGraphicsMapResources(1, &ogl_current_mesh_->cuda_vbo_, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &ogl_current_mesh_->cuda_nbo_, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, ogl_current_mesh_->cuda_vbo_));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&nptr, &num_bytes, ogl_current_mesh_->cuda_nbo_));
#endif

    // force maximal dt to be that for 60fps to prevent very large jumps in joint angles
    dt = std::min(dt, 1000.0f/60.0f);

    // update mesh
    animator_.update_mesh(animate_,animation_speed_*dt,draw_options_ == DRAW_HR,D_GLOBAL_COLLISIONS && collisions_,update_timings_, dptr, nptr);

#ifdef WITH_CUDA
    // unmap cuda buffers
    checkCudaErrors(cudaGraphicsUnmapResources(1, &ogl_current_mesh_->cuda_vbo_, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &ogl_current_mesh_->cuda_nbo_, 0));
#endif

    if(!D_GPU || draw_options_ == DRAW_VOL_SKEL)
        upload_ogl_data();

}


//-----------------------------------------------------------------------------


void Skinning_Viewer::process_imgui()
{
    if(D_GLOBAL_COLLISIONS)
    {
        bool col = collisions_;
        ImGui::Checkbox("Handle Collsions", &collisions_);
        if(col != collisions_ && !collisions_)
        {
            animator_.reset_collisions();
        }

    }

    ImGui::Text("%d FPS", fps_);


    ImGui::Spacing();
    ImGui::Spacing();

    if(ImGui::CollapsingHeader("Animation Options", false))
    {
        if(animate_)
        {
            if(animator_.current_animation_ != -1)
            {
                ImGui::Text("Animation %d/%d", animator_.current_animation_ + 1, animator_.num_available_animations_);
                ImGui::SameLine();
                if(ImGui::Button(" next "))
                    animator_.set_animation_mode((animator_.current_animation_ + 1)%animator_.num_available_animations_);
            }
            else
            {
                ImGui::Text("Animating current Joint");
            }
        }

        if(ImGui::Checkbox("Animate", &animate_))
            update_ = true;


        ImGui::PushItemWidth(120);
        ImGui::Text("Animation Speed");
        ImGui::SliderFloat("##Animation Speed", &animation_speed_, 0.01f, 2.0f, "%.2f");
        ImGui::PopItemWidth();

        if (animator_.get_active_bone() != -1)
        {
            int j = animator_.get_active_bone();
            ImGui::Text("Selected Joint: %s", animator_.mesh_.skeleton_.name_by_joint_[j].c_str());
            if (ImGui::Button("prev"))
            {
                animator_.decrease_bone();
            }
            ImGui::SameLine();
            if (ImGui::Button("next"))
            {
                animator_.increase_bone();
            }

            ImGui::Spacing();
            ImGui::Spacing();

            if(ImGui::Button("Animate selected joint"))
            {
                animator_.animate_current_joint_ = true;
                animator_.current_animation_ = -1;
                update_ = true; animate_ = true;
            }

            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::Text("Rotate selected joint");
            if (ImGui::Button("X+") && !animate_)  {animator_.rotate_current_joint(0,  5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}
            ImGui::SameLine();
            if (ImGui::Button("X-") && !animate_)  {animator_.rotate_current_joint(0, -5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}
            ImGui::SameLine();
            if (ImGui::Button("Y+") && !animate_)  {animator_.rotate_current_joint(1,  5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}
            ImGui::SameLine();
            if (ImGui::Button("Y-") && !animate_)  {animator_.rotate_current_joint(1, -5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}
            ImGui::SameLine();
            if (ImGui::Button("Z+") && !animate_)  {animator_.rotate_current_joint(2,  5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}
            ImGui::SameLine();
            if (ImGui::Button("Z-") && !animate_)  {animator_.rotate_current_joint(2, -5.0f*M_PI/180.0f); update_ = true;}//manipulate_ = true;}

        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if(ImGui::CollapsingHeader("Rendering Options", false))
    {
        int drawop = static_cast<int>(draw_options_);

        ImGui::RadioButton("LR Mesh",&drawop,0);
        if(animator_.mesh_.use_high_res_)
            ImGui::RadioButton("HR Mesh",&drawop,1);
        ImGui::RadioButton("Volumetric Skeleton",&drawop,2);
        ImGui::RadioButton("Shrunken Skin",&drawop,3);

        if(static_cast<int>(draw_options_) != drawop)
        {
            if(drawop == 3 || draw_options_ == DRAW_SHRUNKEN)
            {
                toggle_shrunken(drawop == 3);
            }

            draw_options_ = static_cast<Draw_options>(drawop);
            switch(draw_options_)
            {
            case DRAW_LR: case DRAW_SHRUNKEN:
                ogl_current_mesh_ = &ogl_lr_mesh_;break;
            case DRAW_HR:
                ogl_current_mesh_ = &ogl_hr_mesh_;break;
            case DRAW_VOL_SKEL:
                ogl_current_mesh_ = &ogl_skeleton_mesh_;break;
            }
            upload_ogl_data();
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Checkbox("Wireframe", &wireframe_);

        ImGui::Checkbox("Draw Skeleton", &draw_skeleton_);

        bool vsy = vsync_;
        ImGui::Checkbox("V-Sync", &vsync_);
        if(vsy != vsync_)
        {
            glfwSwapInterval((int)vsync_);
        }

    }

    ImGui::Spacing();
    ImGui::Spacing();

    if(ImGui::CollapsingHeader("Timing Details", false))
    {
        float s = imgui_scaling();
        update_timings_ = true;
        ImGui::Text("PD update: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_simulation_time_);
        ImGui::Text("Collision update: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_collision_time_);
        ImGui::Text("Skeleton update: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_skeleton_time_);
        ImGui::Text("Normal update: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_normal_time_);
        ImGui::Text("Upsampling: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_upsampling_time_);
        ImGui::Text("Rendering: ");
        ImGui::SameLine(s*110); ImGui::Text("%.3fms", animator_.avg_draw_time_);
    }
    else
    {
        update_timings_ = false;
    }
}


void Skinning_Viewer::draw_skeleton(const mat4 &modelviewprojection)
{
    using namespace Projective_Skinning;

    Mat3X &joints = animator_.mesh_.skeleton_.joint_positions_;
    IndexVector &indices = animator_.mesh_.skeleton_.bone_indices_;

    int n_joints = joints.cols();
    std::vector<vec3> colors;

    const vec3 grey  = vec3(0.8f);
    const vec3 red   = vec3(0.8f,0.2f,0.2f);
    const vec3 green = vec3(0.2f,0.8f,0.2f);

    colors.resize(n_joints,grey);


    // setup OpenGL
    skeleton_shader_.use();
    skeleton_shader_.set_uniform("modelview_projection_matrix", modelviewprojection);
    glDisable( GL_DEPTH_TEST );


    // generate array and buffers
    if (skeleton_vertex_array_ == 0)
    {
        glGenVertexArrays(1, &skeleton_vertex_array_);
        glBindVertexArray(skeleton_vertex_array_);
        glGenBuffers(1, &skeleton_vertex_buffer_);
        glGenBuffers(1, &skeleton_color_buffer_);
    }
    else glBindVertexArray(skeleton_vertex_array_);


    // upload data
    glBindBuffer(GL_ARRAY_BUFFER, skeleton_vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, n_joints*3*sizeof(float), joints.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, skeleton_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // draw as lines
    glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, indices.data());


    if(animator_.get_active_bone() >= 0)
    {
        // draw effectors and active bone
        std::vector<int> effector_indices;
        colors.clear();
        colors.resize(n_joints,green);

        effector_indices.push_back(animator_.get_active_bone());
        for(auto jo : animator_.mesh_.skeleton_.joints_)
        {
            if(jo->is_effector_ && jo->index_ != animator_.get_active_bone())
            {
                colors[jo->index_] = red;
                effector_indices.push_back(jo-> index_);
            }
        }

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, skeleton_color_buffer_);
        glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glDrawElements(GL_POINTS, effector_indices.size(), GL_UNSIGNED_INT, effector_indices.data());


        // draw lines to ik targets
        std::vector<Vec3> targets;
        for(auto jo : animator_.mesh_.skeleton_.joints_)
        {
            if(jo->is_effector_)
            {
                targets.push_back(joints.col(jo->index_));
                targets.push_back(jo->target_);
            }
        }
        colors.clear();
        colors.resize(targets.size(),red);

        glBindBuffer(GL_ARRAY_BUFFER, skeleton_vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, targets.size()*3*sizeof(float), targets.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, skeleton_color_buffer_);
        glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
        glEnable(GL_PROGRAM_POINT_SIZE);

        glDrawArrays(GL_LINES, 0, targets.size());
    }
}

//----------------------------------------------------------------------------

void Skinning_Viewer::upload_ogl_data()
{
    glBindVertexArray(ogl_current_mesh_->vertex_array);

    if (ogl_current_mesh_->vertices->cols() > 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_current_mesh_->vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, ogl_current_mesh_->vertices->cols()*3*sizeof(float), ogl_current_mesh_->vertices->data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }

    if (ogl_current_mesh_->normals->cols() > 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_current_mesh_->normal_buffer);
        glBufferData(GL_ARRAY_BUFFER, ogl_current_mesh_->vertices->cols()*3*sizeof(float), ogl_current_mesh_->normals->data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
    }
}

void Skinning_Viewer::toggle_shrunken(bool shrunken)
{
    Projective_Skinning::IndexVector indices;
    if(shrunken)
    {
        indices = animator_.mesh_.skin_.sim_indices;
        for(auto &i : indices)
            i += animator_.mesh_.num_simulated_skin_;
    }
    else
    {
        indices = animator_.mesh_.skin_.all_indices;
    }

    // swap index buffer
    glBindVertexArray(ogl_lr_mesh_.vertex_array);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_lr_mesh_.index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


//-----------------------------------------------------------------------------


void Skinning_Viewer::init_ogl_buffers(Mesh_OGL_Data &ogl_data, Projective_Skinning::Mat3X& vertices, Projective_Skinning::Mat3X& normals, Projective_Skinning::IndexVector& indices)
{
    ogl_data.vertices = &vertices;
    ogl_data.normals = &normals;
    ogl_data.indices = &indices;

    glGenVertexArrays(1, &ogl_data.vertex_array);
    glBindVertexArray(ogl_data.vertex_array);
    glGenBuffers(1,&ogl_data.vertex_buffer);
    glGenBuffers(1,&ogl_data.normal_buffer);
    glGenBuffers(1,&ogl_data.index_buffer);

    glBindVertexArray(ogl_data.vertex_array);

    if (vertices.cols() > 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_data.vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, vertices.cols()*3*sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }

    if (normals.cols() > 0)
    {
        glBindBuffer(GL_ARRAY_BUFFER, ogl_data.normal_buffer);
        glBufferData(GL_ARRAY_BUFFER, vertices.cols()*3*sizeof(float), normals.data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);
    }

    // init index buffer just once
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_data.index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // init cuda buffers
#ifdef WITH_CUDA

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ogl_data.cuda_vbo_, ogl_data.vertex_buffer, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&ogl_data.cuda_nbo_, ogl_data.normal_buffer, cudaGraphicsMapFlagsWriteDiscard));

#endif
}

//-----------------------------------------------------------------------------


void Skinning_Viewer::draw(const std::string&)
{
    // OpenGL state
    glEnable( GL_DEPTH_TEST );

    // clear screen
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // compute required matrices
    mat3 normalMatrix = inverse(transpose(linear_part(modelview_matrix_)));
    mat4 modelviewprojectionMatrix = projection_matrix_ * modelview_matrix_;

    // draw character mesh
    glFrontFace(GL_CCW);
    glEnable( GL_DEPTH_TEST );
    glBindVertexArray(ogl_current_mesh_->vertex_array);

    mesh_shader_.use();
    mesh_shader_.set_uniform("modelview_projection_matrix", modelviewprojectionMatrix);
    mesh_shader_.set_uniform("normal_matrix", normalMatrix);

    int n_vertices = (GLsizei)ogl_current_mesh_->indices->size();

    mesh_shader_.set_uniform("use_texture", false);
    mesh_shader_.set_uniform("use_srgb",    false);
    mesh_shader_.set_uniform("use_black", false);

    if(draw_options_ == DRAW_SHRUNKEN)
    {
        // draw edges of shrunken skin
        mesh_shader_.set_uniform("use_black", true);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_TRIANGLES, n_vertices, GL_UNSIGNED_INT, NULL);
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    }
    else
    {
        if(!wireframe_)
        {
            glDrawElements(GL_TRIANGLES, n_vertices, GL_UNSIGNED_INT, NULL);
        }
        else
        {
            // draw faces
            glDepthRange(0.01, 1.0);
            glDrawElements(GL_TRIANGLES, n_vertices, GL_UNSIGNED_INT, NULL);

            // overlay edges
            glDepthRange(0.0, 1.0);
            glDepthFunc(GL_LEQUAL);
            mesh_shader_.set_uniform("use_black", true);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements(GL_TRIANGLES, n_vertices, GL_UNSIGNED_INT, NULL);
            glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
            glDepthFunc(GL_LESS);
        }
    }


    if(draw_skeleton_)
        draw_skeleton(modelviewprojectionMatrix);
}


//----------------------------------------------------------------------------


void Skinning_Viewer::mouse(int _button, int _action, int _mods)
{
    // CTRL not pressed -> rotate model
    if (!ctrl_pressed())
    {
        TrackballViewer::mouse(_button, _action, _mods);
        return;
    }


    // CTRL pressed -> drag joints around
    if (_action == GLFW_PRESS && ! animate_)
    {
        // select joint
        if (_button==GLFW_MOUSE_BUTTON_LEFT)
        {
            // get point under mouse cursor
            double x, y;
            cursor_pos(x,y);

            // get viewport paramters
            vec4 vp(0.0f);
            glGetFloatv(GL_VIEWPORT, &vp[0]);

            // in OpenGL y=0 is at the 'bottom'
            y = vp[3] - y;

            // invert viewport mapping
            vec2 p2d;
            p2d[0] = ((float)x - (float) vp[0]) / ((float) vp[2]) * 2.0f - 1.0f;
            p2d[1] = ((float)y - (float) vp[1]) / ((float) vp[3]) * 2.0f - 1.0f;

            // find closest joint
            mat4 mvp = projection_matrix_ * modelview_matrix_;
            Projective_Skinning::Mat44 mvp_ps(mvp.data());
            int j = animator_.mesh_.skeleton_.pick_joint(Projective_Skinning::Vec2(p2d[0], p2d[1]), mvp_ps);

            // select this joint
            animator_.set_active_bone(j);

            // set it as IK effector
            ik_->set_effector(j, animator_.mesh_.skeleton_.joint_positions_.col(j));

//            manipulate_ = true;
        }
    }


}


//-----------------------------------------------------------------------------


void Skinning_Viewer::motion(double _x, double _y)
{
    // CTRL not pressed -> rotate model
    if (!ctrl_pressed())
    {
        TrackballViewer::motion(_x,_y);
        return;
    }


    if (left_mouse_pressed() && !animate_)
    {
        using namespace Projective_Skinning;

        Vec3 p = animator_.mesh_.skeleton_.joint_positions_.col(animator_.get_active_bone());
        //Vec3 p = bskel_.stickPositions_.col(1);

        // project joint position to 2D
        mat4 mvp = projection_matrix_ * modelview_matrix_;
        vec4 pp  = mvp * vec4(p(0),p(1),p(2), 1.0f);
        pp /= pp[3];

        // get viewport data
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        int x = (int)_x;
        int y = (int)_y;

        // in OpenGL y=0 is at the 'bottom'
        y = viewport[3] - y;

        // unproject mouse position to 3D
        vec4 pn;
        pn[0] = ((float)x - (float) viewport[0]) / ((float) viewport[2]) * 2.0f - 1.0f;
        pn[1] = ((float)y - (float) viewport[1]) / ((float) viewport[3]) * 2.0f - 1.0f;
        // use z of projected joint position for correct depth
        pn[2] = pp[2];
        pn[3] = 1.0f;

        // unproject
        pn = inverse(mvp) * pn;
        pn /= pn[3];

        // set target position for effector and solve IK
        Vec3 target(pn[0], pn[1], pn[2]);
        animator_.mesh_.skeleton_.joints_[animator_.get_active_bone()]->target_ = target;

        ik_->solve_IK(20);

        if(D_SIMULATE)
        {
            animator_.solver_->update_anchors();
            update_ = true;
        }
    }
}


//=============================================================================
