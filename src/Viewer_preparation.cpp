//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Viewer_preparation.h"
#include "Shaders.h"
#include <imgui.h>
#include <functional>
#include <pmp/algorithms/TriangleKdTree.h>
#include <pmp/algorithms/SurfaceNormals.h>
#include <pmp/algorithms/SurfaceSimplification.h>
#include <pmp/algorithms/SurfaceGeodesic.h>
#include <pmp/Timer.h>

using namespace pmp;

//=============================================================================

Preparation_Viewer::Preparation_Viewer(const char* title, int width, int height)
    : MeshViewer(title,width, height),
      ready_for_skinning_(false),
      skeleton_vertex_array_(0),
      skeleton_vertex_buffer_(0),
      skeleton_color_buffer_(0),
      selected_joint_(-1),
      decimated_(false),
      quadratic_(2),
      decimation_percentage_(20),
      num_us_neighbors_(20)
{
    set_draw_mode("Smooth Shading");

    crease_angle_ = 0;

    if (!skeleton_shader_.source(skeleton_vshader, skeleton_fshader))
    {
        std::cerr << "Error: Could not load skeleton shaders.\n";
        exit(EXIT_FAILURE);
    }

    if (!mesh_shader_.source(mesh_vshader, mesh_fshader))
    {
        std::cerr << "Error: Could not load skeleton shaders.\n";
        exit(EXIT_FAILURE);
    }

    // suppress unused variable warnings
    (void)character_vshader;
    (void)character_fshader;


    add_help_item("Ctrl + Left mouse", "Select/drag joint");
    add_help_item("J", "Add new joint to selected");
}

Preparation_Viewer::~Preparation_Viewer()
{
    if (skeleton_vertex_buffer_)    glDeleteBuffers(1, &skeleton_vertex_buffer_);
    if (skeleton_color_buffer_)     glDeleteBuffers(1, &skeleton_color_buffer_);
    if (skeleton_vertex_array_)     glDeleteVertexArrays(1, &skeleton_vertex_array_);


    if (ogl_skeleton_mesh_.vertex_buffer)             glDeleteBuffers(1, &ogl_skeleton_mesh_.vertex_buffer);
    if (ogl_skeleton_mesh_.normal_buffer)             glDeleteBuffers(1, &ogl_skeleton_mesh_.normal_buffer);
    if (ogl_skeleton_mesh_.vertex_array)       glDeleteVertexArrays(1, &ogl_skeleton_mesh_.vertex_array);
}

void Preparation_Viewer::load_mesh(const char *filename)
{
    MeshViewer::load_mesh(filename);
    if(mesh_.n_vertices() == 0)
    {
    	std::cerr << "Mesh cannot be read from " << filename << std::endl;
    	return;
    }

    mesh_HR_ = mesh_;

    // init skin vertices
    skin_vertices_.resize(3,mesh_.n_vertices());
    for(auto v : mesh_.vertices())
    {
        Point p = mesh_.position(v);
        skin_vertices_.col(v.idx()) = Projective_Skinning::Vec3(p[0],p[1],p[2]);
    }


    // init skeleton
    if(skel_filename_.empty())
    {
        BoundingBox bb = mesh_.bounds();
        Eigen::Vector3f center(bb.center().data());
        float r = 0.5*bb.size();

        joint_positions_.push_back(center);
        joint_positions_.push_back(center + 0.1*r*Eigen::Vector3f(0,1,0));

        joint_indices_.push_back(0);
        joint_indices_.push_back(1);

        Projective_Skinning::Mat3X pos(3,joint_positions_.size());
        for(size_t i = 0; i < joint_positions_.size(); i++)
        	pos.col(i) = joint_positions_[i];

        volumetric_skeleton_.init(pos,joint_indices_,skin_vertices_);
    }
    else
    {
        volumetric_skeleton_.init(skel_filename_.c_str(),skin_vertices_);
        joint_indices_ = volumetric_skeleton_.bone_indices_;
        joint_positions_.resize(volumetric_skeleton_.joint_positions_.cols());
        for(size_t i = 0; i < joint_positions_.size(); i++)
        	joint_positions_[i] = volumetric_skeleton_.joint_positions_.col(i);
    }

    init_ogl_buffers(ogl_skeleton_mesh_,volumetric_skeleton_.vertices_, volumetric_skeleton_.normals_, volumetric_skeleton_.indices_);

}

void Preparation_Viewer::build_from_ini(const char *ini_filename)
{
    ini_filename_ = ini_filename;
    std::ifstream ifs(ini_filename);
    if(!ifs)
    {
        std::cerr << "Could not read file: " << ini_filename << std::endl;
        exit(-3);
    }
    std::string line;
    while(std::getline(ifs,line))
    {
        std::stringstream ss_line(line);
        std::string header,info;

        ss_line >> header >> info;
        std::cout << line << /*"\n" << header << "\n" << info <<*/ std::endl;

        // get rid of " to support spaces in filenames
        if(info[0] == '"')
        {
            info = info.substr(1,info.find_last_of('"')-1);
        }

        if(header == "SIMMESH")
        {
            mesh_LR_filename_ = info;
        }
        else if(header == "SKELETON")
        {
            skel_filename_ = info;
        }
        else if(header == "VISMESH")
        {
            filename_ = info;
        }
        else if(header == "UPSAMPLING")
        {
            us_filename_ = info;
        }
    }

    if(mesh_LR_filename_.empty())
    {
        std::cerr << "no mesh defined in ini!" << std::endl; return;
    }
    else
    {
        if(filename_.empty())
        {
            load_mesh(mesh_LR_filename_.c_str());
        }
        else
        {
            decimated_ = true;
            std::string temp = filename_;
            MeshViewer::load_mesh(mesh_LR_filename_.c_str());
            mesh_LR_ = mesh_;
            filename_ = temp;

            load_mesh(filename_.c_str());
            find_low_to_high();


        }
    }
}


void Preparation_Viewer::keyboard(int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS) // only react on key press events
        return;

    using namespace Projective_Skinning;

    std::vector<Joint*> &joints = volumetric_skeleton_.joints_;

    switch (key)
    {
    // add a joint
    case GLFW_KEY_J:
    {
        if(selected_joint_ < 0) break;

        Joint* jo = joints[selected_joint_];
        Vec3 pos0 = jo->position_;
        Vec3 pos1 = (!jo->is_root_) ? joints[selected_joint_]->parent_->position_ : jo->childreen_[0]->position_;

        joint_positions_.push_back(pos0 + 0.5*(pos0 - pos1));
        joint_indices_.push_back(jo->index_);
        joint_indices_.push_back(joint_positions_.size() - 1);

        Projective_Skinning::Mat3X pos(3,joint_positions_.size());
        std::memcpy(pos.data(), joint_positions_.data(), joint_positions_.size()*sizeof(Eigen::Vector3f)); // todo move this into init and just use float pointer as argument
        volumetric_skeleton_.init(pos,joint_indices_,skin_vertices_);

        init_ogl_buffers(ogl_skeleton_mesh_,volumetric_skeleton_.vertices_, volumetric_skeleton_.normals_, volumetric_skeleton_.indices_);

        break;
    }

    default:
    {
        MeshViewer::keyboard(key, scancode, action, mods);
        break;
    }

    }
}

void Preparation_Viewer::draw(const std::string &draw_mode)
{
    MeshViewer::draw(draw_mode);

    // draw skeleton
    mat4 modelviewprojectionMatrix = projection_matrix_ * modelview_matrix_;
    mat3 normalMatrix = inverse(transpose(linear_part(modelview_matrix_)));
    //mat4 modelviewprojectionMatrix = projection_matrix_ * modelview_matrix_;

    glClear(GL_DEPTH_BUFFER_BIT);

    // draw vol skel mesh
    glFrontFace(GL_CCW);
    glEnable( GL_DEPTH_TEST );
    glBindVertexArray(ogl_skeleton_mesh_.vertex_array);

    mesh_shader_.use();
    mesh_shader_.set_uniform("modelview_projection_matrix", modelviewprojectionMatrix);
    mesh_shader_.set_uniform("normal_matrix", normalMatrix);

    int n_vertices = (GLsizei)ogl_skeleton_mesh_.indices->size();

    mesh_shader_.set_uniform("use_texture", false);
    mesh_shader_.set_uniform("use_srgb",    false);
    mesh_shader_.set_uniform("use_black", false);

    glDrawElements(GL_TRIANGLES, n_vertices, GL_UNSIGNED_INT, NULL);


    int n_joints = joint_positions_.size();
    std::vector<vec3> colors;
    colors.resize(n_joints,vec3(0.8f));

    skeleton_shader_.use();
    skeleton_shader_.set_uniform("modelview_projection_matrix", modelviewprojectionMatrix);
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
    glBufferData(GL_ARRAY_BUFFER, n_joints*3*sizeof(float), joint_positions_.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, skeleton_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // draw as lines
    glDrawElements(GL_LINES, joint_indices_.size(), GL_UNSIGNED_INT, joint_indices_.data());

    // prepare point drawings
    colors.clear();
    colors.resize(n_joints,vec3(0,0,1));

    if(selected_joint_ >= 0)
        colors[selected_joint_] = vec3(0,1,0);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, skeleton_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(vec3), &colors[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // draw points
    glDrawArrays(GL_POINTS, 0, n_joints);

    glEnable( GL_DEPTH_TEST );
}

void Preparation_Viewer::mouse(int _button, int _action, int _mods)
{
    // CTRL not pressed -> rotate model
    if (!ctrl_pressed())
    {
        MeshViewer::mouse(_button, _action, _mods);
        return;
    }


    // CTRL pressed -> drag joints around
    if (_action == GLFW_PRESS)
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
            Eigen::Matrix4f mvp_ps(mvp.data());
            selected_joint_ = pick_joint(Eigen::Vector2f(p2d[0], p2d[1]), mvp_ps);
        }
    }
}

void Preparation_Viewer::motion(double _x, double _y)
{
    // CTRL not pressed -> rotate model
    if (!ctrl_pressed())
    {
        MeshViewer::motion(_x,_y);
        return;
    }


    if (left_mouse_pressed() && selected_joint_ != -1)
    {
        Eigen::Vector3f p = joint_positions_[selected_joint_];

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

        joint_positions_[selected_joint_] = Eigen::Vector3f(pn.data());

        Projective_Skinning::Mat3X pos(3,joint_positions_.size());
        std::memcpy(pos.data(), joint_positions_.data(), joint_positions_.size()*sizeof(Eigen::Vector3f)); // todo move this into init and just use float pointer as argument
        volumetric_skeleton_.init(pos,joint_indices_,skin_vertices_, selected_joint_);

        init_ogl_buffers(ogl_skeleton_mesh_,volumetric_skeleton_.vertices_, volumetric_skeleton_.normals_, volumetric_skeleton_.indices_);
    }
}

PSScalar fweight1(PSScalar d)
{
    return pow(1.0 - d, 3.0);
}

void Preparation_Viewer::process_imgui()
{
    MeshViewer::process_imgui();

    using namespace Projective_Skinning;

    if (ImGui::CollapsingHeader("Skeleton Builder", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        ImGui::Text("Joint Manipulation");
        if(ImGui::Button("Add Joint"))
        {
            keyboard(GLFW_KEY_J,0 , GLFW_PRESS, 0);
        }

        ImGui::SameLine();

        if(ImGui::Button("Delete"))
        {
            if(selected_joint_ >= 2)
            {
                IndexVector indices;
                std::vector<Vec3> positions;

                // adds joints and indices of childreen joints except selected one
                std::function<void(Joint* jo)> add_joints = [&](Joint* jo)
                {
                    unsigned int joi = positions.size() - 1;
                    for(auto j : jo->childreen_)
                    {
                        if(j->index_ == selected_joint_) continue;

                        indices.push_back(joi);
                        indices.push_back(positions.size());
                        positions.push_back(j->position_);

                        // recursive call
                        add_joints(j);
                    }
                };

                positions.push_back(volumetric_skeleton_.joints_[0]->position_);

                // start recursion
                add_joints(volumetric_skeleton_.joints_[0]);

                joint_positions_ = positions;
                joint_indices_ = indices;

                // setup new skeleton
                Projective_Skinning::Mat3X pos(3,joint_positions_.size());
                std::memcpy(pos.data(), joint_positions_.data(), joint_positions_.size()*sizeof(Eigen::Vector3f)); // todo move this into init and just use float pointer as argument
                volumetric_skeleton_.init(pos,joint_indices_,skin_vertices_);

                init_ogl_buffers(ogl_skeleton_mesh_,volumetric_skeleton_.vertices_, volumetric_skeleton_.normals_, volumetric_skeleton_.indices_);

                selected_joint_ -= 1;
            }
            else
            {
                std::cerr << "Cannot delete root or first child" << std::endl;
            }
        }

        if(ImGui::Button("Store Skeleton"))
        {
            // first check if all joints are inside of the mesh
            float error = false;

            TriangleKdTree kdTree(mesh_);

            for(auto jo : volumetric_skeleton_.joints_)
            {
                Point p(jo->position_(0),jo->position_(1),jo->position_(2));
                TriangleKdTree::NearestNeighbor nn = kdTree.nearest(p);

                Normal n = SurfaceNormals::compute_face_normal(mesh_,nn.face);
                float signed_dist = dot(n,nn.nearest - p);

                if(signed_dist < 0)
                {
                    error = true;
                    std::cerr << "Joint " << jo->index_ << " is outside of your mesh!" << std::endl;
                }
            }

            if(!error)
            {
                // store skeleton with same name as meshfile
                skel_filename_ = filename_.substr(0,filename_.find_last_of(".")) + std::string(".skel");

                // if skeleton filename already exists, add number
                std::ifstream ifs; int skel_ctr = 0;
                ifs.open(skel_filename_.c_str());
                while(ifs)
                {
                    skel_ctr++;
                    skel_filename_ = filename_.substr(0,filename_.find_last_of(".")) + std::to_string(skel_ctr) + std::string(".skel");
                    ifs.close(); ifs.open(skel_filename_.c_str());
                }

                volumetric_skeleton_.store_skeleton(skel_filename_.c_str());
            }
        }
    }

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Decimation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();

        // 1000 vertices can be simulated without upsampling
        usint minsize = std::max((size_t)500, mesh_HR_.n_vertices()/100);
        if(minsize < mesh_HR_.n_vertices()/2)
        {
            ImGui::PushItemWidth(80);
            static float percentage = 0.2;
            ImGui::SliderFloat("LR/HR", &percentage, (float)minsize/mesh_HR_.n_vertices(), 0.5f, "%.2f");
            ImGui::PopItemWidth();

            if(ImGui::Button("Decimate HR Mesh"))
            {
                mesh_ = mesh_HR_;

                std::cout << "Mesh is decimated..." << std::endl;
                PSScalar aspect_ratio = 5;
                PSScalar normal_deviation = 135.0;
                SurfaceSimplification ss(mesh_);
                ss.initialize(aspect_ratio, 0.0, 0.0, normal_deviation, 0.0);
                ss.simplify(mesh_.n_vertices() * percentage);
                update_mesh();
                mesh_LR_ = mesh_;

                // avoid error when calling simplification again
                auto q = mesh_.get_vertex_property<Quadric>("v:quadric");
                mesh_.remove_vertex_property(q);

                find_low_to_high();

                decimation_percentage_ = (int)(100*percentage);
                // store skeleton with same name as HR meshfile + _LR + Percentage
                mesh_LR_filename_ = filename_.substr(0,filename_.find_last_of(".")) +
                        std::string("_LR") + std::to_string(decimation_percentage_)
                        + std::string(".off");
                mesh_.write(mesh_LR_filename_.c_str());

                decimated_ = true;
                std::cout << "done" << std::endl;
            }

        }
        else
        {
            ImGui::Text("No decimation needed here.");
        }

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Upsampling", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Spacing();
        if(decimated_)
        {

            ImGui::Text("Mesh Draw Options");
            static int draw_lr = 1;
            if(mesh_.n_vertices() == mesh_HR_.n_vertices())
                draw_lr = 0;


            int last = draw_lr;
            ImGui::RadioButton("highres", &draw_lr, 0);
            ImGui::RadioButton("decimated", &draw_lr, 1);
            if(last != draw_lr)
            {
                mesh_ = draw_lr == 1 ? mesh_LR_ : mesh_HR_;
                update_mesh();
            }

            ImGui::Spacing();


            ImGui::Text("Upsampling Mode");
            ImGui::RadioButton("linear", &quadratic_, 1);
            ImGui::RadioButton("quadratic", &quadratic_, 2);

            ImGui::PushItemWidth(80);
            ImGui::SliderInt("# Neighbors", &num_us_neighbors_, 20, 40);
            num_us_neighbors_+= num_us_neighbors_%2;
            ImGui::PopItemWidth();

            static bool failed = false;
            if(ImGui::Button("Compute US Weights"))
            {
                // if we have enough vertices in mesh
                if((int)mesh_LR_.n_vertices() < num_us_neighbors_*2)
                {
                    std::cerr << "Problem: LR Mesh has just " << mesh_LR_.n_vertices() << " vertices. Consider using fast projective skinning without upsampling!" << std::endl;
                    failed = true;
                }
                else
                {
                    float dist_factor = 1.0f;
                    while(!computeGeodesics(num_us_neighbors_,dist_factor) && dist_factor < 10.0f) dist_factor*= 1.5f;

                    if(dist_factor < 10.0f)
                    {
                        // compute weights (Nij) for upsapling
                        compute_upsampling_weights(num_us_neighbors_,quadratic_,fweight1);

                        // store them in file
                        std::string s_quadratic = (quadratic_ == 2) ? "_q" : "_l";
                        us_filename_ = filename_.substr(0,filename_.find_last_of("."))
                                + std::string("_")
                                + std::to_string(num_us_neighbors_)
                                + s_quadratic
                                + std::string(".txt");
                        print_upsampling_weights(us_filename_.c_str());
                        failed = false;
                    }
                    else
                    {
                        failed = true;
                    }
                }


            }

            if(failed)
            {
                ImGui::SameLine();
                ImGui::Text("Failed!!");
            }


        }
        else
        {
            ImGui::Text("Decimate first");
        }

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Initialization File", ImGuiTreeNodeFlags_DefaultOpen))
    {

        ImGui::Spacing();

        if(ImGui::Button("Create .ini File"))
        {
            if(skel_filename_.empty())
            {
                std::cerr << "You should first create and store a skeleton!" << std::endl;
            }
            else
            {
                ini_filename_ = std::string(filename_.substr(0,filename_.find_last_of(".")));
                if(!mesh_LR_filename_.empty() && !us_filename_.empty())
                {
                    std::string s_quadratic = (quadratic_ == 2) ? "_q" : "_l";

                    ini_filename_ = ini_filename_
                            + std::string("_") + std::to_string(decimation_percentage_)
                            + std::string("_") + std::to_string(num_us_neighbors_)
                            + s_quadratic;
                }
                ini_filename_ = ini_filename_ + std::string(".ini");
                std::ofstream ofs(ini_filename_);
                ofs << "SIMMESH ";
                if(!mesh_LR_filename_.empty() && !us_filename_.empty())
                {
                    ofs << mesh_LR_filename_ << "\n";
                    ofs << "VISMESH " << filename_ << "\n";
                    ofs << "UPSAMPLING " << us_filename_ << "\n";
                }
                else
                {
                    ofs << filename_ << "\n";
                }

                ofs << "SKELETON " << skel_filename_ << "\n";
                ofs.close();

                std::cout << "Created .ini file: " << ini_filename_ << std::endl;

            }
        }

        if(!ini_filename_.empty())
        {
            if(ImGui::Button("Start Skinning"))
            {
                ready_for_skinning_ = true;
                glfwSetWindowShouldClose(glfwGetCurrentContext(),1);
            }
        }
    }
}

int Preparation_Viewer::pick_joint(const Eigen::Vector2f &coord2d, const Eigen::Matrix4f &mvp)
{
    float closest_dist=FLT_MAX;

    int joint = 0;
    for (size_t i=0; i < joint_positions_.size(); ++i)
    {
        const Eigen::Vector4f p(joint_positions_[i](0) , joint_positions_[i](1) , joint_positions_[i](2) , 1.0);
        Eigen::Vector4f ndc = mvp * p;
        ndc /= ndc(3);

        const float d = (Eigen::Vector2f(ndc(0),ndc(1)) - coord2d).norm();
        if (d < closest_dist)
        {
            closest_dist = d;
            joint = (int) i;
        }
    }

    return joint;
}

void Preparation_Viewer::init_ogl_buffers(Preparation_Viewer::Mesh_OGL_Data &ogl_data, Projective_Skinning::Mat3X &vertices, Projective_Skinning::Mat3X &normals, Projective_Skinning::IndexVector &indices)
{
    if(!ogl_data.vertex_array)
    {
        ogl_data.vertices = &vertices;
        ogl_data.normals = &normals;
        ogl_data.indices = &indices;

        glGenVertexArrays(1, &ogl_data.vertex_array);
        glBindVertexArray(ogl_data.vertex_array);
        glGenBuffers(1,&ogl_data.vertex_buffer);
        glGenBuffers(1,&ogl_data.normal_buffer);
        glGenBuffers(1,&ogl_data.index_buffer);
    }

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

}

void Preparation_Viewer::find_low_to_high()
{
    high_to_low_.clear(); low_to_high_.clear(); vertices_in_both_hr_lr_.clear();
    high_to_low_.resize(mesh_HR_.n_vertices(), -1);
    low_to_high_.resize(mesh_LR_.n_vertices(), -1);

    for(auto v : mesh_LR_.vertices())
    {
        PSScalar mindist = FLT_MAX;
        int mini = -1;
        for(auto vv : mesh_HR_.vertices())
        {
            PSScalar dist = distance(mesh_LR_.position(v),mesh_HR_.position(vv));
            if(dist < mindist)
            {
                mindist = dist;
                mini = vv.idx();
            }
        }
        if(mini >= 0)
        {
            if(mindist > 1e-4)
            {
                std::cerr << "Problem: Minimal distance is too high!" << std::endl;
            }

            low_to_high_[v.idx()] = mini;
            high_to_low_[mini] = v.idx();
            if(vertices_in_both_hr_lr_.find(mini) == vertices_in_both_hr_lr_.end())
                vertices_in_both_hr_lr_.insert(mini);
            else
                std::cerr << "Problem: two decimated vertices have the same corresponding fine one!" << std::endl;
        }
    }
}

bool Preparation_Viewer::computeGeodesics(int numNeighbors, float dist_factor)
{
    int N_10 = mesh_LR_.n_vertices()/10;
    unsigned int N = numNeighbors + 1;
    std::cout << "Compute Geodesics..." << std::endl;

    LR_neighbors_in_HR_.resize(mesh_HR_.n_vertices());
    LR_neighbors_.resize(mesh_HR_.n_vertices());
    neighbor_weights_.resize(mesh_HR_.n_vertices());

    std::vector<std::vector<std::pair<int, float>>> idxDists;
    idxDists.resize(mesh_HR_.n_vertices());

    pmp::Timer t;
    t.start();

    // find typical distance in mesh
    PSScalar avg_edge_length = 0;
    for(auto e : mesh_LR_.edges())
        avg_edge_length += distance(mesh_LR_.position(mesh_LR_.vertex(e,0)),mesh_LR_.position(mesh_LR_.vertex(e,1)));
    avg_edge_length/=(PSScalar)mesh_LR_.n_edges();

    PSScalar maxdist = sqrt((PSScalar)(numNeighbors + 1))*avg_edge_length*dist_factor;

    SurfaceGeodesic geodist(mesh_HR_);
    for(unsigned int vi = 0; vi < mesh_LR_.n_vertices(); vi++)
    {
        Vertex v(low_to_high_[vi]);
        // setup seed
        std::vector<Vertex> seed;
        seed.push_back(v);

        // compute geodesic distance
        geodist.compute(seed, maxdist);

        // store (index, distance) in vector
        for(auto vv: mesh_HR_.vertices())
        {
            float dist = geodist(vv);
            if(dist < 2*maxdist)
                idxDists[vv.idx()].push_back(std::pair<int, float>(v.idx(), dist));
        }

        if(vi%N_10 == 0)
        {
            std::cout << (int)(100.0*vi/(float)(mesh_LR_.n_vertices() - 1) + 0.5f) << "%" << std::endl;
        }
    }

    std::cout << "Done " << t.stop().elapsed()/1000.0 << "s" << std::endl;

    std::cout << "Sorting..." << std::endl;
    t.start();

    for(auto v : mesh_HR_.vertices())
    {
        // sort this vector by distance
        std::sort(idxDists[v.idx()].begin(), idxDists[v.idx()].end(),
                  [](std::pair<int, float> &a, std::pair<int, float> &b) { return a.second < b.second; });

        // store the first (nearest) N vertices and their distance
        std::vector<int> &nb = LR_neighbors_in_HR_[v.idx()];
        std::vector<int> &nb_LR = LR_neighbors_[v.idx()];
        std::vector<float> &dists = neighbor_weights_[v.idx()];
        for(auto p : idxDists[v.idx()])
        {
            if(vertices_in_both_hr_lr_.find(p.first) != vertices_in_both_hr_lr_.end())
            {
                if(high_to_low_[p.first] < 0)
                {
                    std::cerr << "Problem with neighboring vertices!" << std::endl;
                    return false;
                }

                nb.push_back(p.first);
                nb_LR.push_back(high_to_low_[p.first]);
                dists.push_back(p.second);
            }
            if(nb.size() >= N)
                break;
        }
        if(nb.size() < N)
        {
            std::cerr << "Problem, not enough neighbors found! Retrying..." << std::endl;
            return false;
        }
    }

    std::cout << "Done " << t.stop().elapsed()/1000.0 << "s" << std::endl;

    return true;
}

void Preparation_Viewer::compute_upsampling_weights(int num, int order, PSScalar (*wfunc)(PSScalar))
{
    std::cout << "compute Nij..." << std::endl;

    typedef Eigen::Matrix<PSScalar,Eigen::Dynamic,Eigen::Dynamic> MatXs;
    typedef Eigen::Matrix<PSScalar,Eigen::Dynamic,1> VecXs;

    neighborNi_.clear();
    neighborNi_.resize(mesh_HR_.n_vertices());

    neighborNNi_.clear();
    neighborNNi_.resize(mesh_HR_.n_vertices());

    // compute normals
    SurfaceNormals::compute_vertex_normals(mesh_HR_);
    auto normals = mesh_HR_.vertex_property<Point>("v:normal");

    SurfaceNormals::compute_vertex_normals(mesh_LR_);
    auto normalsLR = mesh_LR_.vertex_property<Point>("v:normal");

    for(auto v : mesh_HR_.vertices())
    {
        neighborNi_[v.idx()].resize(num);
        neighborNNi_[v.idx()].resize(num);

        int dimG = 1;
        switch (order)
        {
        case 0: dimG = 1; break;
        case 1: dimG = 4; break;
        case 2: dimG = 10; break;
        default: order = 0; std::cout << "Error: Wrong order input. Fallback to order 0" << std::endl; break;
        }
        MatXs G;
        G.resize(dimG, dimG);
        G.setZero();

        // compute G
        PSScalar radius = neighbor_weights_[v.idx()][num];
        for(int i = 0; i < num; i++)
        {
            PSScalar dist = neighbor_weights_[v.idx()][i]/radius;
            PSScalar weight = wfunc(dist);
            Vertex vv(LR_neighbors_in_HR_[v.idx()][i]);

            PSScalar x = mesh_HR_.position(vv)[0];
            PSScalar y = mesh_HR_.position(vv)[1];
            PSScalar z = mesh_HR_.position(vv)[2];

            VecXs pi(dimG);

            switch(order)
            {
            case 0: pi << 1; break;
            case 1: pi << 1,x,y,z; break;
            case 2: pi << 1,x,y,z,x*x,y*y,z*z,x*y,x*z,y*z; break;
            }

            G +=  weight*pi*pi.transpose();
        }


        Eigen::CompleteOrthogonalDecomposition<MatXs> SVD;
        VecXs G_inv_pj;

        PSScalar x = mesh_HR_.position(v)[0];
        PSScalar y = mesh_HR_.position(v)[1];
        PSScalar z = mesh_HR_.position(v)[2];

        VecXs pj(dimG);

        switch(order)
        {
        case 0: pj << 1; break;
        case 1: pj << 1,x,y,z; break;
        case 2: pj << 1,x,y,z,x*x,y*y,z*z,x*y,x*z,y*z; break;
        }

        SVD.compute(G);
        G_inv_pj = SVD.solve(pj);

        for(int i = 0; i < num; i++)
        {
            PSScalar dist = neighbor_weights_[v.idx()][i]/radius;
            PSScalar weight = wfunc(dist);
            Vertex vv(LR_neighbors_in_HR_[v.idx()][i]);

            x = mesh_HR_.position(vv)[0];
            y = mesh_HR_.position(vv)[1];
            z = mesh_HR_.position(vv)[2];

            VecXs pi(dimG);

            switch(order)
            {
            case 0: pi << 1; break;
            case 1: pi << 1,x,y,z; break;
            case 2: pi << 1,x,y,z,x*x,y*y,z*z,x*y,x*z,y*z; break;
            }

            PSScalar Nij = weight*pi.dot(G_inv_pj);

            neighborNi_[v.idx()][i] = Nij;
        }


        // do the same for normals but with linear precision
        G.resize(4, 4);
        G.setZero();

        // compute G
        for(int i = 0; i < num; i++)
        {
            PSScalar dist = neighbor_weights_[v.idx()][i]/radius;
            PSScalar weight = wfunc(dist);
            Vertex vv(high_to_low_[LR_neighbors_in_HR_[v.idx()][i]]);

            PSScalar x = normalsLR[vv][0];
            PSScalar y = normalsLR[vv][1];
            PSScalar z = normalsLR[vv][2];

            VecXs pi(4);
            pi << 1,x,y,z;

            G +=  weight*pi*pi.transpose();
        }

        x = normals[v][0];
        y = normals[v][1];
        z = normals[v][2];

        pj.resize(4);
        pj << 1,x,y,z;

        SVD.compute(G);
        G_inv_pj = SVD.solve(pj);

        for(int i = 0; i < num; i++)
        {
            PSScalar dist = neighbor_weights_[v.idx()][i]/radius;
            PSScalar weight = wfunc(dist);
            Vertex vv(high_to_low_[LR_neighbors_in_HR_[v.idx()][i]]);

            PSScalar x = normalsLR[vv][0];
            PSScalar y = normalsLR[vv][1];
            PSScalar z = normalsLR[vv][2];

            VecXs pi(4);
            pi << 1,x,y,z;

            PSScalar Nij = weight*pi.dot(G_inv_pj);

            neighborNNi_[v.idx()][i] = Nij;
        }
    }
}

void Preparation_Viewer::print_upsampling_weights(const char *filename)
{
    std::cout << "Printing to " << filename << std::endl;
    std::ofstream ofs(filename);
    ofs << LR_neighbors_.size() << " " << neighborNi_[0].size() << "\n";

    for(unsigned int i = 0; i < LR_neighbors_.size();i++)
    {
        ofs << i << " " << neighborNi_[i].size() << " ";
        for(unsigned int j = 0; j < neighborNi_[i].size(); j++)
        {
            ofs << LR_neighbors_[i][j] << " " << neighborNi_[i][j] << " " << neighborNNi_[i][j] << " ";
        }
        ofs << "\n";
    }
    ofs.close();
    std::cout << "done" << std::endl;
}

//=============================================================================
