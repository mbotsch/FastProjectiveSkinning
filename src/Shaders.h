//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================
#pragma once
//=============================================================================


static const char* character_vshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
#endif
"layout (location=0) in vec4 vertex;\n"
"layout (location=1) in vec3 normal;\n"
"layout (location=2) in vec2 texcoords;\n"
"out vec3 v2f_normal;\n"
"out vec2 v2f_texcoords;\n"
"uniform mat4 modelview_projection_matrix;\n"
"uniform mat3 normal_matrix;\n"
"\n"
"void main()\n"
"{\n"
"    v2f_normal    = normalize(normal_matrix * normal);\n"
"    v2f_texcoords = texcoords;\n"
"    gl_Position   = modelview_projection_matrix * vertex;\n"
"}\n";


static const char* character_fshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
"precision mediump float;\n"
#endif
"in vec3  v2f_normal;\n"
"in vec2  v2f_texcoords;\n"
"\n"
"uniform bool      use_texture;\n"
"uniform bool      use_srgb;\n"
"uniform sampler2D mytexture;\n"
"uniform bool      use_black;\n"
"\n"
"vec3  light1 = vec3( 1.0, 1.0, 1.0);\n"
"vec3  light2 = vec3(-1.0, 1.0, 1.0);\n"
"\n"
"float ambient   = 0.1;\n"
"float diffuse   = 0.8;\n"
"float specular  = 0.01;\n"
"float shininess = 30.0;\n"
"\n"
"out vec4 f_color;\n"
"\n"
"void main()\n"
"{\n"
"   vec3 color = use_texture ? texture(mytexture, v2f_texcoords).xyz : vec3(0.75, 0.75, 0.75);\n"
"   color = use_black ? vec3(0.1,0.1,0.1) : color;\n"
"   \n"
"   vec3 L1 = normalize(light1);\n"
"   vec3 L2 = normalize(light2);\n"
"   vec3 N  = v2f_normal;\n"
"   vec3 V  = vec3(0.0, 0.0, 1.0);\n"
"   vec3 R;\n"
"   float NL, RV;\n"
"   \n"
"   vec3 rgb = ambient * 0.1 * color;\n"
"   \n"
"   NL = dot(N, L1);\n"
"   if (NL > 0.0)\n"
"   {\n"
"       rgb += diffuse * NL * color;\n"
"       R  = normalize(-reflect(L1, N));\n"
"       RV = dot(R, V);\n"
"       if (RV > 0.0) rgb += vec3( specular * pow(RV, shininess) );\n"
"   }\n"
"   \n"
"   NL = dot(N, L2);\n"
"   if (NL > 0.0)\n"
"   {\n"
"        rgb += diffuse * NL * color;\n"
"        R  = normalize(-reflect(L2, N));\n"
"        RV = dot(R, V);\n"
"        if (RV > 0.0) rgb += vec3( specular * pow(RV, shininess) );\n"
"   }\n"
"   \n"
"   if (use_srgb)  rgb = pow(clamp(rgb, 0.0, 1.0), vec3(0.45));\n"
"   f_color = vec4(rgb, 1.0);\n"
"}";


//=============================================================================


static const char* skeleton_vshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
"precision mediump float;\n"
#endif
"layout (location=0) in vec4 vertex;\n"
"layout (location=1) in vec4 color;\n"
"out vec4 v2f_color;\n"
"uniform mat4 modelview_projection_matrix;\n"
"void main()\n"
"{\n"
    "gl_PointSize = 10.0;\n"
    "gl_Position = modelview_projection_matrix * vertex;\n"
    "v2f_color   = color;\n"
"}\n";


static const char* skeleton_fshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
"precision mediump float;\n"
#endif
"in vec4 v2f_color;\n"
"out vec4 out_color;\n"
"void main()\n"
"{\n"
    "out_color = v2f_color;\n"
"}\n";


//=============================================================================


static const char* mesh_vshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
#endif
"layout (location=0) in vec4 vertex;\n"
"layout (location=1) in vec3 normal;\n"
"layout (location=2) in vec2 texcoords;\n"
"out vec3 v2f_normal;\n"
"out vec2 v2f_texcoords;\n"
"uniform mat4 modelview_projection_matrix;\n"
"uniform mat3 normal_matrix;\n"
"\n"
"void main()\n"
"{\n"
"    v2f_normal    = normalize(normal_matrix * normal);\n"
"    v2f_texcoords = texcoords;\n"
"    gl_Position   = modelview_projection_matrix * vertex;\n"
"}\n";

static const char* mesh_fshader =
#ifndef __EMSCRIPTEN__
"#version 330\n"
#else
"#version 300 es\n"
"precision mediump float;\n"
#endif
"in vec3  v2f_normal;\n"
"in vec2  v2f_texcoords;\n"
"\n"
"uniform bool      use_texture;\n"
"uniform bool      use_srgb;\n"
"uniform sampler2D mytexture;\n"
"uniform bool      use_black;\n"
"\n"
"vec3  light1 = vec3( 1.0, 1.0, 1.0);\n"
"vec3  light2 = vec3(-1.0, 1.0, 1.0);\n"
"\n"
"float ambient   = 0.1;\n"
"float diffuse   = 0.8;\n"
"float specular  = 0.01;\n"
"float shininess = 30.0;\n"
"\n"
"out vec4 f_color;\n"
"\n"
"void main()\n"
"{\n"
"   vec3 color = use_texture ? texture(mytexture, v2f_texcoords).xyz : vec3(1.0, 0.55, 0.55);\n"
"   color = use_black ? vec3(0.1,0.1,0.1) : color;\n"
"   \n"
"   vec3 L1 = normalize(light1);\n"
"   vec3 L2 = normalize(light2);\n"
"   vec3 N  = v2f_normal;\n"
"   vec3 V  = vec3(0.0, 0.0, 1.0);\n"
"   vec3 R;\n"
"   float NL, RV;\n"
"   \n"
"   vec3 rgb = ambient * 0.1 * color;\n"
"   \n"
"   NL = dot(N, L1);\n"
"   if (NL > 0.0)\n"
"   {\n"
"       rgb += diffuse * NL * color;\n"
"       R  = normalize(-reflect(L1, N));\n"
"       RV = dot(R, V);\n"
"       if (RV > 0.0) rgb += vec3( specular * pow(RV, shininess) );\n"
"   }\n"
"   \n"
"   NL = dot(N, L2);\n"
"   if (NL > 0.0)\n"
"   {\n"
"        rgb += diffuse * NL * color;\n"
"        R  = normalize(-reflect(L2, N));\n"
"        RV = dot(R, V);\n"
"        if (RV > 0.0) rgb += vec3( specular * pow(RV, shininess) );\n"
"   }\n"
"   \n"
"   if (use_srgb)  rgb = pow(clamp(rgb, 0.0, 1.0), vec3(0.45));\n"
"   f_color = vec4(rgb, 1.0);\n"
"}";


//=============================================================================
