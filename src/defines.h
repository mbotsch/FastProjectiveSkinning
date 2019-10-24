//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

// weights
#define D_TETSTRAIN		85.0    // weight corresponding to tetrahedron strain constraint
#define D_ANCHOR		70.0    // weight corresponding to anchor constraint (if used)
#define D_COLLISION		150.0   // weight corresponding to collision constraint (if used)

// some simulation parameters
#define D_DAMPING		1.0     // damping of velocities (1 - no damping; 0 - completely damped)
#define D_MASS			4000.0  // mass of the complete mesh, this will affect dynamics
#define D_TIMESTEP      0.2     // timestep, higher values will lead to increased damping
#define D_ITERATIONS 	10      // Projective Dynamics iterations, this value can be decreased, in case of smooth animations

// for debug purpose: use D_SHRINKSKIN false to not produce the volumetric mesh
// and D_SIMULATE false if you want to test the shrinking without any simulation
#define D_SHRINKSKIN            true
#define D_SIMULATE              (true & D_SHRINKSKIN)

#define D_GPU                   true    // turns the GPU simulation on/off


#define D_USE_SLIDING_JOINTS	true    // turns sliding joints on/off
#define D_USE_BC                true    // turns hard constraints on/off
#define D_USE_DIFF_MASSES       true    // if false, the mass is independent of volume, each vertex will get mass = 1

// Collisions
#define D_GLOBAL_COLLISIONS     true   // turns collision handling on/off



