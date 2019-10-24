//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#pragma once

#include <cuda_runtime.h>

#include "cuda_helper/helper_cuda.h"
#include "mesh/Helper.h"
#include "cuda_profiler_api.h"

#include "PD_solver.h"

namespace Projective_Skinning
{

class PDsolverCuda : public PD_Solver
{
public:

	struct CRSMatrix
	{
		float* values;
		int* cols;
		int* row_ptr;
		int nnz;
		int n_rows;
		bool device;
	};
	
	struct HYBMatrix
	{
        float* diagvalues;
        float* orig_diagvalues;
		float* ellvalues;
		float* crsvalues;
		int* ellcols;
		int* cols;
		int* row_ptr;
		int nnz;
		int n_rows;
		int n_ellrows;
		int n_csr;
		int n_maxNNZperBlock;
		bool device;
	};

    struct ELLMatrix
    {
        float* ellvalues;
        float* crsvalues;
        int* ellcols;
        int* cols;
        int* row_ptr;
        int nnz;
        int n_rows;
        int n_ellrows;
        int n_csr;
        bool device;
        int n_maxNNZperBlock;
    };
	
    PDsolverCuda():d_hr_points_(0), d_hr_normals_(0), d_upsampling_Nij_(0), d_upsampling_normal_Nij_(0), d_upsampling_neighbors_(0){PD_Solver();}
    virtual ~PDsolverCuda(){tidyUp();}
	
    virtual void init(Mesh *mesh, const float tetweight, const float ancweight, const float colweight, const float timestep, const float damping,
                const bool soft_bc) override;

    virtual void update_skin(int iterations) override;
    virtual void update_anchors() override;
    virtual void update_HR(float* d_vbo, float* d_nbo) override;
    virtual void update_normals(bool just_VN = true) override;
    virtual void update_ogl_sim_mesh_buffers(float* vbo, float* nbo) override;

    void reset();

    void tidyUp();
	
    float weight_col_;
    float* d_points;

    IndexVector indices_;
    std::vector<int> coltrigs_, collisions_;
    int num_handCols_;
    int num_surface_points_;
    int num_unified_points_;

    std::vector<int> transformIndices_;
    IndexVector additionalAnchors_;
    bool hardconstraints_;
    bool use_same_Nij_, duplicate_HR_points_;
    int numUS_, numUSNeighbors_, numDuplicatedPoints_;


protected:

    virtual void update_local() override;
    virtual void update_global() override;

    void init_cuda_data();
    void init_cuda_normals();
    void init_cuda_upsampling(const Projective_Skinning::IndexVector &us_indices, std::vector<std::vector<unsigned int> > &us_neighbors,
                                std::vector<std::vector<float> > &us_Nij , std::vector<std::vector<float> > &us_normal_Nij, bool duplicate);

    float *d_orig_points, *d_old_points, *d_velocities, *d_momentum, *d_Atppmom_, *d_masses, *d_projections, *d_tetweights, *d_edgeinv, *d_anchors, *d_face_normals, *d_vertex_normals, *d_bc;
    int *d_tets, *d_collisionIndices_;

    float* d_colcsrvals, *d_collisionprojections_;
    int * d_colcsrcols, *d_colcsrrptr;

    float* d_sjParams_, *d_transformationMatrices_;
    unsigned int * d_sjIndices_, *d_additionalAnchorIndices_;
    int* d_transformIndices_;

    float* d_orig_vertex_normals_;

    float* d_hr_points_, *d_hr_normals_, *d_upsampling_Nij_, *d_upsampling_normal_Nij_;
    unsigned int *d_upsampling_neighbors_;

    unsigned int *d_indices, *d_offsets, *d_neighbors;
    //CRSMatrix d_At_;
	ELLMatrix d_At2_;
    ELLMatrix d_Abc_;
    HYBMatrix d_S_;
    std::vector<float> orig_diag_values_, current_diag_values_;
	
	float *d_d_, *d_q_, *d_r_, *h_greek_, *d_greek_;
	
    cudaStream_t stream1_, stream2_;
    cudaEvent_t event1_, event2_;
	
	
    void convertEigenMatrixToCRSPointers(const Projective_Skinning::SparseMatrix &S, CRSMatrix& out_crs);
    void convertEigenMatrixToCRSDevicePointers(const Projective_Skinning::SparseMatrix &S, CRSMatrix& out_crs);
    void convertEigenMatrixToHYBPointers(const Projective_Skinning::SparseMatrix &S, HYBMatrix& out_hyb, int numellcols);
    void convertEigenMatrixToHYBDevicePointers(const Projective_Skinning::SparseMatrix &S, HYBMatrix& out_hyb, int numellcols);
    void convertEigenMatrixToELLDevicePointers(const Projective_Skinning::SparseMatrix &S, ELLMatrix& out_ell, int numellcols);
	
	void deleteCRSMatrix(CRSMatrix& crs);
	void deleteHYBMatrix(HYBMatrix& hyb);
    void deleteELLMatrix(ELLMatrix& hyb);
	
	
	void initMPCGArrays(float*& d_d, float*& d_q, float*& d_r, float*& h_greek, float*& d_greek, int size)
	{
		
		h_greek = new float[15];
		for(int i = 0; i < 15; i++)
			h_greek[i] = 0.0;
		
		checkCudaErrors(cudaMalloc((void**)&d_d, 	3*size*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_r, 	3*size*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_q, 	3*size*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_greek, 	3*5*sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_greek, h_greek,	(3*5)*sizeof(float), cudaMemcpyHostToDevice));
	}
	
};

class PDsolverCudaCollisions : public PDsolverCuda
{
public:

    PDsolverCudaCollisions():PDsolverCuda(){}

    virtual void reinit(const std::vector<int> &coltrigs, const int num_ti) override;

    virtual void update_anchors() override;
private:

    virtual void update_local() override;
    virtual void update_global() override;
};

}
