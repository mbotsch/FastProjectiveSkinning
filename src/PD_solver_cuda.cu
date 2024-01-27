//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "PD_solver_cuda.h"
#include <stdio.h>
#include "svd/svd3_cuda.h"

#define THREADS_PER_BLOCK 64
#define THREADS_PER_BLOCK_MOMENTUM 96
#define D_S_ELLROWS 4
#define D_AT_ELLROWS 28
#define D_ABC_ELLROWS 1

//global kernels
//---------------

__global__ void k_init_PCG(cudaTextureObject_t tex_points, const float* diavalues,const float* ellvalues, const int* ellcols,const float* crsvalues, const int* crscols, const int* row_ptr, float* d, float *r, float* greek, float* rhs, const int n_rows)
{
    __shared__ float temp[3][THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
	
    if(id < 15)
        greek[id] = 0.0;
	
    float valr1, valr2, valr3, vald1, vald2, vald3;
    // rhs - A*result
    if(id < n_rows)
    {
        //init with diag
        float diag = diavalues[id];
        valr1 = diag*tex1Dfetch<float>(tex_points, 3*id);
        valr2 = diag*tex1Dfetch<float>(tex_points, 3*id + 1);
        valr3 = diag*tex1Dfetch<float>(tex_points, 3*id + 2);
		
        //ell part
        for(int i = 0; i < D_S_ELLROWS; i++)
        {
            float ellval = ellvalues[i*n_rows + id];
			
            if(fabs(ellval) > 0.0)
            {
                int ellcol = 3*ellcols[i*n_rows + id];
                valr1 += ellval*tex1Dfetch<float>(tex_points, ellcol);
                valr2 += ellval*tex1Dfetch<float>(tex_points, ellcol + 1);
                valr3 += ellval*tex1Dfetch<float>(tex_points, ellcol + 2);
            }
        }
		
        //crs part
        const int start = row_ptr[id];
        const int end = row_ptr[id + 1];
        for(int i = start; i < end; i++)
        {
            float csr = crsvalues[i];
            int crscol = 3*crscols[i];
            valr1 += csr*tex1Dfetch<float>(tex_points, crscol);
            valr2 += csr*tex1Dfetch<float>(tex_points, crscol + 1);
            valr3 += csr*tex1Dfetch<float>(tex_points, crscol + 2);
        }
		
        valr1 = rhs[id] - valr1;
        valr2 = rhs[id + n_rows] - valr2;
        valr3 = rhs[id + 2*n_rows] - valr3;
		
        vald1 = valr1/diag;
        vald2 = valr2/diag;
        vald3 = valr3/diag;
		
        // store
        r[id] = valr1;
        r[id + n_rows] = valr2;
        r[id + 2*n_rows] = valr3;
		
        d[id] = vald1;
        d[id + n_rows] = vald2;
        d[id + 2*n_rows] = vald3;
    }
	
    // dot(r,r)
	
    if(id < n_rows)
    {
        temp[0][tid] = valr1*vald1;
        temp[1][tid] = valr2*vald2;
        temp[2][tid] = valr3*vald3;
    }
    else
    {
        //to be save even if it works without
        temp[0][tid] = 0;
        temp[1][tid] = 0;
        temp[2][tid] = 0;
    }
	
    __syncthreads();
	
    if(tid < 3)
    {
        float sum = 0;
		
        #pragma unroll
        for(int i = 0; i <  THREADS_PER_BLOCK; i++)
        {
            sum+= temp[tid][i];
        }
		
        atomicAdd(&(greek[5*tid]), sum);
    }
}


__global__ void k_init_PCG_collisions(cudaTextureObject_t tex_points, const float* diavalues,const float* ellvalues, const int* ellcols,
                                                              const float* crsvalues, const int* crscols, const int* row_ptr,
                                                              const float* collisioncrsvalues, const int* collisioncrscols, const int* collisionrow_ptr,
                                                              float* d, float *r, float* greek, float* rhs, const int n_rows)
{
    __shared__ float temp[3][THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if(id < 15)
        greek[id] = 0.0;

    float valr1, valr2, valr3, vald1, vald2, vald3;
    // rhs - A*result
    if(id < n_rows)
    {
        //init with diag
        float diag = diavalues[id];
        valr1 = diag*tex1Dfetch<float>(tex_points, 3*id);
        valr2 = diag*tex1Dfetch<float>(tex_points, 3*id + 1);
        valr3 = diag*tex1Dfetch<float>(tex_points, 3*id + 2);

        //ell part
        for(int i = 0; i < D_S_ELLROWS; i++)
        {
            float ellval = ellvalues[i*n_rows + id];

            if(fabs(ellval) > 0.0)
            {
                int ellcol = 3*ellcols[i*n_rows + id];
                valr1 += ellval*tex1Dfetch<float>(tex_points, ellcol);
                valr2 += ellval*tex1Dfetch<float>(tex_points, ellcol + 1);
                valr3 += ellval*tex1Dfetch<float>(tex_points, ellcol + 2);
            }
        }

        //crs part
        int start = row_ptr[id];
        int end = row_ptr[id + 1];
        for(int i = start; i < end; i++)
        {
            float csr = crsvalues[i];
            int crscol = 3*crscols[i];
            valr1 += csr*tex1Dfetch<float>(tex_points, crscol);
            valr2 += csr*tex1Dfetch<float>(tex_points, crscol + 1);
            valr3 += csr*tex1Dfetch<float>(tex_points, crscol + 2);
        }

        start = collisionrow_ptr[id];
        end = collisionrow_ptr[id + 1];
        for(int i = start; i < end; i++)
        {
            float csr = collisioncrsvalues[i];
            int crscol = 3*collisioncrscols[i];
            valr1 += csr*tex1Dfetch<float>(tex_points, crscol);
            valr2 += csr*tex1Dfetch<float>(tex_points, crscol + 1);
            valr3 += csr*tex1Dfetch<float>(tex_points, crscol + 2);
        }

        valr1 = rhs[id] - valr1;
        valr2 = rhs[id + n_rows] - valr2;
        valr3 = rhs[id + 2*n_rows] - valr3;

        vald1 = valr1/diag;
        vald2 = valr2/diag;
        vald3 = valr3/diag;

        // store
        r[id] = valr1;
        r[id + n_rows] = valr2;
        r[id + 2*n_rows] = valr3;

        d[id] = vald1;
        d[id + n_rows] = vald2;
        d[id + 2*n_rows] = vald3;
    }

    // dot(r,r)

    if(id < n_rows)
    {
        temp[0][tid] = valr1*vald1;
        temp[1][tid] = valr2*vald2;
        temp[2][tid] = valr3*vald3;
    }
    else
    {
        //to be save even if it works without
        temp[0][tid] = 0;
        temp[1][tid] = 0;
        temp[2][tid] = 0;
    }

    __syncthreads();

    if(tid < 3)
    {
        float sum = 0;

        #pragma unroll
        for(int i = 0; i <  THREADS_PER_BLOCK; i++)
        {
            sum+= temp[tid][i];
        }

        atomicAdd(&(greek[5*tid]), sum);
    }
}




__global__ void k_PCG1(cudaTextureObject_t tex_dd, const float* diavalues,const float* ellvalues, const int* ellcols,
                                  const float* crsvalues, const int* crscols, const int* row_ptr,
                                  float *q,
                                  float* greek, const int size)
{
    __shared__ float temp[3][THREADS_PER_BLOCK];
    extern __shared__ float s_data[];

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const int blockstart = row_ptr[blockIdx.x*THREADS_PER_BLOCK];
    const int blockend = row_ptr[((blockIdx.x + 1)*THREADS_PER_BLOCK < size) ? ((blockIdx.x + 1)*THREADS_PER_BLOCK) : size];

    // read in shared memory
    for(int i = blockstart + tid; i < blockend; i+= THREADS_PER_BLOCK)
    {
        float crsval = crsvalues[i];
        s_data[3*(i - blockstart)] = crsval*tex1Dfetch<float>(tex_dd, crscols[i]);
        s_data[3*(i - blockstart) + 1] = crsval*tex1Dfetch<float>(tex_dd, crscols[i] + size);
        s_data[3*(i - blockstart) + 2] = crsval*tex1Dfetch<float>(tex_dd, crscols[i] + 2*size);
    }
    __syncthreads();


    float val0;
    float val1;
    float val2;
    float d1, d2, d3;

    if(id < 3)
    {
        //deltaold = delta; --> deltaold == greek[1], delta == greek[0]
        greek[1 + 5*id] = greek[5*id];
    }

    if(id < size)
    {
        //q = S*d;

        //init with diag
        float diag = diavalues[id];
        d1 = tex1Dfetch<float>(tex_dd, id);
        d2 = tex1Dfetch<float>(tex_dd, id + size);
        d3 = tex1Dfetch<float>(tex_dd, id + 2*size);
        val0 = diag*d1;
        val1 = diag*d2;
        val2 = diag*d3;

        //ell part
        for(int i = 0; i < D_S_ELLROWS; i++)
        {
            float ellval = ellvalues[i*size + id];

            if(fabs(ellval) > 0.0)
            {
                int ellcol = ellcols[i*size + id];
                val0 += ellval*tex1Dfetch<float>(tex_dd, ellcol);
                val1 += ellval*tex1Dfetch<float>(tex_dd, ellcol + size);
                val2 += ellval*tex1Dfetch<float>(tex_dd, ellcol + 2*size);
            }
        }

        //crs part
        const int start = row_ptr[id] - blockstart;
        const int end = row_ptr[id + 1] - blockstart;
        for(int i = start; i < end; i++)
        {
            val0 += s_data[3*i];
            val1 += s_data[3*i + 1];
            val2 += s_data[3*i + 2];
        }

        // store
        q[id] = val0;
        q[id + size] = val1;
        q[id + 2*size] = val2;
    }


    //alpha = delta/d.dot(q); --> alpha == greek[3] --> store alpha = d.dot(q) and do division in 2nd kernel

    if(id < size)
    {
        temp[0][tid] = val0*d1;
        temp[1][tid] = val1*d2;
        temp[2][tid] = val2*d3;
    }
    else
    {
        //to be save even if it works without
        temp[0][tid] = 0;
        temp[1][tid] = 0;
        temp[2][tid] = 0;
    }

    __syncthreads();

    if(tid < 3)
    {
        float sum = 0;

#pragma unroll
        for(int i = 0; i <  THREADS_PER_BLOCK; i++)
        {
            sum+= temp[tid][i];
        }

        atomicAdd(&(greek[3 + 5*tid]), sum);
    }
}

__global__ void k_PCG1_collisions(cudaTextureObject_t tex_dd, const float* diavalues,const float* ellvalues, const int* ellcols,
                                  const float* crsvalues, const int* crscols, const int* row_ptr,
                                  const float* collisioncrsvalues, const int* collisioncrscols, const int* collisionrow_ptr,
                                  float *q,
                                  float* greek, const int size)
{
    __shared__ float temp[3][THREADS_PER_BLOCK];
    extern __shared__ float s_data[];

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const int blockstart = row_ptr[blockIdx.x*THREADS_PER_BLOCK];
    const int blockend = row_ptr[((blockIdx.x + 1)*THREADS_PER_BLOCK < size) ? ((blockIdx.x + 1)*THREADS_PER_BLOCK) : size];

    const int collisionblockstart = collisionrow_ptr[blockIdx.x*THREADS_PER_BLOCK];
    const int collisionblockend = collisionrow_ptr[((blockIdx.x + 1)*THREADS_PER_BLOCK < size) ? ((blockIdx.x + 1)*THREADS_PER_BLOCK) : size];

    // read in shared memory
    for(int i = blockstart + tid; i < blockend; i+= THREADS_PER_BLOCK)
    {
        float crsval = crsvalues[i];
        s_data[3*(i - blockstart)] = crsval*tex1Dfetch<float>(tex_dd, crscols[i]);
        s_data[3*(i - blockstart) + 1] = crsval*tex1Dfetch<float>(tex_dd, crscols[i] + size);
        s_data[3*(i - blockstart) + 2] = crsval*tex1Dfetch<float>(tex_dd, crscols[i] + 2*size);
    }

    for(int i = collisionblockstart + tid; i < collisionblockend; i+= THREADS_PER_BLOCK)
    {
        float crsval = collisioncrsvalues[i];
        s_data[3*(i - collisionblockstart + blockend - blockstart)] = crsval*tex1Dfetch<float>(tex_dd, collisioncrscols[i]);
        s_data[3*(i - collisionblockstart + blockend - blockstart) + 1] = crsval*tex1Dfetch<float>(tex_dd, collisioncrscols[i] + size);
        s_data[3*(i - collisionblockstart + blockend - blockstart) + 2] = crsval*tex1Dfetch<float>(tex_dd, collisioncrscols[i] + 2*size);
    }

    __syncthreads();


    float val0;
    float val1;
    float val2;
    float d1, d2, d3;

    if(id < 3)
    {
        //deltaold = delta; --> deltaold == greek[1], delta == greek[0]
        greek[1 + 5*id] = greek[5*id];
    }

    if(id < size)
    {
        //q = S*d;

        //init with diag
        float diag = diavalues[id];
        d1 = tex1Dfetch<float>(tex_dd, id);
        d2 = tex1Dfetch<float>(tex_dd, id + size);
        d3 = tex1Dfetch<float>(tex_dd, id + 2*size);
        val0 = diag*d1;
        val1 = diag*d2;
        val2 = diag*d3;

        //ell part
        for(int i = 0; i < D_S_ELLROWS; i++)
        {
            float ellval = ellvalues[i*size + id];

            if(fabs(ellval) > 0.0)
            {
                int ellcol = ellcols[i*size + id];
                val0 += ellval*tex1Dfetch<float>(tex_dd, ellcol);
                val1 += ellval*tex1Dfetch<float>(tex_dd, ellcol + size);
                val2 += ellval*tex1Dfetch<float>(tex_dd, ellcol + 2*size);
            }
        }

        //crs part
        const int start = row_ptr[id] - blockstart;
        const int end = row_ptr[id + 1] - blockstart;
        for(int i = start; i < end; i++)
        {
            val0 += s_data[3*i];
            val1 += s_data[3*i + 1];
            val2 += s_data[3*i + 2];
        }

        const int collisionstart = collisionrow_ptr[id] - collisionblockstart + blockend - blockstart;
        const int collisionend = collisionrow_ptr[id + 1] - collisionblockstart + blockend - blockstart;
        for(int i = collisionstart; i < collisionend; i++)
        {
            val0 += s_data[3*i];
            val1 += s_data[3*i + 1];
            val2 += s_data[3*i + 2];
        }

        // store
        q[id] = val0;
        q[id + size] = val1;
        q[id + 2*size] = val2;
    }


    //alpha = delta/d.dot(q); --> alpha == greek[3] --> store alpha = d.dot(q) and do division in 2nd kernel

    if(id < size)
    {
        temp[0][tid] = val0*d1;
        temp[1][tid] = val1*d2;
        temp[2][tid] = val2*d3;
    }
    else
    {
        //to be save even if it works without
        temp[0][tid] = 0;
        temp[1][tid] = 0;
        temp[2][tid] = 0;
    }

    __syncthreads();

    if(tid < 3)
    {
        float sum = 0;

#pragma unroll
        for(int i = 0; i <  THREADS_PER_BLOCK; i++)
        {
            sum+= temp[tid][i];
        }

        atomicAdd(&(greek[3 + 5*tid]), sum);
    }
}



__global__ void k_PCG2(const float *diagvalues,float* r, float *d, float *q,float *x, float *greek, const int size)
{
    __shared__ float temp[3][THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
	
    if(id < 3)
    {
        greek[5*tid] = 0;
    }
	
    float val0;
    float val1;
    float val2;
    if(id < size)
    {
        //alpha = delta/d.dot(q) --> delta == greek[0] (but this is set to 0 for later) --> greek[1] ; d.dot(q) == alpha == greek[3]
        const float alpha0 = greek[1]/greek[3];
        const float alpha1 = greek[1 + 5]/greek[3 + 5];
        const float alpha2 = greek[1 + 10]/greek[3 + 10];
		
        // x += alpha*d;
        x[3*id] += alpha0*d[id];
        x[3*id + 1] += alpha1*d[id + size];
        x[3*id + 2] += alpha2*d[id + 2*size];
		
        //r -= alpha*q;
        val0 = r[id] - alpha0*q[id];
        val1 = r[id + size] - alpha1*q[id + size];
        val2 = r[id + 2*size] - alpha2*q[id + 2*size];
		
        r[id] = val0;
        r[id +size] = val1;
        r[id + 2*size] = val2;
		
    }
	
    //delta = r.dot(r);  --> delta == greek[0]
	
    if(id < size)
    {
        float diaginv = 1.0/diagvalues[id];
		
        temp[0][tid] = val0*val0*diaginv;
        temp[1][tid] = val1*val1*diaginv;
        temp[2][tid] = val2*val2*diaginv;
    }
    else
    {
        //to be save even if it works without
        temp[0][tid] = 0;
        temp[1][tid] = 0;
        temp[2][tid] = 0;
    }
	
    __syncthreads();
	
    if(tid < 3)
    {
        float sum = 0;
		
        #pragma unroll
        for(int i = 0; i <  THREADS_PER_BLOCK; i++)
        {
            sum+= temp[tid][i];
        }
		
        atomicAdd(&(greek[5*tid]), sum);
    }
}


__global__ void k_PCG3(const float *d_diagvalues, float* r, float *d, float *greek, const int size)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(id < size)
    {
        //beta = delta/deltaold;
        const float beta0 = greek[0]/greek[1];
        const float beta1 = greek[0 + 5]/greek[1 + 5];
        const float beta2 = greek[0 + 10]/greek[1 + 10];
		
        //d = r + beta*d;
        float diaginv = 1.0/d_diagvalues[id];
        d[id] = r[id]*diaginv + beta0*d[id];
        d[id + size] = r[id + size]*diaginv + beta1*d[id + size];
        d[id + 2*size] = r[id + 2*size]*diaginv + beta2*d[id + 2*size];
		
        //init greek for next iteration --> alpha == greek[3] has to be set to 0 for 1st kernel
        if(id < 3)
        {
            greek[3+ 5*id] = 0;
			
            // store just for testing --> /todo comment this out
            //greek[4 + 5*id] = greek[5*id]/greek[1 + 5*id];
        }
		
    }
	
	
}


__global__ void k_local_soft(cudaTextureObject_t tex_points, const float* edgesinv, float* Rs, const int* tets, const float* tetweights, const int ntets, const int nanchors)
{
    __shared__ float s_R[9*THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    int size = THREADS_PER_BLOCK;

    if(id < ntets)
    {
        const int ti[] = {tets[id],tets[id + ntets],tets[id + 2*ntets],tets[id + 3*ntets]};

        float p0[3];
        p0[0] = tex1Dfetch<float>(tex_points, 3*ti[0]);
        p0[1] = tex1Dfetch<float>(tex_points, 3*ti[0] + 1);
        p0[2] = tex1Dfetch<float>(tex_points, 3*ti[0] + 2);

        float a0 = tex1Dfetch<float>(tex_points, 3*ti[1] + 0) - p0[0];
        float a1 = tex1Dfetch<float>(tex_points, 3*ti[1] + 1) - p0[1];
        float a2 = tex1Dfetch<float>(tex_points, 3*ti[1] + 2) - p0[2];
        float a3 = tex1Dfetch<float>(tex_points, 3*ti[2] + 0) - p0[0];
        float a4 = tex1Dfetch<float>(tex_points, 3*ti[2] + 1) - p0[1];
        float a5 = tex1Dfetch<float>(tex_points, 3*ti[2] + 2) - p0[2];
        float a6 = tex1Dfetch<float>(tex_points, 3*ti[3] + 0) - p0[0];
        float a7 = tex1Dfetch<float>(tex_points, 3*ti[3] + 1) - p0[1];
        float a8 = tex1Dfetch<float>(tex_points, 3*ti[3] + 2) - p0[2];

        float b0 = edgesinv[id];
        float b1 = edgesinv[id + ntets];
        float b2 = edgesinv[id + 2*ntets];
        float b3 = edgesinv[id + 3*ntets];
        float b4 = edgesinv[id + 4*ntets];
        float b5 = edgesinv[id + 5*ntets];
        float b6 = edgesinv[id + 6*ntets];
        float b7 = edgesinv[id + 7*ntets];
        float b8 = edgesinv[id + 8*ntets];

        float f11 = b0*a0 + b1*a3 + b2*a6;
        float f21 = b0*a1 + b1*a4 + b2*a7;
        float f31 = b0*a2 + b1*a5 + b2*a8;
        float f12 = b3*a0 + b4*a3 + b5*a6;
        float f22 = b3*a1 + b4*a4 + b5*a7;
        float f32 = b3*a2 + b4*a5 + b5*a8;
        float f13 = b6*a0 + b7*a3 + b8*a6;
        float f23 = b6*a1 + b7*a4 + b8*a7;
        float f33 = b6*a2 + b7*a5 + b8*a8;

        float u11, u12, u13, u21, u22, u23, u31, u32, u33;
        float s11, s12, s13, s21, s22, s23, s31, s32, s33;
        float v11, v12, v13, v21, v22, v23, v31, v32, v33;

        svd(f11, f12, f13, f21, f22, f23, f31, f32, f33,
            u11, u12, u13, u21, u22, u23, u31, u32, u33,
      s11, s12, s13, s21, s22, s23, s31, s32, s33,
      v11, v12, v13, v21, v22, v23, v31, v32, v33);

        if((blockIdx.x + 1)*THREADS_PER_BLOCK > ntets)
        {
            size = ntets%THREADS_PER_BLOCK;
        }

        float weight = tetweights[id];

        s_R[tid*3]     				= weight*(u11*v11 + u12*v12 + u13*v13); //R11 R[id*3]
        s_R[tid*3 + 3*size] 		= weight*(u21*v11 + u22*v12 + u23*v13); //R21 R[id*3 + 3*ntets]
        s_R[tid*3 + 6*size] 		= weight*(u31*v11 + u32*v12 + u33*v13); //R22 R[id*3 + 6*ntets]
        s_R[tid*3 + 1] 				= weight*(u11*v21 + u12*v22 + u13*v23); //R11 R[id*3 + 1]
        s_R[tid*3 + 3*size + 1] 	= weight*(u21*v21 + u22*v22 + u23*v23); //R21 R[id*3 + 3*ntets + 1]
        s_R[tid*3 + 6*size + 1] 	= weight*(u31*v21 + u32*v22 + u33*v23); //R22 R[id*3 + 6*ntets + 1]
        s_R[tid*3 + 2] 				= weight*(u11*v31 + u12*v32 + u13*v33); //R11 R[id*3 + 2]
        s_R[tid*3 + 3*size + 2] 	= weight*(u21*v31 + u22*v32 + u23*v33); //R21 R[id*3 + 3*ntets + 2]
        s_R[tid*3 + 6*size + 2] 	= weight*(u31*v31 + u32*v32 + u33*v33); //R22 R[id*3 + 6*ntets + 2]
    }
    __syncthreads();

    if(id < ntets)
    {
        // for each col
        for(int i = 0; i < 3; i++)
        {
            //store 3 elements
            int plus = blockIdx.x*3*THREADS_PER_BLOCK + i*(3*ntets + nanchors);
            Rs[tid + plus] 			= s_R[tid + 3*i*size];
            Rs[tid + plus + size] 		= s_R[tid + (3*i + 1)*size];
            Rs[tid + plus + 2*size] 	= s_R[tid + (3*i + 2)*size];
        }
    }
}

__global__ void k_local_hard(cudaTextureObject_t tex_points, const float* edgesinv, float* Rs, const float* bc, const int* tets, const float* tetweights, const int ntets)
{
    __shared__ float s_R[9*THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    int size = THREADS_PER_BLOCK;

    if(id < ntets)
    {
        const int ti[] = {tets[id],tets[id + ntets],tets[id + 2*ntets],tets[id + 3*ntets]};

        float p0[3];
        p0[0] = tex1Dfetch<float>(tex_points, 3*ti[0]);
        p0[1] = tex1Dfetch<float>(tex_points, 3*ti[0] + 1);
        p0[2] = tex1Dfetch<float>(tex_points, 3*ti[0] + 2);

        float a0 = tex1Dfetch<float>(tex_points, 3*ti[1] + 0) - p0[0];
        float a1 = tex1Dfetch<float>(tex_points, 3*ti[1] + 1) - p0[1];
        float a2 = tex1Dfetch<float>(tex_points, 3*ti[1] + 2) - p0[2];
        float a3 = tex1Dfetch<float>(tex_points, 3*ti[2] + 0) - p0[0];
        float a4 = tex1Dfetch<float>(tex_points, 3*ti[2] + 1) - p0[1];
        float a5 = tex1Dfetch<float>(tex_points, 3*ti[2] + 2) - p0[2];
        float a6 = tex1Dfetch<float>(tex_points, 3*ti[3] + 0) - p0[0];
        float a7 = tex1Dfetch<float>(tex_points, 3*ti[3] + 1) - p0[1];
        float a8 = tex1Dfetch<float>(tex_points, 3*ti[3] + 2) - p0[2];

        float b0 = edgesinv[id];
        float b1 = edgesinv[id + ntets];
        float b2 = edgesinv[id + 2*ntets];
        float b3 = edgesinv[id + 3*ntets];
        float b4 = edgesinv[id + 4*ntets];
        float b5 = edgesinv[id + 5*ntets];
        float b6 = edgesinv[id + 6*ntets];
        float b7 = edgesinv[id + 7*ntets];
        float b8 = edgesinv[id + 8*ntets];

        float f11 = b0*a0 + b1*a3 + b2*a6;
        float f21 = b0*a1 + b1*a4 + b2*a7;
        float f31 = b0*a2 + b1*a5 + b2*a8;
        float f12 = b3*a0 + b4*a3 + b5*a6;
        float f22 = b3*a1 + b4*a4 + b5*a7;
        float f32 = b3*a2 + b4*a5 + b5*a8;
        float f13 = b6*a0 + b7*a3 + b8*a6;
        float f23 = b6*a1 + b7*a4 + b8*a7;
        float f33 = b6*a2 + b7*a5 + b8*a8;

        float u11, u12, u13, u21, u22, u23, u31, u32, u33;
        float s11, s12, s13, s21, s22, s23, s31, s32, s33;
        float v11, v12, v13, v21, v22, v23, v31, v32, v33;

        svd(f11, f12, f13, f21, f22, f23, f31, f32, f33,
            u11, u12, u13, u21, u22, u23, u31, u32, u33,
      s11, s12, s13, s21, s22, s23, s31, s32, s33,
      v11, v12, v13, v21, v22, v23, v31, v32, v33);

        if((blockIdx.x + 1)*THREADS_PER_BLOCK > ntets)
        {
            size = ntets%THREADS_PER_BLOCK;
        }

        float weight = tetweights[id];

        s_R[tid*3]     				= weight*(u11*v11 + u12*v12 + u13*v13); //R11 R[id*3]
        s_R[tid*3 + 3*size] 		= weight*(u21*v11 + u22*v12 + u23*v13); //R21 R[id*3 + 3*ntets]
        s_R[tid*3 + 6*size] 		= weight*(u31*v11 + u32*v12 + u33*v13); //R22 R[id*3 + 6*ntets]
        s_R[tid*3 + 1] 				= weight*(u11*v21 + u12*v22 + u13*v23); //R11 R[id*3 + 1]
        s_R[tid*3 + 3*size + 1] 	= weight*(u21*v21 + u22*v22 + u23*v23); //R21 R[id*3 + 3*ntets + 1]
        s_R[tid*3 + 6*size + 1] 	= weight*(u31*v21 + u32*v22 + u33*v23); //R22 R[id*3 + 6*ntets + 1]
        s_R[tid*3 + 2] 				= weight*(u11*v31 + u12*v32 + u13*v33); //R11 R[id*3 + 2]
        s_R[tid*3 + 3*size + 2] 	= weight*(u21*v31 + u22*v32 + u23*v33); //R21 R[id*3 + 3*ntets + 2]
        s_R[tid*3 + 6*size + 2] 	= weight*(u31*v31 + u32*v32 + u33*v33); //R22 R[id*3 + 6*ntets + 2]
    }
    __syncthreads();

    if(id < ntets)
    {
        // for each col
        for(int i = 0; i < 3; i++)
        {
            //store 3 elements
            int plus = blockIdx.x*3*THREADS_PER_BLOCK + i*3*ntets;
            Rs[tid + plus]              = s_R[tid + 3*i*size]       - bc[tid + plus];
            Rs[tid + plus + size] 		= s_R[tid + (3*i + 1)*size] - bc[tid + plus + size];
            Rs[tid + plus + 2*size] 	= s_R[tid + (3*i + 2)*size] - bc[tid + plus + 2*size];

//            if(tid + plus < 10)
//            {
//                printf("i %d: R %f, bc %f, proj %f\n", tid + plus, s_R[tid + 3*i*size], bc[tid + plus], Rs[tid + plus]);
//            }
        }
    }
}

__global__ void k_update_momentum(const float* masses, const float* velocities, float* points, float* old_points, float* momentum, const float delta, const int numpoints)
{
    __shared__ float s_points[3*THREADS_PER_BLOCK];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int numbefore = 3*THREADS_PER_BLOCK*blockIdx.x;
    if(id < numpoints)
    {
        int size = THREADS_PER_BLOCK;
        if((blockIdx.x + 1)*THREADS_PER_BLOCK > numpoints)
        {
            size = numpoints%THREADS_PER_BLOCK;
        }
		
        old_points[tid + numbefore] = s_points[tid] = points[tid + numbefore];
        old_points[tid + numbefore + size] = s_points[tid + size] = points[tid + numbefore + size];
        old_points[tid + numbefore + 2*size] = s_points[tid + 2*size] = points[tid + numbefore + 2*size];

        //velocities rowwise
        s_points[tid] += delta*velocities[tid + numbefore];
        s_points[tid + size] += delta*velocities[tid + numbefore + size];
        s_points[tid + 2*size] += delta*velocities[tid + numbefore + 2*size];
		
        //points rowwise
        points[tid + numbefore] = s_points[tid];
        points[tid + numbefore + size] = s_points[tid + size];
        points[tid + numbefore + 2*size] = s_points[tid + 2*size];
		
        __syncthreads();
		
        float mass = masses[id];
		
        //mom colwise
        momentum[id] = mass*s_points[3*tid];
        momentum[id + numpoints] = mass*s_points[3*tid + 1];
        momentum[id + 2*numpoints] = mass*s_points[3*tid + 2];
    }
}

__global__ void k_update_velocity(float* velocities, const float* points, const float* old_points, const float damping_div_delta, const int numpoints)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
	
    if(id < 3*numpoints)
    {
        velocities[id] = damping_div_delta*(points[id] - old_points[id]);
    }
}

__global__ void k_update_boundary_soft(const float* anchors, float* projections, const float anchorweight, const int nanchors, const int ntets)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < 3*nanchors)
    {
        const int col = id%3;
        projections[id/3 + 3*ntets + col*(3*ntets + nanchors)] = anchorweight*anchors[id];
    }
}

__global__ void k_update_boundary_hard(cudaTextureObject_t tex_points, const float* ellvalues, const int* ellcols,const float* crsvalues, const int* crscols, const int* row_ptr, float* result,  const int n_rows)
{
    extern __shared__ float s_data[];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const int blockstart = row_ptr[blockIdx.x*THREADS_PER_BLOCK];
    const int blockend = row_ptr[((blockIdx.x + 1)*THREADS_PER_BLOCK < n_rows) ? ((blockIdx.x + 1)*THREADS_PER_BLOCK) : n_rows];

    // read in shared memory
    for(int i = blockstart + tid; i < blockend; i+= THREADS_PER_BLOCK)
    {
        float crsval = crsvalues[i];
        int crscol = 3*(crscols[i]);
        s_data[3*(i - blockstart)] = crsval*tex1Dfetch<float>(tex_points, crscol);
        s_data[3*(i - blockstart) + 1] = crsval*tex1Dfetch<float>(tex_points, crscol + 1);
        s_data[3*(i - blockstart) + 2] = crsval*tex1Dfetch<float>(tex_points, crscol + 2);

    }
    __syncthreads();

    if(id < n_rows)
    {
        //init with zero
        float val0 = 0;
        float val1 = 0;
        float val2 = 0;

        //ell part
        for(int i = 0; i < D_ABC_ELLROWS; i++)
        {
            float ellval = ellvalues[i*n_rows + id];

            if(fabs(ellval) > 0.0)
            {
                int ellcol = 3*(ellcols[i*n_rows + id]);
                val0 += ellval*tex1Dfetch<float>(tex_points, ellcol);
                val1 += ellval*tex1Dfetch<float>(tex_points, ellcol + 1);
                val2 += ellval*tex1Dfetch<float>(tex_points, ellcol + 2);
            }
        }

        //crs part
        const int start = row_ptr[id] - blockstart;
        const int end = row_ptr[id + 1] - blockstart;
        for(int i = start; i < end; i++)
        {
            val0 += s_data[3*i];
            val1 += s_data[3*i + 1];
            val2 += s_data[3*i + 2];
        }

        // store
        result[id] = val0;
        result[id + n_rows] = val1;
        result[id + 2*n_rows] = val2;
    }
}

__global__ void k_prepare_rhs(cudaTextureObject_t tex_proj, const float* ellvalues, const int* ellcols,const float* crsvalues, const int* crscols, const int* row_ptr, const float* mom, float* result,  const int n_rows, const int n_cols)
{
    extern __shared__ float s_data[];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    const int blockstart = row_ptr[blockIdx.x*THREADS_PER_BLOCK];
    const int blockend = row_ptr[((blockIdx.x + 1)*THREADS_PER_BLOCK < n_rows) ? ((blockIdx.x + 1)*THREADS_PER_BLOCK) : n_rows];

    // read in shared memory
    for(int i = blockstart + tid; i < blockend; i+= THREADS_PER_BLOCK)
    {
        float crsval = crsvalues[i];
        int crscol = crscols[i];
        s_data[3*(i - blockstart)] = crsval*tex1Dfetch<float>(tex_proj, crscol);
        s_data[3*(i - blockstart) + 1] = crsval*tex1Dfetch<float>(tex_proj, crscol + n_cols);
        s_data[3*(i - blockstart) + 2] = crsval*tex1Dfetch<float>(tex_proj, crscol + 2*n_cols);
    }
    __syncthreads();

    if(id < n_rows)
    {
        //init with mom
        float val0 = mom[id];
        float val1 = mom[id + n_rows];
        float val2 = mom[id + 2*n_rows];

        //ell part
        for(int i = 0; i < D_AT_ELLROWS; i++)
        {
            float ellval = ellvalues[i*n_rows + id];

            if(fabs(ellval) > 0.0)
            {
                int ellcol = ellcols[i*n_rows + id];
                val0 += ellval*tex1Dfetch<float>(tex_proj, ellcol);
                val1 += ellval*tex1Dfetch<float>(tex_proj, ellcol + n_cols);
                val2 += ellval*tex1Dfetch<float>(tex_proj, ellcol + 2*n_cols);
            }
        }

        //crs part
        const int start = row_ptr[id] - blockstart;
        const int end = row_ptr[id + 1] - blockstart;
        for(int i = start; i < end; i++)
        {
            val0 += s_data[3*i];
            val1 += s_data[3*i + 1];
            val2 += s_data[3*i + 2];
        }

        // store
        result[id] = val0;
        result[id + n_rows] = val1;
        result[id + 2*n_rows] = val2;
    }
}

__global__ void k_update_face_normals(cudaTextureObject_t tex_points, const unsigned int* indices, float* face_normals, const int size)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
    {
        int i0 = indices[3*id];
        int i1 = indices[3*id + 1];
        int i2 = indices[3*id + 2];

        float p0x = tex1Dfetch<float>(tex_points, 3*i0);
        float p0y = tex1Dfetch<float>(tex_points, 3*i0 + 1);
        float p0z = tex1Dfetch<float>(tex_points, 3*i0 + 2);

        float p1x = tex1Dfetch<float>(tex_points, 3*i1);
        float p1y = tex1Dfetch<float>(tex_points, 3*i1 + 1);
        float p1z = tex1Dfetch<float>(tex_points, 3*i1 + 2);

        float p2x = tex1Dfetch<float>(tex_points, 3*i2);
        float p2y = tex1Dfetch<float>(tex_points, 3*i2 + 1);
        float p2z = tex1Dfetch<float>(tex_points, 3*i2 + 2);

        float e01x = p1x - p0x;
        float e01y = p1y - p0y;
        float e01z = p1z - p0z;

        //use p1 to store e02
        p1x = p2x - p0x;
        p1y = p2y - p0y;
        p1z = p2z - p0z;

        //use p0 to store n = cross(e01,e02)
        p0x = e01y*p1z - e01z*p1y;
        p0y = e01z*p1x - e01x*p1z;
        p0z = e01x*p1y - e01y*p1x;

        //normalize n
        float norminv = 1.0/sqrt(p0x*p0x + p0y*p0y + p0z*p0z);
        p0x*=norminv;
        p0y*=norminv;
        p0z*=norminv;

        //store n rowwise
        face_normals[3*id] = p0x;
        face_normals[3*id + 1] = p0y;
        face_normals[3*id + 2] = p0z;
    }
}

__global__ void k_update_vertex_normals(cudaTextureObject_t tex_normals_f, const unsigned int *offsets, const unsigned int *neighbors, float* vertex_normals, const int size)
{

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
    {
        const unsigned int start = offsets[id];
        const unsigned int end = offsets[id + 1];

        float nx = 0;
        float ny = 0;
        float nz = 0;

        for(int i = start; i < end; i++)
        {
            unsigned int n = neighbors[i];
            nx += tex1Dfetch<float>(tex_normals_f, 3*n);
            ny += tex1Dfetch<float>(tex_normals_f, 3*n + 1);
            nz += tex1Dfetch<float>(tex_normals_f, 3*n + 2);
        }

        float norminv = 1.0/sqrt(nx*nx + ny*ny + nz*nz);
        nx*=norminv;
        ny*=norminv;
        nz*=norminv;

        vertex_normals[3*id] = nx;
        vertex_normals[3*id + 1] = ny;
        vertex_normals[3*id + 2] = nz;
    }
}

__global__ void k_transform_boundary(float* points, const float* orig_points, float* normals, const float* orig_normals, const float* tra, const int* VBJptr, const int base, const int baseI, const int baseN, const int numN, unsigned int *aa_indices, const int sizeNormal, const int size)
{
    __shared__ float s_points[THREADS_PER_BLOCK_MOMENTUM];

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // read points into shared memory
    if(id < 3*sizeNormal)
    {
        s_points[tid] = orig_points[3*base + blockIdx.x*THREADS_PER_BLOCK_MOMENTUM + tid];
    }

    __syncthreads();

    if(id < 3*size)
    {
        float ti0, ti1, ti2, ti3;
        int vbj = VBJptr[baseI + id/3];

        int tb = 16*(vbj) + id%3;
        ti0 = tra[tb];
        ti1 = tra[tb + 4];
        ti2 = tra[tb + 8];
        ti3 = tra[tb + 12];

        if(id < 3*sizeNormal)
        {
            float px = s_points[3*(tid/3)];
            float py = s_points[3*(tid/3) + 1];
            float pz = s_points[3*(tid/3) + 2];

            points[3*base + id] = ti0*px + ti1*py + ti2*pz + ti3;

            // transform ignored vertexnormals
            if(id > 3*baseN && id < (3*(baseN + numN)))
            {
                px = orig_normals[3*((id/3) - baseN)];
                py = orig_normals[3*((id/3) - baseN) + 1];
                pz = orig_normals[3*((id/3) - baseN) + 2];

                normals[3*base + id] = ti0*px + ti1*py + ti2*pz;
            }
        }
        else
        {
            unsigned int index = aa_indices[(id - 3*sizeNormal)/3];


            float px = orig_points[3*index];
            float py = orig_points[3*index + 1];
            float pz = orig_points[3*index + 2];

            points[3*index + id%3] = ti0*px + ti1*py + ti2*pz + ti3;
        }
    }
}

__global__ void k_skin_sliding(float* points, unsigned int* interpolIndices, float * interpolparams, const int base_ss, const int base_ss_base, const int size)
{

    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size)
    {
        float h = interpolparams[id];
        float a = interpolparams[id + size];

        unsigned int i00 = 3 * (base_ss_base + interpolIndices[id]);
        unsigned int i01 = 3 * (base_ss_base + interpolIndices[id + size]);
        unsigned int i10 = 3 * (base_ss_base + interpolIndices[id + 2*size]);
        unsigned int i11 = 3 * (base_ss_base + interpolIndices[id + 3*size]);

        float bp00x = points[i00];
        float bp01x = points[i01];
        float bp10x = points[i10];
        float bp11x = points[i11];


        float bp00y = points[i00 + 1];
        float bp01y = points[i01 + 1];
        float bp10y = points[i10 + 1];
        float bp11y = points[i11 + 1];


        float bp00z = points[i00 + 2];
        float bp01z = points[i01 + 2];
        float bp10z = points[i10 + 2];
        float bp11z = points[i11 + 2];

        float h0a0 = (1.0f - a)*(1.0f - h);
        float h0a1 = (a)*(1.0f - h);
        float h1a0 = (1.0f - a)*(h);
        float h1a1 = (a)*(h);

        points[3 * (base_ss + id) + 0] = h0a0*bp00x + h0a1*bp01x + h1a0*bp10x + h1a1*bp11x;
        points[3 * (base_ss + id) + 1] = h0a0*bp00y + h0a1*bp01y + h1a0*bp10y + h1a1*bp11y;
        points[3 * (base_ss + id) + 2] = h0a0*bp00z + h0a1*bp01z + h1a0*bp10z + h1a1*bp11z;
    }
}

__global__ void k_upsampling(cudaTextureObject_t tex_points, cudaTextureObject_t tex_normals_v, float *us_points,float *us_normals,
                                 const float *nis, const float *normal_nis, const unsigned int *lr_index,
                                 const int n_ni, const int size)
{
    __shared__ float s_pts[3*THREADS_PER_BLOCK];
    __shared__ float s_nrmls[3*THREADS_PER_BLOCK];
    const int id = threadIdx.x + blockIdx.x*blockDim.x;
    const int tid = threadIdx.x;

    if(id < size)
    {
        float usx = 0;
        float usy = 0;
        float usz = 0;

        float nx = 0;
        float ny = 0;
        float nz = 0;

        for(int i = 0; i < n_ni; i++)
        {
            unsigned int lr_id = lr_index[id + (i/2)*size];

            float ni = nis[id + i*size];
            float nni = normal_nis[id + i*size];

            unsigned int i0 = 3*(lr_id & 0x0000FFFF);
            usx += ni*tex1Dfetch<float>(tex_points, i0);
            usy += ni*tex1Dfetch<float>(tex_points, i0 + 1);
            usz += ni*tex1Dfetch<float>(tex_points, i0 + 2);

            nx += nni*tex1Dfetch<float>(tex_normals_v, i0);
            ny += nni*tex1Dfetch<float>(tex_normals_v, i0 + 1);
            nz += nni*tex1Dfetch<float>(tex_normals_v, i0 + 2);

            i++;
            if(i == n_ni)
                break;

            lr_id = 3*(lr_id >> 16);

            ni = nis[id + i*size];
            nni = normal_nis[id + i*size];

            usx += ni*tex1Dfetch<float>(tex_points, lr_id);
            usy += ni*tex1Dfetch<float>(tex_points, lr_id + 1);
            usz += ni*tex1Dfetch<float>(tex_points, lr_id + 2);

            nx += nni*tex1Dfetch<float>(tex_normals_v, lr_id);
            ny += nni*tex1Dfetch<float>(tex_normals_v, lr_id + 1);
            nz += nni*tex1Dfetch<float>(tex_normals_v, lr_id + 2);
        }

        s_pts[3*tid] = usx;
        s_pts[3*tid + 1] = usy;
        s_pts[3*tid + 2] = usz;

        float norm = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

        s_nrmls[3*tid] = nx*norm;
        s_nrmls[3*tid + 1] = ny*norm;
        s_nrmls[3*tid + 2] = nz*norm;
    }

    __syncthreads();

    const int base = 3*blockIdx.x*blockDim.x;
    int end = base + 3*THREADS_PER_BLOCK;
    if(end > 3*size)
        end = 3*size;
    for(int i = base + tid; i < end; i+= THREADS_PER_BLOCK)
    {
        us_points[i] = s_pts[i - base];
        us_normals[i] = s_nrmls[i - base];
    }
}


__global__ void k_collision_projection(cudaTextureObject_t tex_points, float* colproj, const int * coltrigs, const float weight, const int size, const int size_nonHand)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
    {
        int i = coltrigs[id];
        int i1 = coltrigs[id + size];
        int i2 = coltrigs[id + 2*size];
        int i3 = coltrigs[id + 3*size];

        //collisionpoint
        float p0x = tex1Dfetch<float>(tex_points, 3*i);
        float p0y = tex1Dfetch<float>(tex_points, 3*i + 1);
        float p0z = tex1Dfetch<float>(tex_points, 3*i + 2);

        // points on triangle
        float p1x = tex1Dfetch<float>(tex_points, 3*i1);
        float p1y = tex1Dfetch<float>(tex_points, 3*i1 + 1);
        float p1z = tex1Dfetch<float>(tex_points, 3*i1 + 2);

        float p2x = tex1Dfetch<float>(tex_points, 3*i2);
        float p2y = tex1Dfetch<float>(tex_points, 3*i2 + 1);
        float p2z = tex1Dfetch<float>(tex_points, 3*i2 + 2);

        float p3x = tex1Dfetch<float>(tex_points, 3*i3);
        float p3y = tex1Dfetch<float>(tex_points, 3*i3 + 1);
        float p3z = tex1Dfetch<float>(tex_points, 3*i3 + 2);

        // triangle edge12
        float e12x = p2x - p1x;
        float e12y = p2y - p1y;
        float e12z = p2z - p1z;

        // triangle edge13
        float e13x = p3x - p1x;
        float e13y = p3y - p1y;
        float e13z = p3z - p1z;

        // triangle normal
        float nx = e12y*e13z - e12z*e13y;
        float ny = e12z*e13x - e12x*e13z;
        float nz = e12x*e13y - e12y*e13x;

        // distance to triangle plane (times norm(n))
        float dot = (p0x - p1x)*nx + (p0y - p1y)*ny + (p0z - p1z)*nz;

        if(id < size_nonHand)
        {
            // side check
            if(dot < 0)
            {
                float dot_div_ndotn = dot/(nx*nx + ny*ny + nz*nz);
                p0x -= dot_div_ndotn*nx;
                p0y -= dot_div_ndotn*ny;
                p0z -= dot_div_ndotn*nz;
            }

            // store 3*p - p1 - p2 - p3
            colproj[id]          =  weight*(3*p0x - p1x - p2x - p3x);
            colproj[id + 4*size] =  weight*(3*p0y - p1y - p2y - p3y);
            colproj[id + 8*size] =  weight*(3*p0z - p1z - p2z - p3z);

            // store p1 - p
            colproj[id + size] =   weight*(p1x - p0x);
            colproj[id + 5*size] = weight*(p1y - p0y);
            colproj[id + 9*size] = weight*(p1z - p0z);
            // store p2 - p
            colproj[id + 2*size] = weight*(p2x - p0x);
            colproj[id + 6*size] = weight*(p2y - p0y);
            colproj[id + 10*size]= weight*(p2z - p0z);
            // store p3 - p
            colproj[id + 3*size] = weight*(p3x - p0x);
            colproj[id + 7*size] = weight*(p3y - p0y);
            colproj[id + 11*size]= weight*(p3z - p0z);
        }
        else
        {
            // side check
            if(dot < 0)
            {
                float dot_div_ndotn = dot/(nx*nx + ny*ny + nz*nz);

                p1x += dot_div_ndotn*nx;
                p1y += dot_div_ndotn*ny;
                p1z += dot_div_ndotn*nz;
                p2x += dot_div_ndotn*nx;
                p2y += dot_div_ndotn*ny;
                p2z += dot_div_ndotn*nz;
                p3x += dot_div_ndotn*nx;
                p3y += dot_div_ndotn*ny;
                p3z += dot_div_ndotn*nz;
            }

            // store p1 - p
            colproj[id + size] =   weight*(p1x);
            colproj[id + 5*size] = weight*(p1y);
            colproj[id + 9*size] = weight*(p1z);
            // store p2 - p
            colproj[id + 2*size] = weight*(p2x);
            colproj[id + 6*size] = weight*(p2y);
            colproj[id + 10*size]= weight*(p2z);
            // store p3 - p
            colproj[id + 3*size] = weight*(p3x);
            colproj[id + 7*size] = weight*(p3y);
            colproj[id + 11*size]= weight*(p3z);
        }


//        if(id <= 1)
//        {
//            printf("id %d: p0=(%f,%f,%f)\n",id,colproj[id],colproj[id + size],colproj[id + 2*size]);
//            printf("id %d: p1=(%f,%f,%f)\n",id,colproj[id + 3*size],colproj[id + 4*size],colproj[id + 5*size]);
//            printf("id %d: p2=(%f,%f,%f)\n",id,colproj[id + 6*size],colproj[id + 7*size],colproj[id + 8*size]);
//            printf("id %d: p3=(%f,%f,%f)\n",id,colproj[id + 9*size],colproj[id + 10*size],colproj[id + 11*size]);
//        }

    }
}

__global__ void k_add_collisions_to_rhs(float* atppmom, const float* colproj, const int * coltrigs, const int size, const int size_noHand, const int nAtppmom)
{
    extern __shared__ float s_colinds[];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    // store all collisionindices in shared memory
    for(int i = tid; i < 4*size; i+= THREADS_PER_BLOCK)
    {
        s_colinds[i] = coltrigs[i];
    }
    __syncthreads();

    if(id < nAtppmom)
    {
        for(int i = 0; i < 4*size; i++)
        {
            // if id and index match
            int idx = s_colinds[i];
            if(idx == id)
            {
                // skip handcollision handvertices
                if(i < size_noHand || i >= size)
                {
                    // add collision constraints to atppmom
                    atppmom[id] += colproj[i];
                    atppmom[id + nAtppmom] += colproj[i + 4*size];
                    atppmom[id + 2*nAtppmom] += colproj[i + 8*size];
                }
            }
        }
    }
}


//----------------------------------------------------------------------------------------------------------------------------------------------

// main global functions

namespace Projective_Skinning
{

void PDsolverCuda::init(Mesh *mesh,
                        const float tetweight, const float anchorweight, const float colweight,
                        const float timestep, const float damping, const bool soft_bc)
{
    PD_Solver::init(mesh, tetweight, anchorweight, colweight, timestep, damping,soft_bc);

    weight_col_ = colweight;
    transformIndices_ = mesh_->transform_indices_;
    for(auto a : mesh_->additional_anchors_)
    {
        additionalAnchors_.push_back(a);
        transformIndices_.push_back(mesh_->additional_anchor_bones_[a] + 1);
    }

    indices_ = mesh_->skin_.sim_indices;
    num_surface_points_ = mesh_->num_simulated_skin_;
    num_unified_points_ = p_points_->cols();

    hardconstraints_ = !soft_bc;

    size_t freeMemBefore, freeMemAfter, totalMem;
    cudaMemGetInfo(&freeMemBefore, &totalMem);

    init_cuda_data();
    init_cuda_normals();
    if(!mesh_->high_res_indices_.empty())
        init_cuda_upsampling(mesh_->high_res_indices_, mesh_->us_neighors_, mesh_->us_Nij_, mesh_->us_normal_Nij_,false);

    init_cuda_textures();

    cudaDeviceSynchronize();
    cudaMemGetInfo(&freeMemAfter, &totalMem);

    std::cout << "GPU-solver uses " << ((freeMemBefore - freeMemAfter)/1024)/1024.0 <<
                 " MB of Device Memory (still " << (double)freeMemAfter/(double)totalMem << "% free)" << std::endl;
}


void PDsolverCudaCollisions::reinit(const std::vector<int> &coltrigs, const int num_ti)
{
    coltrigs_ = coltrigs;
    num_handCols_ = coltrigs.size()/4 - num_ti;

    int nCols = coltrigs_.size()/4;

    {
        //TI GPU
        current_diag_values_ = orig_diag_values_;
        collisions_.resize(coltrigs_.size());
        std::map<int,std::map<int,float>> colmap;
        for(int i = 0; i < nCols - num_handCols_; i++)
        {
            int i0 = coltrigs_[4*i];
            int i1 = coltrigs_[4*i + 1];
            int i2 = coltrigs_[4*i + 2];
            int i3 = coltrigs_[4*i + 3];

            collisions_[i] = i0;
            collisions_[i + nCols] = i1;
            collisions_[i + 2*nCols] = i2;
            collisions_[i + 3*nCols] = i3;

            colmap[i0][i1] -= 1.0;
            colmap[i1][i0] -= 1.0;
            colmap[i0][i2] -= 1.0;
            colmap[i2][i0] -= 1.0;
            colmap[i0][i3] -= 1.0;
            colmap[i3][i0] -= 1.0;

            current_diag_values_[i0] += 3*weight_col_;
            current_diag_values_[i1] += weight_col_;
            current_diag_values_[i2] += weight_col_;
            current_diag_values_[i3] += weight_col_;
        }


        // build collision csr matrix \todo test using simple eigen triplets to build csr instead
        int nnz = 0;
        for(auto &row : colmap)
        {
            nnz+= row.second.size();
        }
        //std::cout << "nnz: " << nnz << std::endl;
        std::vector<float> values(nnz);
        std::vector<int> cols(nnz);
        std::vector<int> rowptr(S_.rows() + 1,nnz);

        int ctr = 0;
        int rctr = 0;
        for(auto &row : colmap)
        {
            while(rctr <= row.first)
            {
                rowptr[rctr] = ctr;
                rctr++;
            }

            for(auto &el : row.second)
            {
                values[ctr] = el.second*weight_col_;
                cols[ctr] = el.first;
                ctr++;
            }
        }


        for(int i = num_ti; i < nCols; i++)
        {
            int i0 = coltrigs_[4*i];
            int i1 = coltrigs_[4*i + 1];
            int i2 = coltrigs_[4*i + 2];
            int i3 = coltrigs_[4*i + 3];

            collisions_[i] = i0;
            collisions_[i + nCols] = i1;
            collisions_[i + 2*nCols] = i2;
            collisions_[i + 3*nCols] = i3;

            current_diag_values_[i1] += weight_col_;
            current_diag_values_[i2] += weight_col_;
            current_diag_values_[i3] += weight_col_;
        }


        // update col csr on gpu
        cudaMemcpy(d_colcsrvals, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colcsrcols, cols.data(), cols.size()*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colcsrrptr, rowptr.data(), rowptr.size()*sizeof(float), cudaMemcpyHostToDevice);
    }

    // update diagonal
    cudaMemcpy(d_S_.diagvalues, current_diag_values_.data(), d_S_.n_rows*sizeof(float), cudaMemcpyHostToDevice);
    // set colindices
    cudaMemcpy(d_collisionIndices_, collisions_.data() ,collisions_.size()*sizeof(int), cudaMemcpyHostToDevice);

}

void PDsolverCuda::reset()
{
    mesh_->reset(true);
    // set velocity to zero (or set oldpoints to points if you have no velocity)
    int N = velocity_.cols()*3;
    velocity_.setZero();
    cudaMemcpy(d_velocities,velocity_.data(),N*sizeof(float),cudaMemcpyHostToDevice);

    //cudaMemcpy(d_old_points,p_points_->data(),nNonBC_*3*sizeof(float),cudaMemcpyHostToDevice);

    cudaMemcpy(d_points,mesh_->orig_vertices_.data(),num_unified_points_*3*sizeof(float),cudaMemcpyHostToDevice);

    if(!hardconstraints_)
    {
        checkCudaErrors(cudaMemcpy(d_anchors, d_points + 3*mesh_->base_sliding_,3*mesh_->num_rigid_*sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void PDsolverCuda::tidyUp()
{
    delete[] h_greek_;
	
    //deleteCRSMatrix(d_At_);
    deleteELLMatrix(d_At2_);
    deleteELLMatrix(d_Abc_);
    deleteHYBMatrix(d_S_);
	
    //cusparseDestroy(cusparseHandle_);
    cudaStreamDestroy(stream1_);
    cudaStreamDestroy(stream2_);
    cudaEventDestroy(event1_);
    cudaEventDestroy(event2_);
    checkCudaErrors(cudaHostUnregister(p_points_->data()));
	
    checkCudaErrors(cudaFree((void*)d_masses));
    checkCudaErrors(cudaFree((void*)d_points));
    checkCudaErrors(cudaFree((void*)d_velocities));
    checkCudaErrors(cudaFree((void*)d_old_points));
    checkCudaErrors(cudaFree((void*)d_momentum));
    checkCudaErrors(cudaFree((void*)d_Atppmom_));
    checkCudaErrors(cudaFree((void*)d_tets));
    checkCudaErrors(cudaFree((void*)d_tetweights));
    checkCudaErrors(cudaFree((void*)d_projections));
    checkCudaErrors(cudaFree((void*)d_bc));
    checkCudaErrors(cudaFree((void*)d_edgeinv));
	
	
    checkCudaErrors(cudaFree((void*)d_d_));
    checkCudaErrors(cudaFree((void*)d_q_));
    checkCudaErrors(cudaFree((void*)d_r_));
    checkCudaErrors(cudaFree((void*)d_greek_));

    checkCudaErrors(cudaFree((void*)d_indices));
    checkCudaErrors(cudaFree((void*)d_offsets));
    checkCudaErrors(cudaFree((void*)d_neighbors));
    checkCudaErrors(cudaFree((void*)d_face_normals));
    checkCudaErrors(cudaFree((void*)d_vertex_normals));
    checkCudaErrors(cudaFree((void*)d_orig_vertex_normals_));

    checkCudaErrors(cudaFree((void*)d_anchors));
    checkCudaErrors(cudaFree((void*)d_collisionIndices_));
    checkCudaErrors(cudaFree((void*)d_collisionprojections_));
    checkCudaErrors(cudaFree((void*)d_colcsrvals));
    checkCudaErrors(cudaFree((void*)d_colcsrcols));
    checkCudaErrors(cudaFree((void*)d_colcsrrptr));

    checkCudaErrors(cudaFree((void*)d_transformationMatrices_));
    checkCudaErrors(cudaFree((void*)d_transformIndices_));
    checkCudaErrors(cudaFree((void*)d_sjParams_));
    checkCudaErrors(cudaFree((void*)d_sjIndices_));
    checkCudaErrors(cudaFree((void*)d_additionalAnchorIndices_));
    checkCudaErrors(cudaFree((void*)d_orig_points));

    if(d_upsampling_neighbors_)     checkCudaErrors(cudaFree((void*)d_upsampling_neighbors_));
    if(d_upsampling_Nij_)           checkCudaErrors(cudaFree((void*)d_upsampling_Nij_));
    if(d_upsampling_normal_Nij_)    checkCudaErrors(cudaFree((void*)d_upsampling_normal_Nij_));
    if(d_hr_normals_)               checkCudaErrors(cudaFree((void*)d_hr_normals_));
    if(d_hr_points_)                checkCudaErrors(cudaFree((void*)d_hr_points_));
}

void PDsolverCuda::update_anchors()
{
    cudaMemcpy(d_transformationMatrices_, mesh_->skeleton_.transformations_[0].data() ,16*mesh_->skeleton_.transformations_.size()*sizeof(float), cudaMemcpyHostToDevice);
    const static int numAnchors = mesh_->num_simple_rigid_ + mesh_->num_ignored_ + mesh_->num_sliding_references_;
    const static int baseTI = mesh_->base_simple_rigid_ - num_surface_points_;
    const static int numAnchorsNAdditionals = numAnchors + additionalAnchors_.size();

    k_transform_boundary<<<(3*numAnchorsNAdditionals + THREADS_PER_BLOCK_MOMENTUM - 1)/THREADS_PER_BLOCK_MOMENTUM, THREADS_PER_BLOCK_MOMENTUM>>>
    (d_points, d_orig_points, d_vertex_normals, d_orig_vertex_normals_, d_transformationMatrices_, d_transformIndices_, mesh_->base_simple_rigid_, baseTI, mesh_->base_ignored_ - mesh_->base_simple_rigid_, mesh_->num_ignored_, d_additionalAnchorIndices_, numAnchors, numAnchorsNAdditionals);

    k_skin_sliding<<<(mesh_->num_sliding_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (d_points, d_sjIndices_, d_sjParams_, mesh_->base_sliding_, mesh_->base_sliding_references_, mesh_->num_sliding_);

    if(!hardconstraints_)
    {
        cudaMemcpy(d_anchors, d_points + 3*mesh_->base_sliding_,3*mesh_->num_rigid_*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void PDsolverCudaCollisions::update_anchors()
{
    PDsolverCuda::update_anchors();
    mesh_->transform_collision_tet_basepoints_();
}

void PDsolverCuda::update_HR(float *d_vbo, float *d_nbo)
{
    //cudaBindTexture(NULL,rhs_tex,d_points,3*num_unified_points_*sizeof(float));
    //cudaBindTexture(NULL,normal_tex,d_vertex_normals,3*num_unified_points_*sizeof(float));

    k_upsampling<<<(numUS_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (tex_points_, tex_normals_v_, d_vbo, d_nbo,d_upsampling_Nij_,d_upsampling_normal_Nij_, d_upsampling_neighbors_,numUSNeighbors_, numUS_);

    // use this version if you do not want to copy to vbo directly
//     k_upsampling<<<(numUS_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
//     (d_hr_points_, d_hr_normals_,d_upsampling_Nij_,d_upsampling_normal_Nij_, d_upsampling_neighbors_,numUSNeighbors_, numUS_);

//     // copy back
//     cudaMemcpy(mesh_->high_res_Vertices_.data(), d_hr_points_, 3*numUS_*sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(mesh_->high_res_vertex_normals_.data(), d_hr_normals_, 3*numUS_*sizeof(float), cudaMemcpyDeviceToHost);


    ////cudaUnbindTexture(rhs_tex);
    //cudaUnbindTexture(normal_tex);
}

void PDsolverCuda::update_normals(bool just_VN)
{
    //cudaBindTexture(NULL, rhs_tex, d_points, 3*num_points_*sizeof(float));

    k_update_face_normals<<<(indices_.size()/3 + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(tex_points_, d_indices, d_face_normals, indices_.size()/3);

    //cudaBindTexture(NULL, rhs_tex, d_face_normals, indices_.size()*sizeof(float));

    k_update_vertex_normals<<<(num_surface_points_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(tex_normals_f_, d_offsets, d_neighbors, d_vertex_normals, num_surface_points_);

    cudaMemcpy(mesh_->vertex_normals_.data(), d_vertex_normals ,3*num_surface_points_*sizeof(float), cudaMemcpyDeviceToHost);

    if(!just_VN)
        cudaMemcpy(mesh_->face_normals_.data(), d_face_normals ,indices_.size()*sizeof(float), cudaMemcpyDeviceToHost);
}



//----------------------------------------------------------------------------------------------------------------------------------------------

// non collision solver

void PDsolverCuda::update_skin(int iterations)
{
    // preparations
    if(hardconstraints_)
    {
        //cudaBindTexture(NULL, rhs_tex, d_points, 3*num_points_*sizeof(float));

        k_update_boundary_hard<<<(d_Abc_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 3*d_Abc_.n_maxNNZperBlock*sizeof(float)>>>
        (tex_points_, d_Abc_.ellvalues, d_Abc_.ellcols, d_Abc_.crsvalues, d_Abc_.cols, d_Abc_.row_ptr, d_bc, d_Abc_.n_rows);
    }
    else
    {
        k_update_boundary_soft<<<(3*mesh_->num_rigid_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (d_anchors, d_projections, weight_anc_, mesh_->num_rigid_, num_tets_);
    }
    cudaEventSynchronize(event2_);

    // momentum update
    k_update_momentum<<<(num_non_boundary_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (d_masses, d_velocities, d_points, d_old_points, d_momentum, dt_, num_non_boundary_);

    for(int it = 0; it < iterations; it++)
    {
        update_local();
        update_global();
    }

    // copy points back to host async (not needed if non-collision simulation and directly streamed to OGL)
    cudaMemcpyAsync(p_points_->data() ,d_points, 3*num_unified_points_*sizeof(float), cudaMemcpyDeviceToHost, stream1_);
    cudaEventRecord(event1_, stream1_);

    //velocity update
    k_update_velocity<<<(3*num_non_boundary_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream2_>>>
    (d_velocities, d_points, d_old_points, damping_, num_non_boundary_);
    cudaEventRecord(event2_, stream2_);

    cudaEventSynchronize(event1_);
}

void PDsolverCuda::update_local()
{
    //cudaBindTexture(NULL, rhs_tex, d_points, 3*num_points_*sizeof(float));
    if(hardconstraints_)
    {
        k_local_hard<<<(num_tets_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (tex_points_, d_edgeinv, d_projections, d_bc, d_tets, d_tetweights, num_tets_);
    }
    else
    {
        k_local_soft<<<(num_tets_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (tex_points_, d_edgeinv, d_projections, d_tets, d_tetweights, num_tets_, mesh_->num_rigid_);
    }

    // At*proj + mom
    //cudaBindTexture(NULL, rhs_tex, d_projections, 3*num_projections_*sizeof(float));
    k_prepare_rhs<<<(d_At2_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 3*d_At2_.n_maxNNZperBlock*sizeof(float)>>>
    (tex_proj_, d_At2_.ellvalues, d_At2_.ellcols, d_At2_.crsvalues, d_At2_.cols, d_At2_.row_ptr, d_momentum, d_Atppmom_, d_At2_.n_rows, num_projections_);

}

void PDsolverCuda::update_global()
{

    // PCG
    //cudaBindTexture(NULL, rhs_tex, d_points, 3*num_points_*sizeof(float));
    k_init_PCG<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (tex_points_, d_S_.diagvalues, d_S_.ellvalues, d_S_.ellcols, d_S_.crsvalues, d_S_.cols, d_S_.row_ptr, d_d_, d_r_, d_greek_ ,d_Atppmom_, d_S_.n_rows);

    int i = 0;
    //cudaBindTexture(NULL, rhs_tex, d_d_, 3*d_S_.n_rows*sizeof(float));
    while(i < 10)
    {

        k_PCG1<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 3*d_S_.n_maxNNZperBlock*sizeof(float)>>>
        (tex_dd_, d_S_.diagvalues, d_S_.ellvalues, d_S_.ellcols, d_S_.crsvalues, d_S_.cols, d_S_.row_ptr, d_q_, d_greek_, d_S_.n_rows);

        k_PCG2<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (d_S_.diagvalues, d_r_, d_d_, d_q_, d_points, d_greek_, d_S_.n_rows);

        if(i != 9)
            k_PCG3<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (d_S_.diagvalues, d_r_, d_d_, d_greek_, d_S_.n_rows);

        i++;
    }
}

// collision solver

void PDsolverCudaCollisions::update_local()
{
    PDsolverCuda::update_local();

    //cudaBindTexture(NULL, rhs_tex, d_points, 3*num_points_*sizeof(float));

    k_collision_projection<<<(collisions_.size()/4 + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (tex_points_, d_collisionprojections_, d_collisionIndices_, weight_col_, collisions_.size()/4, collisions_.size()/4 - num_handCols_);

    // todo: fuse this with prepare_rhs kernel
    k_add_collisions_to_rhs<<<(num_non_boundary_ + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, collisions_.size()*sizeof(float)>>>
    (d_Atppmom_, d_collisionprojections_, d_collisionIndices_, collisions_.size()/4, collisions_.size()/4 - num_handCols_, num_non_boundary_);
}

void PDsolverCudaCollisions::update_global()
{
    k_init_PCG_collisions<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (tex_points_, d_S_.diagvalues, d_S_.ellvalues, d_S_.ellcols, d_S_.crsvalues, d_S_.cols, d_S_.row_ptr, d_colcsrvals, d_colcsrcols, d_colcsrrptr, d_d_, d_r_, d_greek_ ,d_Atppmom_, d_S_.n_rows);

    int i = 0;
    //cudaBindTexture(NULL, rhs_tex, d_d_, 3*d_S_.n_rows*sizeof(float));
    while(i < 10)
    {
        k_PCG1_collisions<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK, 3*(d_S_.n_maxNNZperBlock + 6*(collisions_.size()/4))*sizeof(float)>>>
        (tex_dd_, d_S_.diagvalues, d_S_.ellvalues, d_S_.ellcols, d_S_.crsvalues, d_S_.cols, d_S_.row_ptr, d_colcsrvals, d_colcsrcols, d_colcsrrptr, d_q_, d_greek_, d_S_.n_rows);

        k_PCG2<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
        (d_S_.diagvalues, d_r_, d_d_, d_q_, d_points, d_greek_, d_S_.n_rows);

        if(i != 9)
            k_PCG3<<<(d_S_.n_rows + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (d_S_.diagvalues, d_r_, d_d_, d_greek_, d_S_.n_rows);

        i++;
    }
}


void PDsolverCuda::update_ogl_sim_mesh_buffers(float *vbo, float *nbo)
{
    cudaMemcpy(vbo ,d_points, 3*num_unified_points_*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(nbo ,d_vertex_normals, 3*num_unified_points_*sizeof(float), cudaMemcpyDeviceToDevice);
}

//----------------------------------------------------------------------------------------------------------------------------------------------

// initializer helper functions

void PDsolverCuda::init_cuda_data()
{
    //init CUDA data
    checkCudaErrors(cudaMalloc((void**)&d_velocities,3*num_non_boundary_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_points,3*num_unified_points_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_orig_points,3*num_unified_points_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_old_points,3*num_non_boundary_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_momentum,3*num_non_boundary_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_Atppmom_,3*num_non_boundary_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_masses,num_non_boundary_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_edgeinv,9*num_tets_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_projections,3*num_projections_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_bc,3*num_projections_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_tetweights,num_tets_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_tets,tets_.size()*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_anchors,3*mesh_->num_rigid_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_collisionIndices_,4*num_surface_points_*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_collisionprojections_,4*3*num_surface_points_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_colcsrvals,4*num_surface_points_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_colcsrcols,4*num_surface_points_*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_colcsrrptr,(S_.rows() + 1)*sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&d_transformationMatrices_,16*mesh_->skeleton_.transformations_.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_transformIndices_,transformIndices_.size()*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_additionalAnchorIndices_, additionalAnchors_.size()*sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(d_points, p_points_->data() ,3*num_unified_points_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_orig_points, mesh_->orig_vertices_.data() ,3*num_unified_points_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_velocities, velocity_.data() ,3*num_non_boundary_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_old_points, p_points_->data() ,3*num_non_boundary_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_momentum, momentum_.data() ,3*num_non_boundary_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_masses, masses_.data() ,num_non_boundary_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_tetweights, tetweights_.data(),num_tets_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_transformIndices_, transformIndices_.data(),transformIndices_.size()*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_additionalAnchorIndices_, additionalAnchors_.data() ,additionalAnchors_.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));

    // sliding coalesed storage
    int nSJP = mesh_->sj_parameters_.size();
    IndexVector sjinds(4*nSJP);
    std::vector<float> sjparams(2*nSJP);
    for(int i = 0; i < mesh_->num_sliding_; i++)
    {
        sjinds[i + 0*nSJP] = mesh_->sj_parameters_[i].i00;
        sjinds[i + 1*nSJP] = mesh_->sj_parameters_[i].i01;
        sjinds[i + 2*nSJP] = mesh_->sj_parameters_[i].i10;
        sjinds[i + 3*nSJP] = mesh_->sj_parameters_[i].i11;

        sjparams[i + 0*nSJP] = mesh_->sj_parameters_[i].p_h;
        sjparams[i + 1*nSJP] = mesh_->sj_parameters_[i].p_angle;
    }

    checkCudaErrors(cudaMalloc((void**)&d_sjParams_,sjparams.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_sjIndices_,sjinds.size()*sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(d_sjParams_, sjparams.data() ,sjparams.size()*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sjIndices_, sjinds.data() ,sjinds.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));

    //old anchors
    //Projective_Skinning::MatX3 anchtrans = anchors_.transpose();
    //checkCudaErrors(cudaMemcpy(d_anchors, anchtrans.data() ,3*anchors_.cols()*sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_anchors, d_points + 3*mesh_->base_sliding_,3*mesh_->num_rigid_*sizeof(float), cudaMemcpyDeviceToDevice));

    float* h_edgeinv = new float[9*num_tets_];
    for(int j = 0; j < num_tets_ ;j++)
    {
        for(int i = 0; i < 9; i++)
        {
            int j2 = 3*j + i/3;
            int i2 = i%3;
            h_edgeinv[j + i*(num_tets_)] = rest_edge_inverse_(i2,j2);
        }
    }
    checkCudaErrors(cudaMemcpy(d_edgeinv, h_edgeinv,9*num_tets_*sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_edgeinv;


    int *h_tets = new int[tets_.size()];
    for(int i = 0; i < num_tets_; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            h_tets[j*num_tets_ + i] = tets_[4*i + j];
        }
    }
    checkCudaErrors(cudaMemcpy(d_tets, h_tets,tets_.size()*sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_tets;


    //Matrices

    //convertEigenMatrixToCRSDevicePointers(A, d_At_);
    convertEigenMatrixToELLDevicePointers(At_.transpose(), d_At2_, D_AT_ELLROWS);
    convertEigenMatrixToELLDevicePointers(A_bc_.transpose(), d_Abc_, D_ABC_ELLROWS);
    convertEigenMatrixToHYBDevicePointers(S_, d_S_, D_S_ELLROWS);

    // store original diagvalues
    orig_diag_values_.resize(S_.cols());
    checkCudaErrors(cudaMemcpy(orig_diag_values_.data(), d_S_.diagvalues, S_.cols()*sizeof(float), cudaMemcpyDeviceToHost));
    current_diag_values_ = orig_diag_values_;

    //initCuSparseAndMatrixDesc(cusparseHandle_, A_descr_);
    initMPCGArrays(d_d_, d_q_, d_r_, h_greek_, d_greek_, num_non_boundary_);

    //checkCudaErrors(cudaFree((void*)d_points));

    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
    cudaEventCreateWithFlags(&event1_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event2_, cudaEventDisableTiming);
    checkCudaErrors(cudaHostRegister(p_points_->data(), num_unified_points_*3*sizeof(float), 0));
}

void PDsolverCuda::init_cuda_normals()
{
    //normal stuff
    checkCudaErrors(cudaMalloc((void**)&d_indices,indices_.size()*sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d_indices, &(indices_[0]) ,indices_.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));
    unsigned int *h_neighbors, *h_offsets;
    int ctr = 0;
    unsigned int i0,i1,i2;
    std::vector<std::vector<unsigned int>> neighbors;
    neighbors.resize(num_surface_points_);
    for(unsigned int p = 0; p < num_surface_points_; p++)
    {
        for(unsigned int j = 0; j < indices_.size(); j+=3)
        {
            i0 = indices_[j];
            i1 = indices_[j + 1];
            i2 = indices_[j + 2];

            if(i0 == p || i1 == p || i2 == p)
            {
                neighbors[p].push_back(j/3);
            }

        }
    }
    for(auto vi : neighbors)
    {
        for(auto n : vi)
        {
            ctr++;
        }
    }
    h_neighbors = new unsigned int[ctr];
    h_offsets = new unsigned int[neighbors.size() + 1];
    ctr = 0;
    for(int k = 0; k < neighbors.size(); k++)
    {
        h_offsets[k] = ctr;
        for(auto n : neighbors[k])
        {
            h_neighbors[ctr] = n;
            ctr++;
        }

    }
    h_offsets[neighbors.size()] = ctr;

    checkCudaErrors(cudaMalloc((void**)&d_neighbors,ctr*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_offsets,(neighbors.size() + 1)*sizeof(unsigned int)));

    checkCudaErrors(cudaMemcpy(d_neighbors, h_neighbors ,ctr*sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_offsets, h_offsets ,(neighbors.size() + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_face_normals,indices_.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_vertex_normals,3*num_unified_points_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_orig_vertex_normals_, 3*mesh_->num_ignored_*sizeof(float)));

    Mat3X orig_ignored_vertex_normals = mesh_->vertex_normals_.block(0,mesh_->base_ignored_, 3,mesh_->num_ignored_);
    checkCudaErrors(cudaMemcpy(d_orig_vertex_normals_, orig_ignored_vertex_normals.data() ,3*mesh_->num_ignored_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vertex_normals + 3*mesh_->base_ignored_, orig_ignored_vertex_normals.data() ,3*mesh_->num_ignored_*sizeof(float), cudaMemcpyHostToDevice));

    delete [] h_offsets;
    delete [] h_neighbors;
}

void PDsolverCuda::init_cuda_textures()
{
    tex_points_ = 0;
    tex_normals_v_ = 0;
    tex_normals_f_ = 0;
    tex_proj_ = 0;
    tex_dd_ = 0;
    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    resDesc.res.linear.devPtr = d_points;
    resDesc.res.linear.sizeInBytes = 3*num_unified_points_*sizeof(float);
    checkCudaErrors(cudaCreateTextureObject(&tex_points_, &resDesc, &texDesc, NULL));

    resDesc.res.linear.devPtr = d_vertex_normals;
    resDesc.res.linear.sizeInBytes = 3*num_surface_points_*sizeof(float);
    checkCudaErrors(cudaCreateTextureObject(&tex_normals_v_, &resDesc, &texDesc, NULL));

    resDesc.res.linear.devPtr = d_face_normals;
    resDesc.res.linear.sizeInBytes = indices_.size()*sizeof(float);
    checkCudaErrors(cudaCreateTextureObject(&tex_normals_f_, &resDesc, &texDesc, NULL));
    
    resDesc.res.linear.devPtr = d_projections;
    resDesc.res.linear.sizeInBytes = 3*num_projections_*sizeof(float);
    checkCudaErrors(cudaCreateTextureObject(&tex_proj_, &resDesc, &texDesc, NULL));
    
    resDesc.res.linear.devPtr = d_d_;
    resDesc.res.linear.sizeInBytes = 3*d_S_.n_rows*sizeof(float);
    checkCudaErrors(cudaCreateTextureObject(&tex_dd_, &resDesc, &texDesc, NULL));
}

void PDsolverCuda::init_cuda_upsampling(const Projective_Skinning::IndexVector &us_indices, std::vector<std::vector<unsigned int> > &us_neighbors, std::vector<std::vector<float> > &us_Nij, std::vector<std::vector<float> > &us_normal_Nij, bool duplicate)
{
    duplicate_HR_points_ = duplicate;

    using namespace  Projective_Skinning;

    numUS_ = us_neighbors.size();
    numUSNeighbors_ = us_neighbors[0].size();
    numDuplicatedPoints_ = us_indices.size();

    checkCudaErrors(cudaMalloc((void**)&d_hr_points_,3*numUS_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_hr_normals_,3*numUS_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_upsampling_Nij_,numUSNeighbors_*numUS_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_upsampling_normal_Nij_,numUSNeighbors_*numUS_*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_upsampling_neighbors_,numUSNeighbors_*numUS_*sizeof(unsigned int)));


    float* h_Nij = new float[numUSNeighbors_*numUS_];
    float* h_normal_Nij = new float[numUSNeighbors_*numUS_];
    unsigned int * h_nb = new unsigned int[numUSNeighbors_*numUS_];
    for(int i = 0; i<  numUS_; i++)
    {
        for(int j = 0; j < numUSNeighbors_; j++)
        {
            h_Nij[i + j*numUS_] = us_Nij[i][j];
            h_normal_Nij[i + j*numUS_] = us_normal_Nij[i][j];
            h_nb[i + j*numUS_] = us_neighbors[i][j];
        }
    }

    // packed neighbors
    for(int i = 0; i<  numUS_; i++)
    {
        for(int j = 0; j < (numUSNeighbors_ + 1)/2; j++)
        {
            unsigned int i0 = us_neighbors[i][2*j];
            unsigned int i1 = 0;
            if(2*j +1 < numUSNeighbors_)
                i1 = us_neighbors[i][2*j + 1];

            i1 <<= 16;
            unsigned int i01 = i0 + i1;
            h_nb[i + j*numUS_] = i01;
        }
    } // todo copy just half of neighbor vector if all works
    checkCudaErrors(cudaMemcpy(d_upsampling_Nij_, h_Nij,numUSNeighbors_*numUS_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_upsampling_normal_Nij_, h_normal_Nij,numUSNeighbors_*numUS_*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_upsampling_neighbors_, h_nb,numUSNeighbors_*numUS_*sizeof(unsigned int), cudaMemcpyHostToDevice));

    delete[] h_Nij;
    delete[] h_normal_Nij;
    delete[] h_nb;
}

//----------------------------------------------------------------------------------------------------------------------------------------------

// Matrix initializer and delete functions

void PDsolverCuda::convertEigenMatrixToCRSPointers(const Projective_Skinning::SparseMatrix &S, CRSMatrix& out_crs)
{
    out_crs.device = false;
    out_crs.values = new float[S.nonZeros()];
    out_crs.cols = new int[S.nonZeros()];
    out_crs.row_ptr = new int[S.cols() + 1];
    out_crs.nnz = S.nonZeros();
    out_crs.n_rows = S.cols();

    for(int i = 0; i < out_crs.nnz; i++)
    {
        out_crs.values[i] = S.valuePtr()[i];
        out_crs.cols[i] = S.innerIndexPtr()[i];
    }
    for(int i = 0; i < out_crs.n_rows; i++)
    {
        out_crs.row_ptr[i] = S.outerIndexPtr()[i];
    }
    out_crs.row_ptr[out_crs.n_rows] = out_crs.nnz;
}

void PDsolverCuda::convertEigenMatrixToCRSDevicePointers(const Projective_Skinning::SparseMatrix &S, CRSMatrix& out_crs)
{
    out_crs.device = true;
    float *values = new float[S.nonZeros()];
    int *cols = new int[S.nonZeros()];
    int* row_ptr = new int[S.cols() + 1];
    out_crs.nnz = S.nonZeros();
    out_crs.n_rows = S.cols();

    for(int i = 0; i < out_crs.nnz; i++)
    {
        values[i] = S.valuePtr()[i];
        cols[i] = S.innerIndexPtr()[i];
    }
    for(int i = 0; i < out_crs.n_rows; i++)
    {
        row_ptr[i] = S.outerIndexPtr()[i];
    }
    row_ptr[out_crs.n_rows] = out_crs.nnz;

    checkCudaErrors(cudaMalloc((void**)&out_crs.values, out_crs.nnz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_crs.cols, out_crs.nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&out_crs.row_ptr, (out_crs.n_rows + 1)*sizeof(int)));
    checkCudaErrors(cudaMemcpy(out_crs.values, values,out_crs.nnz*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_crs.cols, cols,out_crs.nnz*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_crs.row_ptr, row_ptr,(out_crs.n_rows+1)*sizeof(int), cudaMemcpyHostToDevice));

    delete[] values;
    delete[] cols;
    delete[] row_ptr;
}

void PDsolverCuda::convertEigenMatrixToHYBPointers(const Projective_Skinning::SparseMatrix &S, HYBMatrix& out_hyb, int numellrows)
{
    if(numellrows < 1) std::cerr << "wrong numellrows < 1" << std::endl;

    out_hyb.device = false;
    out_hyb.diagvalues = new float[S.cols()];
    out_hyb.ellvalues = new float[numellrows*S.cols()];
    out_hyb.ellcols = new int[numellrows*S.cols()];
    out_hyb.row_ptr = new int[S.cols() + 1];
    out_hyb.nnz = S.nonZeros();
    out_hyb.n_rows = S.cols();
    out_hyb.n_ellrows = numellrows;

    //fill row pointer
    for(int i = 0; i < out_hyb.n_rows; i++)
    {
        out_hyb.row_ptr[i] = S.outerIndexPtr()[i];
    }
    out_hyb.row_ptr[out_hyb.n_rows] = out_hyb.nnz;

    // count remaining csr
    int remainingcsr = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start = out_hyb.row_ptr[i];
        int end = out_hyb.row_ptr[i+1];

        int rest = end - start - numellrows - 1;
        if(rest > 0)
        {
            remainingcsr += rest;
        }
    }
    out_hyb.crsvalues = new float[remainingcsr];
    out_hyb.cols = new int[remainingcsr];

    // fill diag, ell and csr
    int i2 = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start = out_hyb.row_ptr[i];
        int end = out_hyb.row_ptr[i+1];

        int numell = 0;
        bool diagdone = false;
        for(int j = start; j < end; j++)
        {
            //std::cout << j << std::endl;
            int col = S.innerIndexPtr()[j];
            if(col == i)
            {
                if(diagdone)
                {
                    std::cerr << "this should not happen: two diagonal elements in one row!!! " << std::endl;
                }
                out_hyb.diagvalues[i] = S.valuePtr()[j];
                //std::cout << "write dia " << i << " " << S.valuePtr()[j] << std::endl;
                diagdone = true;
            }
            else
            {
                if(numell < numellrows)
                {
                    out_hyb.ellvalues[numell*out_hyb.n_rows + i] = S.valuePtr()[j];
                    out_hyb.ellcols[numell*out_hyb.n_rows + i] = col;
                    //std::cout << "write ellv " << numell*out_hyb.n_rows + i << " " << S.valuePtr()[j] << std::endl;
                    numell++;
                }
                else
                {
                    //std::cout << "csr " << j -start << std::endl;
                    out_hyb.crsvalues[i2] = S.valuePtr()[j];
                    out_hyb.cols[i2] = col;
                    i2++;
                }
            }
        }

        // fill some parts of ell with zeros

        for(int j = numell; j < numellrows; j++)
        {
            out_hyb.ellvalues[j*out_hyb.n_rows + i] = 0.0;
            out_hyb.ellcols[j*out_hyb.n_rows + i] = 0;
            //std::cout << "write ell " << j*out_hyb.n_rows + i << " " << 0.0 << std::endl;
        }

    }

    // change csr row pointer to remainingcsr
    int csrdiff = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start = out_hyb.row_ptr[i];
        int end = out_hyb.row_ptr[i+1];

        //std::cout << i << " " << end - start << std::endl;
        int rest = end - start - numellrows - 1;
        out_hyb.row_ptr[i] = csrdiff;

        csrdiff += std::max(0,rest);
    }
    out_hyb.row_ptr[out_hyb.n_rows] = csrdiff;

    out_hyb.n_csr = remainingcsr;

}

void PDsolverCuda::convertEigenMatrixToHYBDevicePointers(const Projective_Skinning::SparseMatrix &S, HYBMatrix& out_hyb, int numellrows)
{
    if(numellrows < 1) std::cerr << "wrong numellrows < 1" << std::endl;
    if(numellrows < 0) numellrows = 0;

    out_hyb.device = true;
    float* diagvalues = new float[S.cols()];
    float* ellvalues = new float[numellrows*S.cols()];
    int *ellcols = new int[numellrows*S.cols()];
    int *row_ptr = new int[S.cols() + 1];
    out_hyb.nnz = S.nonZeros();
    out_hyb.n_rows = S.cols();
    out_hyb.n_ellrows = numellrows;

    //fill row pointer
    for(int i = 0; i < out_hyb.n_rows; i++)
    {
        row_ptr[i] = S.outerIndexPtr()[i];
    }
    row_ptr[out_hyb.n_rows] = out_hyb.nnz;

    // count remaining csr
    int remainingcsr = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int rest = end - start - numellrows - 1;
        if(rest > 0)
        {
            remainingcsr += rest;
        }
    }
    float *crsvalues = new float[remainingcsr];
    int *cols = new int[remainingcsr];

    // fill diag, ell and csr
    int i2 = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int numell = 0;
        bool diagdone = false;
        for(int j = start; j < end; j++)
        {
            //std::cout << j << std::endl;
            int col = S.innerIndexPtr()[j];
            if(col == i)
            {
                if(diagdone)
                {
                    std::cerr << "this should not happen: two diagonal elements in one row!!! " << std::endl;
                }
                diagvalues[i] = S.valuePtr()[j];
                //std::cout << "write dia " << i << " " << S.valuePtr()[j] << std::endl;
                diagdone = true;
            }
            else
            {
                if(numell < numellrows)
                {
                    ellvalues[numell*out_hyb.n_rows + i] = S.valuePtr()[j];
                    ellcols[numell*out_hyb.n_rows + i] = col;
                    //std::cout << "write ellv " << numell*out_hyb.n_rows + i << " " << S.valuePtr()[j] << std::endl;
                    numell++;
                }
                else
                {
                    //std::cout << "csr " << j -start << std::endl;
                    crsvalues[i2] = S.valuePtr()[j];
                    cols[i2] = col;
                    i2++;
                }
            }
        }

        // fill some parts of ell with zeros

        for(int j = numell; j < numellrows; j++)
        {
            ellvalues[j*out_hyb.n_rows + i] = 0.0;
            ellcols[j*out_hyb.n_rows + i] = 0;
            //std::cout << "write ell " << j*out_hyb.n_rows + i << " " << 0.0 << std::endl;
        }

    }

    // change csr row pointer to remainingcsr
    int csrdiff = 0;
    for(int i = 0; i < out_hyb.n_rows ;i++)
    {
        int start =row_ptr[i];
        int end = row_ptr[i+1];

        //std::cout << i << " " << end - start << std::endl;
        int rest = end - start - numellrows - 1;
        row_ptr[i] = csrdiff;

        csrdiff += std::max(0,rest);
    }
    row_ptr[out_hyb.n_rows] = csrdiff;

    out_hyb.n_csr = remainingcsr;

    checkCudaErrors(cudaMalloc((void**)&out_hyb.diagvalues, 	out_hyb.n_rows*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.orig_diagvalues, 	out_hyb.n_rows*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.ellvalues, 	(out_hyb.n_rows*numellrows)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.ellcols, 		(out_hyb.n_rows*numellrows)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.crsvalues, 	(out_hyb.n_csr)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.cols, 		(out_hyb.n_csr)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&out_hyb.row_ptr,(out_hyb.n_rows + 1)*sizeof(int)));

    checkCudaErrors(cudaMemcpy(out_hyb.diagvalues, diagvalues,	out_hyb.n_rows*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.orig_diagvalues, diagvalues,	out_hyb.n_rows*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.ellvalues, ellvalues,	(out_hyb.n_rows*numellrows)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.ellcols, ellcols,				(out_hyb.n_rows*numellrows)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.crsvalues, crsvalues,	(out_hyb.n_csr)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.cols, cols,				(out_hyb.n_csr)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_hyb.row_ptr, row_ptr,			(out_hyb.n_rows + 1)*sizeof(int), cudaMemcpyHostToDevice));

    int maxNondiagPerBlock = 0;
    for(int i = 0; i < out_hyb.n_rows; i+=THREADS_PER_BLOCK)
    {
        int nd;
        if(i + THREADS_PER_BLOCK < out_hyb.n_rows)
        {
            nd = row_ptr[i + THREADS_PER_BLOCK] - row_ptr[i];
        }
        else
        {
            nd =  row_ptr[out_hyb.n_rows] - row_ptr[i];
        }
        if(nd > maxNondiagPerBlock)
            maxNondiagPerBlock = nd;
    }
    out_hyb.n_maxNNZperBlock = maxNondiagPerBlock;
    if(maxNondiagPerBlock*12 > 48000)
        std::cerr << "\n\n WARNING!!!!! \n\n Too much shared memory needed!! \n\n";

    delete[] diagvalues;
    delete[] ellvalues;
    delete[] ellcols;
    delete[] crsvalues;
    delete[] cols;
    delete[] row_ptr;

}

void PDsolverCuda::convertEigenMatrixToELLDevicePointers(const Projective_Skinning::SparseMatrix &S, ELLMatrix& out_ell, const int numellrows)
{
    if(numellrows < 1) std::cerr << "numellrows < 1 -> csr matrix" << std::endl;

    float* ellvalues = new float[numellrows*S.cols()];
    int* ellcols = new int[numellrows*S.cols()];
    int* row_ptr = new int[S.cols() + 1];
    out_ell.nnz = S.nonZeros();
    out_ell.n_rows = S.cols();
    out_ell.n_ellrows = numellrows;
    out_ell.device = true;

    //fill row pointer
    for(int i = 0; i < out_ell.n_rows; i++)
    {
        row_ptr[i] = S.outerIndexPtr()[i];
    }
    row_ptr[out_ell.n_rows] = out_ell.nnz;

    // count remaining csr
    int remainingcsr = 0;
    for(int i = 0; i < out_ell.n_rows ;i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int rest = end - start - numellrows;
        if(rest > 0)
        {
            remainingcsr += rest;
        }
    }
    float* crsvalues = new float[remainingcsr];
    int* cols = new int[remainingcsr];

    // fill diag, ell and csr
    int i2 = 0;
    for(int i = 0; i < out_ell.n_rows ;i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        int numell = 0;
        for(int j = start; j < end; j++)
        {
            //std::cout << j << std::endl;
            int col = S.innerIndexPtr()[j];

            if(numell < numellrows)
            {
                ellvalues[numell*out_ell.n_rows + i] = S.valuePtr()[j];
                ellcols[numell*out_ell.n_rows + i] = col;
                //std::cout << "write ellv " << numell*out_hyb.n_rows + i << " " << S.valuePtr()[j] << std::endl;
                numell++;
            }
            else
            {
                //std::cout << "csr " << j -start << std::endl;
                crsvalues[i2] = S.valuePtr()[j];
                cols[i2] = col;
                i2++;
            }

        }

        // fill some parts of ell with zeros

        for(int j = numell; j < numellrows; j++)
        {
            ellvalues[j*out_ell.n_rows + i] = 0.0;
            ellcols[j*out_ell.n_rows + i] = 0;
            //std::cout << "write ell " << j*out_ell.n_rows + i << " " << 0.0 << std::endl;
        }

    }

    // change csr row pointer to remainingcsr
    int csrdiff = 0;
    for(int i = 0; i < out_ell.n_rows ;i++)
    {
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        //std::cout << i << " " << end - start << std::endl;
        int rest = end - start - numellrows;
        row_ptr[i] = csrdiff;

        csrdiff += std::max(0,rest);
    }
    row_ptr[out_ell.n_rows] = csrdiff;

    out_ell.n_csr = remainingcsr;

    out_ell.n_maxNNZperBlock = 0;
    for(int i = 0; i < out_ell.n_rows; i+=THREADS_PER_BLOCK)
    {
        int nd;
        if(i + THREADS_PER_BLOCK < out_ell.n_rows)
        {
            nd = row_ptr[i + THREADS_PER_BLOCK] - row_ptr[i];
        }
        else
        {
            nd =  row_ptr[out_ell.n_rows] - row_ptr[i];
        }
        if(nd > out_ell.n_maxNNZperBlock)
            out_ell.n_maxNNZperBlock = nd;
    }

//    int min = 1e6;
//    int max = 0;
//    for(int i = 0; i < out_ell.n_rows; i++)
//    {
//        int nd = row_ptr[i +1] - row_ptr[i];
//        if(nd < min)
//            min = nd;
//        if(nd > max)
//            max = nd;
//        //std::cout << nd << std::endl;
//    }
//    std::cout << "min: " << min << " max: " << max << std::endl;

    checkCudaErrors(cudaMalloc((void**)&out_ell.ellvalues, 	(out_ell.n_rows*numellrows)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_ell.ellcols, 		(out_ell.n_rows*numellrows)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&out_ell.crsvalues, 	(out_ell.n_csr)*sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_ell.cols, 		(out_ell.n_csr)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&out_ell.row_ptr,(out_ell.n_rows + 1)*sizeof(int)));

    checkCudaErrors(cudaMemcpy(out_ell.ellvalues, ellvalues,	(out_ell.n_rows*numellrows)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_ell.ellcols, ellcols,				(out_ell.n_rows*numellrows)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_ell.crsvalues, crsvalues,	(out_ell.n_csr)*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_ell.cols, cols,				(out_ell.n_csr)*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(out_ell.row_ptr, row_ptr,			(out_ell.n_rows + 1)*sizeof(int), cudaMemcpyHostToDevice));

    delete[] ellvalues;
    delete[] ellcols;
    delete[] crsvalues;
    delete[] cols;
    delete[] row_ptr;
}

void PDsolverCuda::deleteCRSMatrix(CRSMatrix& crs)
{
    if(!crs.device)
    {
        if(crs.row_ptr != NULL)
        {
            delete[] crs.row_ptr;
            crs.row_ptr = NULL;
        }
        if(crs.cols != NULL)
        {
            delete[] crs.cols;
            crs.cols = NULL;
        }
        if(crs.values != NULL)
        {
            delete[] crs.values;
            crs.values = NULL;
        }
    }
    else
    {
        checkCudaErrors(cudaFree((void*)crs.values));
        checkCudaErrors(cudaFree((void*)crs.cols));
        checkCudaErrors(cudaFree((void*)crs.row_ptr));
    }
}

void PDsolverCuda::deleteHYBMatrix(HYBMatrix& hyb)
{
    if(!hyb.device)
    {
        if(hyb.row_ptr != NULL)
        {
            delete[] hyb.row_ptr;
            hyb.row_ptr = NULL;
        }
        if(hyb.cols != NULL)
        {
            delete[] hyb.cols;
            hyb.cols = NULL;
        }
        if(hyb.diagvalues != NULL)
        {
            delete[] hyb.diagvalues;
            hyb.diagvalues = NULL;
        }
        if(hyb.crsvalues != NULL)
        {
            delete[] hyb.crsvalues;
            hyb.crsvalues = NULL;
        }
        if(hyb.ellvalues != NULL)
        {
            delete[] hyb.ellvalues;
            hyb.ellvalues = NULL;
        }
        if(hyb.ellcols != NULL)
        {
            delete[] hyb.ellcols;
            hyb.ellcols = NULL;
        }
    }
    else
    {
        checkCudaErrors(cudaFree((void*)hyb.diagvalues));
        checkCudaErrors(cudaFree((void*)hyb.row_ptr));
        checkCudaErrors(cudaFree((void*)hyb.cols));
        checkCudaErrors(cudaFree((void*)hyb.crsvalues));
        checkCudaErrors(cudaFree((void*)hyb.ellvalues));
        checkCudaErrors(cudaFree((void*)hyb.ellcols));
    }
}

void PDsolverCuda::deleteELLMatrix(ELLMatrix& hyb)
{
    if(!hyb.device)
    {
        if(hyb.row_ptr != NULL)
        {
            delete[] hyb.row_ptr;
            hyb.row_ptr = NULL;
        }
        if(hyb.cols != NULL)
        {
            delete[] hyb.cols;
            hyb.cols = NULL;
        }
        if(hyb.crsvalues != NULL)
        {
            delete[] hyb.crsvalues;
            hyb.crsvalues = NULL;
        }
        if(hyb.ellvalues != NULL)
        {
            delete[] hyb.ellvalues;
            hyb.ellvalues = NULL;
        }
        if(hyb.ellcols != NULL)
        {
            delete[] hyb.ellcols;
            hyb.ellcols = NULL;
        }
    }
    else
    {
        checkCudaErrors(cudaFree((void*)hyb.row_ptr));
        checkCudaErrors(cudaFree((void*)hyb.cols));
        checkCudaErrors(cudaFree((void*)hyb.crsvalues));
        checkCudaErrors(cudaFree((void*)hyb.ellvalues));
        checkCudaErrors(cudaFree((void*)hyb.ellcols));
    }
}

}
