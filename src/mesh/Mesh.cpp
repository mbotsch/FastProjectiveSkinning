//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Mesh.h"
#include <stdio.h>

namespace Projective_Skinning
{

Mesh::Mesh(bool sj)
    :use_sliding_joints_(sj),
      num_us_neighbors_(0),
      num_simulated_skin_(0),
      num_non_rigid_(0),
      num_rigid_(0),
      num_simple_rigid_(0),
      num_ignored_(0),
      num_sliding_(0),
      num_sliding_references_(0),
      base_rigid_(0),
      base_simple_rigid_(0),
      base_ignored_(0),
      base_sliding_(0),
      base_sliding_references_(0)
{}

bool Mesh::init(const char* skinfilename, const char* skel_filename,
                const bool shrink, const bool diffmasses, const float mass,
                const char* skin_hr_filename, const char *upsampling_filename)
{
    pmp::Timer t;
    t.start();

    // load skin
    if (!load_OFF( skinfilename, vertices_, skin_.sim_indices))
    {
        std::cerr << "Can't load skinmesh " << skinfilename << std::endl;
        return false;
    }
    orig_vertices_ = vertices_;
    skin_.num_vertices = vertices_.cols();
    skin_.all_indices = skin_.sim_indices;

    // load skeleton
   if(!skeleton_.init(skel_filename, vertices_))
    {
        std::cerr << "Error: could not read skeleton file: " << skel_filename << std::endl;
        return false;
    }

    // fill some arrays with ignored bones, edges, one-ring neighbors and compute bounding box
    set_ignored_bones();
    find_edges_and_neighbors();
    computeBB();

    if(shrink)
    {
        // setup some temporary variables
        IndexVector vertexIsIgnored;
        Mat3X sliding_joint_references;
        std::map<usint, IndexVector> sjIndices;

        // shrink skin to bones
        shrink_skin_to_bones(vertexIsIgnored);

        // set sliding joints either automatically or by a pre-defined set of joints
        if(use_sliding_joints_)
            init_sliding_joints(sliding_joint_references, sjIndices);

        // resort the vertex array, such that ignored vertices are at the end and non boundary condition vertices at the beginning
        create_sorted_vertices(vertexIsIgnored, sliding_joint_references, sjIndices);

        // fill mesh with tetrahedrons and compute collision tetrahedrons
        tetrahedralize();
    }

    // load high resolution visualization mesh
    load_HR(skin_hr_filename, upsampling_filename);

    // fill mass vector either by mass generated with tetvolume or uniform mass
    if(diffmasses)
        compute_masses(mass);
    else
        vertex_masses_.resize(skin_.num_vertices, 1.0);

    print_statistics();

    std::cout << "Mesh construction done... (" << t.stop() << ")" << std::endl;
    return true;
}

bool Mesh::load_OFF(const char* _filename, Mat3X &verts, IndexVector &inds)
{

    // parse the file
    std::ifstream ifs ( _filename );
    if ( !ifs )
    {
        return false;
    }
    else
    {
        unsigned int   nV, nF, dummy;
        unsigned int   i, idx;
        float          x, y, z;
        std::string    magic;


        // header: OFF #Vertice, #Faces, #Edges
        ifs >> magic;
        if ( magic != "OFF" )
        {
            std::cerr << "No OFF file\n";
            return false;
        }
        ifs >> nV >> nF >> dummy;
        verts.resize(3,nV);
        //orig_vertices_.resize(3,nV);

        // read vertices
        for ( i=0; i < nV && !ifs.eof(); ++i )
        {
            ifs >> x >> y >> z;
            verts.col(i) << x,y,z;
        }


        // faces
        for ( i=0; i<nF; ++i )
        {
            ifs >> nV;

            if ( nV == 3 )
            {
                ifs >> idx;
                inds.push_back ( idx );
                ifs >> idx;
                inds.push_back ( idx );
                ifs >> idx;
                inds.push_back ( idx );
            }
            else
            {
                for ( unsigned int j=0; j<nV; ++j )
                {
                    ifs >> dummy;
                }
                std::cerr << "Only triangular faces are supported\n";
            }
        }


        std::cout
        << "read "
        << _filename << ": "
        << verts.cols() << " vertices, "
        << ( inds.size() /3 ) << " triangles" << std::endl;
    }

    ifs.close();


    //copy vertices
    //orig_vertices_ = vertices_;

    return true;
}


bool Mesh::load_HR(const char* skin_hr_filename, const char *us_parameter_filename)
{
    //load high res mesh and compute hr upscaling parameters
    std::cout << "Loading high res mesh" << std::endl;

    if ( !load_OFF( skin_hr_filename, high_res_vertices_, high_res_indices_) )
    {
        std::cout << "Can't load high-res-skinmesh. Building without HR." << std::endl;
        use_high_res_ = false;
        return false;
    }
    else
    {
        high_res_face_normals_.resize(3,high_res_indices_.size()/3);
        high_res_vertex_normals_.resize(3,high_res_vertices_.cols());
        high_res_face_normals_.setZero();
        high_res_vertex_normals_.setZero();
        compute_normals_HR();
    }

    use_high_res_ = load_upsampling_file(us_parameter_filename);

    if(use_high_res_)
        upsample();
    else
    {
        high_res_indices_.clear();
    }
    return use_high_res_;

}

bool Mesh::load_upsampling_file(const char* filename) // todo look at this function
{
    //std::string path = ANIMATION_DATA_PATH + std::string ( filename );
    std::ifstream ifs ( filename );
    if ( !ifs )
    {
        std::cerr << "Opening " << filename << " failed. Maybe wrong filename?\n";
        return false;
    }

    usint numV, numN;
    ifs >> numV >> numN;
    if(numV != high_res_vertices_.cols())
    {
        std::cerr << "Wrong number of US parameters: " << numV << " should be " << high_res_vertices_.cols() << std::endl;
        return false;
    }

    us_neighors_.resize(numV);
    us_Nij_.resize(numV);
    us_normal_Nij_.resize(numV);
    num_us_neighbors_ = numN;

    for(usint i = 0; i < numV; i++)
    {
        std::vector<usint> nb;
        std::vector<float> Nijs, NNijs;
        usint n, vi;
        ifs >> vi >> n;
        if(n != numN)
        {
            std::cerr << "One or more vertices have a different number of neighbors! " << std::endl;
        }
        nb.resize(n);
        Nijs.resize(n);
        NNijs.resize(n);
        float sum = 0;
        for(usint j = 0; j < n; j++)
        {
            usint ni;
            float Nij, NNij;

            ifs >> ni >> Nij >> NNij;

            nb[j] = ni;
            Nijs[j] = Nij;
            NNijs[j] = NNij;
            sum+=Nij;
        }

        // check partition of unity
        for(usint j = 0; j < n; j++)
        {
            Nijs[j]/=sum;
        } // todo make this optional...

        us_neighors_[i] = nb;
        us_Nij_[i] = Nijs;
        us_normal_Nij_[i] = NNijs;
    }

    if(old_to_new_.empty())
    {
        std::cerr << "no old to new order available!!" << std::endl;
        return false;
    }
    // resort neighbor indices
    for(auto &nb : us_neighors_)
    {
        for(auto &n : nb)
        {
            n = old_to_new_[n];

            // corresponding to ignored vertex
            if(n >= skin_.num_vertices)
            {
                // recompute ignored index
                n = n - skin_.num_vertices + base_rigid_ + num_rigid_;
            }
        }
    }

    std::cout << "Read Nij from " << filename << " " << numN << " neighbors per vertex." << std::endl;
    return true;
}


void Mesh::computeBB ()
{
    // bounding box
    if(!(vertices_.cols() == 0))
    {
        bbMin_(0) = vertices_.row(0).minCoeff();
        bbMin_(1) = vertices_.row(1).minCoeff();
        bbMin_(2) = vertices_.row(2).minCoeff();

        bbMax_(0) = vertices_.row(0).maxCoeff();
        bbMax_(1) = vertices_.row(1).maxCoeff();
        bbMax_(2) = vertices_.row(2).maxCoeff();
    }
}


void Mesh::set_shrink_pairs_to_nearest(Mat3X& shrinkpairs, IndexVector &vertexIsIgnored)
{
    shrinkpairs.resize(3,skin_.num_vertices);

    for(unsigned int i = 0; i< skin_.num_vertices; i++)
    {
        if(vertexIsIgnored[i]) continue;

        Vec3 v = vertices_.col(i);

        Vec3 minSP(0,0,0);
        float mindist = FLT_MAX;
        for(unsigned int b = 0; b< skeleton_.bone_indices_.size(); b+=2)
        {
            Vec3 s0 = skeleton_.joint_positions_.col(skeleton_.bone_indices_[b]);
            Vec3 s1 = skeleton_.joint_positions_.col(skeleton_.bone_indices_[b + 1]);

            Vec3 nearest = getNearestLinePoint(v,s0,s1);
            float dist = (nearest - v).squaredNorm();
            if(dist < mindist)
            {
                mindist = dist;
                minSP = nearest;
                skin_.assoc_bone[i] = b/2;
            }
        }
        shrinkpairs.col(i) = minSP;
    }
}


void Mesh::tetrahedralize()
{
    init_simulation_tets();
    init_collision_tets();
}

#include <omp.h>

void Mesh::update_anchors()
{
    if(vertices_.cols() > skin_.num_vertices)
    {

        for(auto aa : additional_anchors_)
        {
            vertices_.col(aa) = skeleton_.transformations_[additional_anchor_bones_[aa] + 1]*orig_vertices_.col(aa);
        }


        #pragma omp parallel for
        for(int i = base_simple_rigid_ - (int)skin_.num_vertices; i < (int)skin_.num_vertices + num_ignored_ + num_sliding_references_; i++)
        {
            usint vi = i + skin_.num_vertices;
            vertices_.col(vi) = skeleton_.transformations_[transform_indices_[i]] * orig_vertices_.col(vi);
        }

        if(use_sliding_joints_)
        {
            skin_sliding();
        }
    }
}

void Mesh::transform_collision_tet_basepoints_()
{
    if(vertices_.cols() > skin_.num_vertices)
    {
        #pragma omp parallel for
        for(int i = 0; i < collision_tet_basepoints_.cols(); i++)
        {
            collision_tet_basepoints_.col(i) = skeleton_.transformations_[coltet_transform_indices_[i]]*orig_collision_tet_basepoints_.col(i);
        }
    }
}

void Mesh::reset(bool withshrinkV)
{
    skeleton_.reset();
    update_anchors();

    if(withshrinkV)
        transform_collision_tet_basepoints_();

    vertices_ = orig_vertices_;

    compute_normals();
}

void Mesh::store_OFF(const char* off_filename, bool high_res)
{
    //save OFF
    std::ofstream ofs ( off_filename );
    if ( !ofs )
    {
        std::cerr << "Opening " << off_filename << " failed. Maybe wrong filename?\n";
        return;
    }

    Mat3X &vertices = high_res ? high_res_vertices_ : vertices_;
    IndexVector &indices = high_res ? high_res_indices_ : skin_.all_indices;

    ofs << "OFF\n";
    //basic information
    ofs << vertices.cols() << " " << indices.size()/3  << " 0" << "\n";

    //skinvertices
    ofs << vertices.transpose();
    ofs << "\n";

    //triangles
    for(unsigned int i = 0; i < indices.size();i+=3)
    {
        ofs << "3 " << indices[i] << " " << indices[i + 1] << " " << indices[i + 2] << "\n";
    }

    ofs.close();
}

void Mesh::find_ignored_vertices(IndexVector &vertexIsIgnored)
{
    vertexIsIgnored.resize(skin_.num_vertices,0);
    for(usint i = 0; i < skin_.num_vertices; i++)
    {
        Vec3 v = vertices_.col(i);
        usint bone = 0;
        float min = FLT_MAX;
        for(usint b = 0; b < skeleton_.bone_indices_.size(); b+=2)
        {
            usint b0 = skeleton_.bone_indices_[b];
            usint b1 = skeleton_.bone_indices_[b + 1];

            Vec3 vb0 = skeleton_.joint_positions_.col(b0);
            Vec3 vb1 = skeleton_.joint_positions_.col(b1);

            Vec3 x = vb1 - vb0;
            float L = x.norm();
            x/=L;

            float Lnew = x.dot(v - vb0);
            Lnew = (Lnew > L)? L : Lnew;
            Lnew = (Lnew < 0)? 0 : Lnew;
            Vec3 pnear = vb0 + Lnew*x;

            float distsq = (pnear - v).dot(pnear - v);
            if(distsq < min)
            {
                min = distsq;
                bone = b/2;
            }
        }

        if(ignored_bones_.find(bone) != ignored_bones_.end())
        {
            vertexIsIgnored[i] = bone;
            if(skin_.assoc_bone.size() > 0)
                skin_.assoc_bone[i] = bone;
        }
    }


    // delete outer most layer from ignored
    for(int k = 0; k < 2; k++)
    {
        IndexVector toUnset;
        for(usint e = 0; e < skin_.edges.size(); e+=2)
        {
            usint e0 = skin_.edges[e];
            usint e1 = skin_.edges[e + 1];
            if(!vertexIsIgnored[e0] && vertexIsIgnored[e1])
            {
                toUnset.push_back(e1);
            }

            if(!vertexIsIgnored[e1] && vertexIsIgnored[e0])
            {
                toUnset.push_back(e0);
            }
        }
        for(auto ni : toUnset)
            vertexIsIgnored[ni] = 0;
    }


    // find ignored neighbors
    for(usint e = 0; e < skin_.edges.size(); e+=2)
    {
        usint e0 = skin_.edges[e];
        usint e1 = skin_.edges[e + 1];
        if(!vertexIsIgnored[e0] && vertexIsIgnored[e1] && (ignored_neighbors_.find(e0) == ignored_neighbors_.end()))
        {
            ignored_neighbors_.insert(e0);
        }

        if(!vertexIsIgnored[e1] && vertexIsIgnored[e0] && (ignored_neighbors_.find(e1) == ignored_neighbors_.end()))
        {
            ignored_neighbors_.insert(e1);
        }
    }

    for(auto in : ignored_neighbors_)
    {
        bool hasNonIgnoredNeighbor = false;
        usint bone = 0;
        for(usint f = 0; f < skin_.sim_indices.size(); f+=3)
        {
            usint fi[3];
            bool adjTrig = false;
            for(usint i = 0; i < 3; i++)
            {
                fi[i] = skin_.sim_indices[f+i];
                if(fi[i] == in)
                    adjTrig = true;
            }
            if(adjTrig)
            {
                bool trigIgnored = false;
                for(usint i = 0; i < 3; i++)
                {
                    if(vertexIsIgnored[fi[i]])
                    {
                        trigIgnored = true;
                        bone = vertexIsIgnored[fi[i]];
                    }
                }
                if(!trigIgnored)
                {
                    hasNonIgnoredNeighbor = true;
                }
            }

        }
        if(!hasNonIgnoredNeighbor)
        {
            vertexIsIgnored[in] = bone;
        }
    }

    // add small non ignored clusters (inner mouth)
    IndexSet restIN = ignored_neighbors_;
    while(restIN.size() > 0)
    {
        IndexSet cluster;
        usint vstart = *(restIN.begin());
        cluster.insert(vstart);
        restIN.erase(restIN.begin());

        add_non_ignored_neighbors_to_cluster(cluster, restIN, vstart, vertexIsIgnored);

        if(cluster.size() < 0.1*skin_.num_vertices)
        {
            // get bone
            int bone = -1;
            for(auto i : cluster)
            {
                for(auto n : skin_.neighbors_[i])
                {
                    if(vertexIsIgnored[n])
                    {
                        bone = vertexIsIgnored[n];
                        break;
                    }
                }
                if(bone > -1)
                    break;
            }

            if(bone > -1)
            {
                for(auto i : cluster)
                {
                    vertexIsIgnored[i] = bone;
                    auto iti = ignored_neighbors_.find(i);
                    if(iti != ignored_neighbors_.end())
                        ignored_neighbors_.erase(iti);
                }
            }
            else
            {
                std::cerr << "\n\nPROBLEM!!! No associated bone found for ignored vertex!!!\n" << std::endl;
            }
        }
    }
}

void Mesh::add_non_ignored_neighbors_to_cluster(IndexSet &cluster, IndexSet &remaining, usint vstart, IndexVector &vertexIsIgnored)
{
    IndexSet added;
    for(auto n : skin_.neighbors_[vstart])
    {
        if(cluster.find(n) == cluster.end() && !vertexIsIgnored[n])
        {
            cluster.insert(n);
            added.insert(n);

            auto it = remaining.find(n);
            if(it != remaining.end())
            {
                remaining.erase(it);
            }
        }
    }

    for(auto a : added)
    {
        add_non_ignored_neighbors_to_cluster(cluster, remaining, a, vertexIsIgnored);
    }
}


void Mesh::setup_normals(const int nV, const IndexVector &indices)
{
    face_normals_.resize(3,indices.size()/3);
    vertex_normals_.resize(3, nV);

    neighbor_faces_.clear();
    neighbor_faces_.resize(nV);

    for(unsigned int j = 0; j < indices.size(); j++)
    {
        neighbor_faces_[indices[j]].push_back(j/3);
    }
    compute_normals();
}

void Mesh::print_statistics()
{
    std::cout << "\nSimulation Mesh Statistics\n" <<
                 "Triangles:            " << skin_.all_indices.size()/3 << "\n" <<
                 "Tetrahedra:           " << tets_.indices.size()/4 << "\n" <<
                 "Ignored Anchors:      " << additional_anchors_.size() << "\n" <<
                 "Collision Tetrahedra: " << tets_.collision_indices.size()/4 << "\n" << std::endl;

    std::cout << "Vertices:            " << vertices_.cols() << "\t[" << 0 << ":" << vertices_.cols() << ")\n" <<
                 "- skin vertices:     " << skin_.num_vertices << "\t[" << 0 << ":" << skin_.num_vertices << ")\n" <<
                 "- nonrigid vertices: " << num_non_rigid_ << "\t[" << 0 << ":" << num_non_rigid_ << ")\n"<<
                 "- sliding vertices:  " << num_sliding_ << "\t[" << base_sliding_ << ":" << base_sliding_ + num_sliding_ << ")\n"<<
                 "- simple rigid v.:   " << num_simple_rigid_ << "\t[" << base_simple_rigid_ << ":" << base_simple_rigid_ + num_simple_rigid_ << ")\n" <<
                 "- rigid vertices:    " << num_rigid_ << "\t[" << base_rigid_ << ":" << base_rigid_ + num_rigid_ << ")\n" <<
                 "- ignored vertices:  " << num_ignored_ << "\t[" << base_ignored_ << ":" << base_ignored_ + num_ignored_ << ")\n" <<
                 "- sj ref vertices:   " << num_sliding_references_ << "\t[" << base_sliding_references_ << ":" << base_sliding_references_ + num_sliding_references_ << ")\n" << std::endl;


    std::cout << "Visualisation Mesh Statistics\n" <<
                 "Triangles:            " << high_res_indices_.size()/3 << "\n" <<
                 "Vertices:             " << high_res_vertices_.cols() << "\n" <<
                 "Upsampling Neighbors: " << num_us_neighbors_ << "\n" << std::endl;
}


void Mesh::shrink_skin_to_bones(IndexVector &vertexIsIgnored)
{

    std::cout << "\nShrinking skin to skeleton " << std::flush;
    pmp::Timer t;
    t.start();

    std::vector<bool> isIn(skin_.num_vertices, false);


    skin_.assoc_bone.resize(skin_.num_vertices);
    find_ignored_vertices(vertexIsIgnored);


    for(auto n : ignored_neighbors_)
    {
        Vec3 v = vertices_.col(n);
        float min = FLT_MAX;
        usint bone = 0;
        for(auto bi : ignored_bones_)
        {
            usint b0 = skeleton_.bone_indices_[2*bi];
            usint b1 = skeleton_.bone_indices_[2*bi + 1];

            Vec3 vb0 = skeleton_.joint_positions_.col(b0);
            Vec3 vb1 = skeleton_.joint_positions_.col(b1);

            Vec3 x = vb1 - vb0;
            float L = x.norm();
            x/=L;

            float Lnew = x.dot(v - vb0);
            Lnew = (Lnew > L)? L : Lnew;
            Lnew = (Lnew < 0)? 0 : Lnew;
            Vec3 pnear = vb0 + Lnew*x;

            float distsq = (pnear - v).dot(pnear - v);
            if(distsq < min)
            {
                min = distsq;
                bone = bi;
            }
        }

        usint j = skeleton_.bone_indices_[2*bone];
        Vec3 J = skeleton_.joint_positions_.col(j);
        float r = skeleton_.get_joint_radius_from_stickindex(j);
        vertices_.col(n) = J + r*((v - J).normalized());
        isIn[n] = true;
    }

    skin_.all_indices = skin_.sim_indices;

    Mat3X shrinkV;
    set_shrink_pairs_to_nearest(shrinkV, vertexIsIgnored);
    shrink_pair_smoothing(shrinkV, vertexIsIgnored);
    move_to_shrinkpairs(shrinkV, vertexIsIgnored);

    t.stop();
    std::cout << "Done (" << t.elapsed() << "ms)\n" << std::endl; //, Error: " << error/(float)skin.nSkinVertices<<")"<< std::endl;

    //merge vertices and shrinked vertices in vertices_
    Mat3X mergedVertices;
    unsigned int nV = vertices_.cols();
    mergedVertices.resize(3,nV*2);
    mergedVertices.block(0,0,3,nV) = orig_vertices_;
    mergedVertices.block(0,nV,3,nV) = vertices_;
    orig_vertices_.resize(3,nV*2);
    vertices_.resize(3,nV*2);
    vertices_ = orig_vertices_ = mergedVertices;
}

void Mesh::find_edges_and_neighbors()
{
    //clear to make sure that nothing bad happens when called twice
    skin_.edges.clear();
    skin_.neighbors_.clear();
    skin_.neighbors_2_ring.clear();

    // first edges and avg edgelength
    float avg = 0;
    std::set<std::pair<usint,usint>> edges;

    for(unsigned int f = 0; f< skin_.sim_indices.size(); f+=3)
    {
        unsigned int idx[] = {skin_.sim_indices[f], skin_.sim_indices[f + 1],skin_.sim_indices[f + 2]};

        for(int i = 0; i < 3; i++)
        {
            std::pair<usint,usint> edge_pair = std::make_pair(std::min(idx[i] , idx[(i+1)%3]), std::max(idx[i] , idx[(i+1)%3]));
            if(edges.find(edge_pair) == edges.end())
            {
                edges.insert(edge_pair);
                skin_.edges.push_back(edge_pair.first);
                skin_.edges.push_back(edge_pair.second);

                avg += (vertices_.col(edge_pair.first)-vertices_.col(edge_pair.second)).norm();
            }
        }
    }
    skin_.avg_edge_length = avg/(float)edges.size();

    skin_.neighbors_.resize(skin_.num_vertices);
    for(auto &p : edges)
    {
        skin_.neighbors_[p.first].insert(p.second);
        skin_.neighbors_[p.second].insert(p.first);
    }

    // to test our mesh, we now look for lonely vertices, that are not part of any edge (resp. triangle)
    for(usint i = 0; i < skin_.num_vertices; i++)
    {
        if(skin_.neighbors_[i].empty())
            std::cerr << "\nProblem: Lonely vertex found: " << i << "\n\n" << std::endl;
    }

    // now 2-ring neighbors
    for(usint i = 0; i < skin_.num_vertices; i++)
    {
        IndexSet ns = skin_.neighbors_[i];

        for(auto n : skin_.neighbors_[i]) // 1 - ring
            for(auto nn : skin_.neighbors_[n]) // 2 - ring
                if(ns.find(nn) == ns.end()) ns.insert(nn);

        skin_.neighbors_2_ring.push_back(ns);
    }
}

void Mesh::shrink_pair_smoothing(Mat3X& shrinkV, IndexVector &vertexIsIgnored)
{
    float error = 10;
    Mat3X laplace;
    laplace.resize(3,shrinkV.cols());
    int its = 0;

    //find vertices, that are either at leaf joints or bend joints and keep them fixed
    std::vector<bool> notMove(shrinkV.cols(),false);

    std::vector<Joint*> relevant_joints_;
    for(auto jo : skeleton_.joints_)
    {
        if(jo->childreen_.empty() || (jo->is_root_ && jo->childreen_.size() == 1))
            relevant_joints_.push_back(jo);

        if(skeleton_.is_character_mesh_ &&  (jo->name_ == "l_elbow"   || jo->name_ == "r_elbow"  || jo->name_ == "r_knee"   || jo->name_ == "l_knee"))
            relevant_joints_.push_back(jo);
    }

    for(usint i = 0; i < shrinkV.cols(); i++)
    {
        if(vertexIsIgnored[i]) continue;

        Vec3 v = shrinkV.col(i);

        for(auto jo : relevant_joints_)
        {

            Vec3 x = v  - jo->position_;
            float r = skeleton_.get_joint_radius_from_stickindex(jo->index_);
            if((x).dot(x) < r*r)
            {
                notMove[i] = true; break;
            }
        }
    }
    for(auto i : ignored_neighbors_)
    {
        notMove[i] = true;
    }

    //stretch out bones for a better laplacemovement (baggy pants fix)
    VecX angles;
    Mat3X axis;
    stretch_out(shrinkV, angles, axis);


    float minerror = std::max(1e-5f, (bbMin_ - bbMax_).norm()/25000.0f);
    while(error > minerror && its < 2000)
    {

        error = 0;
        laplace.setZero();

        #pragma omp parallel for
        for(int i = 0; i < shrinkV.cols(); i++)
        {

            if(vertexIsIgnored[i]) continue;

            Vec3 v = shrinkV.col(i);

            if(!notMove[i])
            {
                IndexSet n = skin_.neighbors_[i];
                float norm = 0;
                for(auto ni : n)
                {
                    float edgeLengthSkin = 1.0/((vertices_.col(ni) - vertices_.col(i)).norm());
                    norm+= edgeLengthSkin;
                    laplace.col(i) += edgeLengthSkin*(shrinkV.col(ni) - v);
                }
                laplace.col(i)/=norm;
            }
        }


        #pragma omp parallel for reduction(+:error)
        for(int i = 0; i < shrinkV.cols(); i++)
        {
            if(vertexIsIgnored[i]) continue;

            if(!notMove[i])
            {
                Vec3 before = shrinkV.col(i);
                shrinkV.col(i) += 0.4*laplace.col(i);

                usint bone = 0;
                shrinkV.col(i) = skeleton_.get_projection_on_boneline(shrinkV.col(i), bone);
                skin_.assoc_bone[i] = bone;
                error += (before - shrinkV.col(i)).dot(before - shrinkV.col(i));
            }
        }
        its++;
        if(its%20 == 0)
        std::cout << ". " << std::flush;
    }

    if(its >= 2000)
        std::cerr << "Shrink-Pair-Move not converged." << std::endl;

    for(usint i = 0; i < shrinkV.cols(); i++)
    {
        if(vertexIsIgnored[i]) continue;

        if(ignored_bones_.find(skin_.assoc_bone[i]) != ignored_bones_.end())
        {
            vertexIsIgnored[i] = skin_.assoc_bone[i];
        }
    }

    // retransform (de-stretch-out) shrinkv
    for(usint i = 0; i < shrinkV.cols(); i++)
    {
        usint co = skin_.assoc_bone[i];
        Eigen::Affine3f transform = skeleton_.transformations_[co + 1].inverse();
        shrinkV.col(i) = transform*shrinkV.col(i);
    }

    //retransform skeleton
    skeleton_.reset();

    //save boneshrinkV
    orig_collision_tet_basepoints_ = collision_tet_basepoints_ = shrinkV;
}

void Mesh::move_to_shrinkpairs(Mat3X& shrinkV, IndexVector &vertexIsIgnored)
{
    for(usint i = 0; i < vertices_.cols(); i++)
    {
        if(ignored_neighbors_.find(i) == ignored_neighbors_.end())
            vertices_.col(i) = orig_vertices_.col(i);
    }

    shrinking_correspondences_.resize(skin_.num_vertices, std::make_pair(Skeleton::Correspondence::NONE,-1));
    transform_indices_.resize(skin_.num_vertices,-1);
    for(usint i = 0; i < shrinkV.cols(); i++)
    {
        if(vertexIsIgnored[i]) {transform_indices_[i] = vertexIsIgnored[i] + 1; continue;}

        Vec3 v = vertices_.col(i);
        Vec3 dest = shrinkV.col(i);

        vertices_.col(i) = skeleton_.get_skeleton_line_intersection(v,dest, shrinking_correspondences_[i]);// todo directly store transform

        if(shrinking_correspondences_[i].first == Skeleton::Correspondence::BONE)
            transform_indices_[i] = shrinking_correspondences_[i].second + 1;
        if(shrinking_correspondences_[i].first == Skeleton::Correspondence::JOINT)
            transform_indices_[i] = shrinking_correspondences_[i].second + skeleton_.joint_positions_.cols();
    }


}

void Mesh::find_edgedirections(std::map<std::pair<usint, usint>, bool> &edge_directions)
{
    for(usint t = 0; t < skin_.sim_indices.size(); t+=3)
    {
        usint t0 = skin_.sim_indices[t];
        usint t1 = skin_.sim_indices[t + 1];
        usint t2 = skin_.sim_indices[t + 2];

        // store edge and counteredge
        std::pair<usint, usint> e[3], ec[3];
        e[0] = std::make_pair(t0,t1);  ec[0] = std::make_pair(t1,t0);
        e[1] = std::make_pair(t1,t2);  ec[1] = std::make_pair(t2,t1);
        e[2] = std::make_pair(t2,t0);  ec[2] = std::make_pair(t0,t2);

        // init randomly
        for(int i = 0; i < 3; i++)
        {
            if(edge_directions.find(e[i]) == edge_directions.end())
            {
                edge_directions[e[i]] = (bool)((t/3 + i)%2);
                edge_directions[ec[i]] = !edge_directions[e[i]];
            }
        }
    }

    int n_problems = 10; int iterations = 0;
    // as long as we have problematic cases
    while(n_problems > 0 && iterations < 1000)
    {
        iterations++;
        n_problems = 0;
        for(usint t = 0; t < skin_.sim_indices.size(); t+=3)
        {
            usint t0 = skin_.sim_indices[t];
            usint t1 = skin_.sim_indices[t + 1];
            usint t2 = skin_.sim_indices[t + 2];

            // store edge and counteredge
            std::pair<usint, usint> e[3], ec[3];
            e[0] = std::make_pair(t0,t1);  ec[0] = std::make_pair(t1,t0);
            e[1] = std::make_pair(t1,t2);  ec[1] = std::make_pair(t2,t1);
            e[2] = std::make_pair(t2,t0);  ec[2] = std::make_pair(t0,t2);

            // if all point clockwise or counterclockwise
            if(edge_directions[e[0]] == edge_directions[e[1]] && edge_directions[e[0]] == edge_directions[e[2]])
            {
                // flip one randomly
                int rand_0_to_2 = std::rand()%3;
                edge_directions[e[rand_0_to_2]] = !edge_directions[e[rand_0_to_2]];
                edge_directions[ec[rand_0_to_2]] = !edge_directions[ec[rand_0_to_2]];

                n_problems++;
            }
        }
    }

    if(n_problems > 0)
    {
        std::cerr << "Problem with sorting tetrahedra: Not converged in 1000 steps, still " << n_problems << " problems!" << std::endl;
    }
}

void Mesh::init_simulation_tets()
{
    std::map<std::pair<usint, usint>, bool> edgeDirectionFS_;
    find_edgedirections(edgeDirectionFS_);

    tets_.indices.resize(0);
    for(usint t = 0; t < skin_.sim_indices.size(); t+=3)
    {
        usint t0 = skin_.sim_indices[t];
        usint t1 = skin_.sim_indices[t + 1];
        usint t2 = skin_.sim_indices[t + 2];

        std::pair<usint, usint> e[3];
        e[0] = std::make_pair(t0,t1);
        e[1] = std::make_pair(t1,t2);
        e[2] = std::make_pair(t2,t0);

        usint nV = skin_.num_vertices;

        int doubleedge = -1;
        for(usint i = 0; i < 3; i++)
        {
            usint i1 = (i+1)%3;
            if(edgeDirectionFS_[e[i]] && !edgeDirectionFS_[e[i1]])
            {
                doubleedge = i1;
            }
        }

        // resort indices that the one with double true directions is first
        usint inds[3];
        inds[0] = e[(0 + doubleedge)%3].first;
        inds[1] = e[(1 + doubleedge)%3].first;
        inds[2] = e[(2 + doubleedge)%3].first;
        bool opposite_edge_direction = edgeDirectionFS_[e[(1 + doubleedge)%3]];

        //1st tet
        tets_.indices.push_back(inds[0]);
        tets_.indices.push_back(inds[1] + nV);
        tets_.indices.push_back(inds[0] + nV);
        tets_.indices.push_back(inds[2] + nV);
        //2nd tet
        tets_.indices.push_back(inds[0]);
        tets_.indices.push_back(inds[1]);
        tets_.indices.push_back(inds[1] + nV);
        if(opposite_edge_direction)
        {
            tets_.indices.push_back(inds[2]);
            //3rd tet
            tets_.indices.push_back(inds[1] + nV);
        }
        else
        {
            tets_.indices.push_back(inds[2] + nV);
            //3rd tet
            tets_.indices.push_back(inds[1]);
        }
        tets_.indices.push_back(inds[2] + nV);
        tets_.indices.push_back(inds[2]);
        tets_.indices.push_back(inds[0]);
    }

    // correct degenerated tets by moving one vertex slightly
    for(usint t3 = 0; t3 < tets_.indices.size(); t3+=12)
    {
        float smallestV = FLT_MAX;
        float biggestV = -1;
        int smallestT = -1;
        for(int t = 0; t < 12; t+= 4)
        {
            Mat33 edges;
            for(int j = 0; j < 3; j++) edges.col(j) = vertices_.col(tets_.indices[t3 + t + j + 1]) - vertices_.col(tets_.indices[t3 + t]);

            float vol = fabs(edges.determinant());
            if(vol < smallestV)
            {
                smallestV = vol;
                smallestT = t;
            }

            if(vol > biggestV)
            {
                biggestV = vol;
            }
        }

        if(smallestV/biggestV < 1e-2)
        {
            //find out how many shrinkV we have
            bool svs[] ={false,false,false,false};
            Vec3 ps[4];
            for(int i = 0; i < 4; i++)
            {
                ps[i] = vertices_.col(tets_.indices[t3 + smallestT + i]);
                svs[i] = static_cast<int>(tets_.indices[t3 + smallestT + i]) >= num_simulated_skin_;
            }

            // find biggest face
            float maxA = -1;
            int maxLast = -1;
            Vec3 maxn;
            for(int f = 0; f < 4; f++)
            {
                Vec3 n = (ps[(f + 1)%4] - ps[f]).cross(ps[(f + 2)%4] - ps[f]);
                float A = fabs(n.norm());
                int last = (f + 3)%4;
                if(svs[last] && A > maxA)
                {
                    maxA = A;
                    maxLast = last;
                    maxn = n.normalized();
                }
            }

            // move vertex slightly to get a better volume
            if(maxLast != -1)
            {
                float targetVol = biggestV/10.0;
                Vec3 target = ps[maxLast] + maxn*targetVol/maxA; // note: factors 0.5 (area), 1/3 (volume formula), 1/6 (determinant) cancel out
                vertices_.col(tets_.indices[t3 + smallestT + maxLast]) = target;
                orig_vertices_.col(tets_.indices[t3 + smallestT + maxLast]) = target;
            }
            else
            {
                std::cerr << "Problem: Not able to find correction to Tet" << std::endl;
            }
        }
    }
}

void Mesh::compute_normals()
{
    vertex_normals_.setZero();
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (int)skin_.all_indices.size(); i+=3)
        {
            int i0 = skin_.all_indices[i];
            int i1 = skin_.all_indices[i + 1];
            int i2 = skin_.all_indices[i + 2];

            const Vec3& p0 = vertices_.col(i0);
            const Vec3& p1 = vertices_.col(i1);
            const Vec3& p2 = vertices_.col(i2);

            face_normals_.col(i/3) = ((p1-p0).cross(p2-p0)).normalized();
        }


        #pragma omp for
        for(int i = 0; i < vertex_normals_.cols(); i++)
        {
            for(auto n : neighbor_faces_[i])
            {
                vertex_normals_.col(i) += face_normals_.col(n);
            }
        }
    }

    vertex_normals_.leftCols(skin_.num_vertices).colwise().normalize();
    vertex_normals_.rightCols(num_ignored_).colwise().normalize();
}

void Mesh::compute_normals_HR()
{

    std::vector<unsigned int>::const_iterator  idx_it, idx_end(high_res_indices_.end());
    unsigned int i, i0, i1, i2;


    // calculate face normals, accumulate vertex normals
    for (i=0, idx_it=high_res_indices_.begin(); idx_it!=idx_end; ++i)
    {
        i0 = *idx_it++;
        i1 = *idx_it++;
        i2 = *idx_it++;

        const Vec3& p0 = high_res_vertices_.col(i0);
        const Vec3& p1 = high_res_vertices_.col(i1);
        const Vec3& p2 = high_res_vertices_.col(i2);

        // compute angle weights
// 		const Vec3 e01 = (p1-p0).normalized();
// 		const Vec3 e12 = (p2-p1).normalized();
// 		const Vec3 e20 = (p0-p2).normalized();
// 		const double w0 = acos( std::max(-1.0f, std::min(1.0f, e01.dot(-e20) )));
// 		const double w1 = acos( std::max(-1.0f, std::min(1.0f, e12.dot(-e01) )));
// 		const double w2 = acos( std::max(-1.0f, std::min(1.0f, e20.dot(-e12) )));

        high_res_face_normals_.col(i) = ((p1-p0).cross(p2-p0)).normalized();

        high_res_vertex_normals_.col(i0) += high_res_face_normals_.col(i);//*w0;
        high_res_vertex_normals_.col(i1) += high_res_face_normals_.col(i);//*w1;
        high_res_vertex_normals_.col(i2) += high_res_face_normals_.col(i);//*w2;
    }


    // normalize vertex normals
    high_res_vertex_normals_.colwise().normalize();

//    #pragma omp parallel for
//    for (unsigned int n = 0; n < high_res_vertex_normals_.cols(); n++)
    //        high_res_vertex_normals_.col(n).normalize();
}

void Mesh::adjust_tet_weights_(std::vector<float> &weight_factors)
{
    // dedimensionalization has to be the inverse scaling factor 1/s
    // in A you find Edge^-1 that is scaled by 1/s because of inverse
    // all tetconstraints are scaled by sqrt(s^3)
    // that means tetconstaints in S are scaled by (sqrt(s^3)/s)^2 = s
    // to compensate that -> scale tetweight weight by 1/s
    Vec3 bbdiff = bbMax_ - bbMin_;
    float bbVol = bbdiff(0)*bbdiff(1)*bbdiff(2);
    float dedimensionalizationfactor = pow(1000.0f/bbVol,1.0f/3.0f);

    // compute tetweightfactors multiplied by tetweight and sqrt(tetvolume) later for fingers, we want to have higher weights
    weight_factors.clear();
    weight_factors.resize(num_simulated_skin_, dedimensionalizationfactor);
    for(int i = 0; i < num_simulated_skin_; i++)
    {
        float maxweight = 8.0;
        Vec3 v = vertices_.col(i);
        usint ab = skin_.assoc_bone[i];
        std::string name = skeleton_.name_by_joint_[ab + 1];
        if((name.find("thumb") != name.npos || name.find("wrist") != name.npos ||
                name.find("bow") != name.npos || name.find("middle") != name.npos || name.find("index") != name.npos || name.find("pinky") != name.npos || name.find("ring") != name.npos))// && !(name.find("distal") != name.npos))
        {
            // left
            Joint* jo = skeleton_.joints_[skeleton_.joint_by_name_["l_wrist"]];
            if(name.find("r_") != name.npos)
            {
                //right
                jo = skeleton_.joints_[skeleton_.joint_by_name_["r_wrist"]];
            }
            Vec3 s0 = jo->position_;
            Vec3 s1 = jo->parent_->position_;
            Vec3 s2 = jo->parent_->parent_->position_;
            Vec3 x = (s0 - s1).normalized();

            float negative_distmax = (s2 - s0).dot(x);
            float dist = (v - s0).dot(x);
            if(dist > 0)
              weight_factors[i] = maxweight*dedimensionalizationfactor;
            else
            {
              weight_factors[i] = dedimensionalizationfactor * std::min(maxweight,std::max(1.0f,1.0f - (dist - negative_distmax)/negative_distmax*maxweight));
            }
        }
    }
}

void Mesh::init_collision_tets()
{
    float avgVol = 0;
    for(usint t = 0; t < tets_.indices.size(); t+=4)
    {
        Mat34 tv;
        for(usint i = 0; i < 4; i++)
        {
            //if vertex on shrinked skin
            usint ti = tets_.indices[t + i];
            tv.col(i) = vertices_.col(ti);
        }
        Mat33 edges;
        edges.col(0) = tv.col(1) - tv.col(0);
        edges.col(1) = tv.col(2) - tv.col(0);
        edges.col(2) = tv.col(3) - tv.col(0);
        float V = fabs(edges.determinant());
        avgVol += V;
    }
    avgVol/= (float)(tets_.indices.size()/4);

    for(usint t = 0; t < tets_.indices.size(); t+=4)
    {
        Mat34 tv;
        for(usint i = 0; i < 4; i++)
        {
            //if vertex on shrinked skin
            usint ti = tets_.indices[t + i];
            if(ti >= skin_.num_vertices)
            {
                //take shrinkV instead
                tv.col(i) = collision_tet_basepoints_.col(ti - skin_.num_vertices);
            }
            else
            {
                tv.col(i) = vertices_.col(ti);
            }
        }
        Mat33 edges;
        edges.col(0) = tv.col(1) - tv.col(0);
        edges.col(1) = tv.col(2) - tv.col(0);
        edges.col(2) = tv.col(3) - tv.col(0);
        float V = fabs(edges.determinant());
        if(V > avgVol/100.0)
        {
            tets_.collision_indices.push_back(tets_.indices[t + 0]);
            tets_.collision_indices.push_back(tets_.indices[t + 1]);
            tets_.collision_indices.push_back(tets_.indices[t + 2]);
            tets_.collision_indices.push_back(tets_.indices[t + 3]);

            tets_.collision_tet_to_trig.push_back(t/4/3);
        }
    }
}

void Mesh::upsample()
{
    #pragma omp parallel for
    for(int i = 0; i < high_res_vertices_.cols(); i++)
    {
        Vec3 v = Vec3(0,0,0);
        Vec3 n = Vec3(0,0,0);
        for(int j = 0; j < num_us_neighbors_; j++)
        {
            v += us_Nij_[i][j]*(vertices_.col(us_neighors_[i][j]));// - orig_vertices_.col(us_neighors_[i][j]));
            n += us_normal_Nij_[i][j]*vertex_normals_.col(us_neighors_[i][j]);
        }
        high_res_vertices_.col(i) = v;// + orig_hrv_.col(i);
        high_res_vertex_normals_.col(i) = n.normalized();
    }

    if(!use_new_us_normals_)
        compute_normals_HR();
}

void Mesh::init_sliding_joints(Mat3X &sj_refs, std::map<usint, IndexVector> &sjIndices)
{
    typedef Skeleton::Correspondence COR;
    const float sliding_joint_bone_factor = 0.4;
    skeleton_.find_sliding_joints();

    //setup matrix containing sj reference point (attached to vertices_ later)
    sj_refs.resize(3,30*skeleton_.sliding_joints_.size());
    sj_refs.setZero();

    sj_parameters_.resize(skin_.num_vertices);

    int ctr = 0;

    for(auto SJ : skeleton_.sliding_joints_)
    {
        int j = SJ.j;
        int b0 = SJ.b0;
        int b1 = SJ.b1;

        const Vec3 &J = skeleton_.joints_[j]->position_;
        //store jointvertex s, and vectors from joint in bonedirections
        Vec3 l[] = {(skeleton_.joint_positions_.col(SJ.j0)-J),(skeleton_.joint_positions_.col(SJ.j1)-J)};

        Vec3 B0 = J + sliding_joint_bone_factor*l[0];
        Vec3 B1 = J + sliding_joint_bone_factor*l[1];
        l[0].normalize(); l[1].normalize();

        // find all vertices that are on the sliding joint or the adjoining bones in the range of sliding_joint_bone_factor
        for(usint i = 0; i < skin_.num_vertices; i++)
        {
            auto cor = shrinking_correspondences_[i];

            if( ((cor.first == COR::JOINT || cor.first == COR::INTER) && (cor.second == SJ.jointindex))   ||
                 (cor.first == COR::BONE                              && (cor.second == b0 || cor.second == b1))  )
            {
                // test if vertex is in between planes
                Vec3 v = vertices_.col(i + skin_.num_vertices);
                float dot0 = (v - B0).dot(l[0]);
                float dot1 = (v - B1).dot(l[1]);
                if(dot0 < 0 && dot1 < 0)
                {
                    sjIndices[j].push_back(i);
                }
            }
        }

        // determine boneangle via dotproduct
        l[0].normalize();l[1].normalize();
        float dot = l[0].dot(l[1]);

        // find vector that is perpendicular to plane spanned by bonelines
        // bent bone -> choose vector in plane spanned by l[0] and l[1]
        // straight joint boneline --> choose perpendicular vector by switching coeffs
        Vec3 np = (dot + 1 > 1e-4) ?
                    l[0].cross(l[1]) :
                  (fabs(l[0](0)) >= 1e-4 || fabs(l[0](1)) >= 1e-4) ? Vec3(-l[0](1),l[0](0),0) : Vec3(-l[0](2),0,l[0](0));


        Vec3 n[] = {(l[0].cross(np)).normalized() , (l[1].cross(-np)).normalized()};
        Vec3 nj = (n[0] + n[1]).normalized();

        int refID = ctr*30;
        //compute 10 reference points via circulating around bones and joint
        for(usint k = 0; k < 10; k++)
        {
            float kangle = float(k)/5.0*M_PI;
            Mat33 R;

            R = Eigen::AngleAxis<float>(kangle, l[0]);
            sj_refs.col(refID + k) = B0 + skeleton_.vol_bones_[SJ.b0].radius_*R*n[0];

            R = Eigen::AngleAxis<float>(kangle, -l[1]);
            sj_refs.col(refID + 10 + k) = B1 + skeleton_.vol_bones_[SJ.b1].radius_*R*n[1];

            R = Eigen::AngleAxis<float>(kangle, (l[0] - l[1]).normalized());
            sj_refs.col(refID + 20 + k) = J + skeleton_.vol_joints_[SJ.jointindex].radius_*R*nj;
        }

        //determine interpolation parameters alpha and h
        for(usint i = 0; i < sjIndices[j].size(); i++)
        {
            usint vi = sjIndices[j][i];
            Vec3 v = vertices_.col(vi + skin_.num_vertices);

            int i_zero_one = ((l[0] - l[1]).dot(v - J) > 0) ? 0 : 1;

            // compute nearest point to v one boneline l[0]
            Vec3 nearest = J + l[i_zero_one].dot(v -J)*l[i_zero_one];

            // vector frome nearest to v
            Vec3 a = (v - nearest).normalized();
            // compute angle between a and n[0] via dot product
            float angle = acos(std::max(-1.0f,std::min(a.dot(n[i_zero_one]),1.0f)));

            // decide on which halfcircle v is located and adapt angle if necessary
            float dotp = (a.cross(n[i_zero_one])).dot(l[i_zero_one]);
            if((dotp >= 0 && i_zero_one == 0) || (dotp < 0 && i_zero_one == 1))
                angle = 2*M_PI - angle;

            SJ_Interpolation_Parameters sjp;

            // the vertice's angle is between angleid and angleid +1 'st reference point
            int angleid = floor(angle/(2.0f*M_PI)*10.0f);

            // save reference points on b0, b1 with 10 --> 0
            sjp.i00 = refID + 10*i_zero_one + angleid%10;
            sjp.i01 = refID + 10*i_zero_one + (angleid + 1)%10;

            // save reference points on joint with 10 --> 0
            sjp.i10 = refID + 20 + angleid%10;
            sjp.i11 = refID + 20 + (angleid + 1)%10;

            // compute angle of both reference points, this time setting 2*M_PI == 0 would lead to wrong results
            float angle0 = (float)(angleid)/10.0f*2.0f*M_PI;
            float angle1 = (float)(angleid + 1)/10.0f*2.0f*M_PI;
            // determine angle interpolation parameter
            sjp.p_angle = (angle - angle0)/(angle1 - angle0);

            // interpolate reference points in angle direction
            Vec3 v0 = sjp.p_angle*sj_refs.col(sjp.i00) + (1.0f - sjp.p_angle)*sj_refs.col(sjp.i01);
            Vec3 v1 = sjp.p_angle*sj_refs.col(sjp.i10) + (1.0f - sjp.p_angle)*sj_refs.col(sjp.i11);

            // find point on line spanned by those interpolated reference points
            Vec3 p = getNearestLinePoint(v,v0,v1);

            // compute second interpolation parameter
            float h = (v0 - p).norm()/(v0 - v1).norm();
            sjp.p_h = h;

            // store all interpolation relevant information
            sj_parameters_[vi] = sjp;
        }
        ctr++;
    }
}

void Mesh::skin_sliding()
{

    #pragma omp parallel for
    for(int i = 0; i < num_sliding_; i++)
    {
        SJ_Interpolation_Parameters &sjp = sj_parameters_[i];

        float ang = sjp.p_angle;
        float h = sjp.p_h;
        float h0a0 = h*ang;
        float h1a0 = (1.0f - h)*ang;
        float h0a1 = h*(1.0f - ang);
        float h1a1 = (1.0f - h)*(1.0f - ang);

        vertices_.col(base_sliding_ + i) =
                h1a0*vertices_.col(base_sliding_references_ + sjp.i01) +
                h1a1*vertices_.col(base_sliding_references_ + sjp.i00) +
                h0a0*vertices_.col(base_sliding_references_ + sjp.i11) +
                h0a1*vertices_.col(base_sliding_references_ + sjp.i10);
    }

}

void Mesh::set_ignored_bones()
{
    if(skeleton_.is_character_mesh_)
    {
        //new version with names
        IndexSet ignJ;
        for(auto sn : skeleton_.joint_by_name_)
        {
            if(/*sn.first == "skullbase" ||*/
                sn.first.find("eye") != sn.first.npos/* ||
                ((sn.first.find("thumb") != sn.first.npos ||
                sn.first.find("index") != sn.first.npos ||
                sn.first.find("middle") != sn.first.npos ||
                sn.first.find("ring") != sn.first.npos ||
                sn.first.find("pinky") != sn.first.npos) && sn.first.find("1") == sn.first.npos)*/)
            {
                ignJ.insert(sn.second);
            }
        }

        for(usint i = 0; i < skeleton_.bone_indices_.size(); i+=2)
        {
            if(ignJ.find(skeleton_.bone_indices_[i + 1]) != ignJ.end())
            {
                ignored_bones_.insert(i/2);
            }
        }
    }
}

void Mesh::stretch_out(Mat3X &shrinkV, VecX &angles, Mat3X &axis)
{
    IndexVector pcounter(skeleton_.joint_positions_.cols(), 0);
    for(usint i = 0; i < skeleton_.bone_indices_.size(); i+=2)
    {
        pcounter[skeleton_.bone_indices_[i + 1]]++;
        pcounter[skeleton_.bone_indices_[i]]++;
    }

    angles.resize(skeleton_.joint_positions_.cols());
    angles.setZero();

    axis.resize(3, skeleton_.joint_positions_.cols());
    axis.setZero();
    axis.row(0).setConstant(1.0);

    for(unsigned int i = 1; i < skeleton_.joint_positions_.cols(); i++)
    {
        unsigned int p = skeleton_.joints_[i]->parent_->index_;
        if(pcounter[p] == 2 && p != 0)
        {
            unsigned int pp = skeleton_.joints_[p]->parent_->index_;
            Vec3 a0 = (skeleton_.joint_positions_.col(i) - skeleton_.joint_positions_.col(p)).normalized();
            Vec3 a1 = (skeleton_.joint_positions_.col(p) - skeleton_.joint_positions_.col(pp)).normalized();
            float dot = a1.dot(a0);
            if(fabs(dot) < 0.99)
            {
                axis.col(i) = (a0.cross(a1)).normalized();
                angles(i) = acos(dot);
            }

        }

    }

    skeleton_.transform(angles, axis);
    for(usint i = 0; i < shrinkV.cols(); i++)
    {
        usint co = skin_.assoc_bone[i];
        Eigen::Affine3f transform = skeleton_.transformations_[co + 1];
        shrinkV.col(i) = transform*shrinkV.col(i);
    }
}

void Mesh::create_sorted_vertices(IndexVector &vertexIsIgnored, Mat3X &sj_refs, std::map<usint, IndexVector> &sjIndices)
{

    IndexVector vertexOrder;
    std::vector<bool> alreadyIn(skin_.num_vertices, false);

    std::map<int,IndexVector> vbj_to_index;
    std::map<int,IndexVector> ign_to_index;

    for(usint i = 0; i < skin_.num_vertices; i++)
    {
        if(vertexIsIgnored[i])
        {
            usint assoc = (skin_.assoc_bone[i] + 1)*1000;
            ign_to_index[assoc].push_back(i);
        }
        else
        {
            auto cor = shrinking_correspondences_[i];
            if(cor.first == Skeleton::Correspondence::BONE)
            {
                vbj_to_index[cor.second].push_back(i);
            }
            else if(cor.first == Skeleton::Correspondence::JOINT)
            {
                vbj_to_index[500 + cor.second].push_back(i); // joints 500, 501, ...
            }
            else
            {
                vbj_to_index[1000 + cor.second].push_back(i); // interbonejoint 1000, 1001, ...; none 999
            }
        }
    }

    // first all nonBC shrunken skin vertices
    for(auto vbjIV : vbj_to_index)
    {
        if(vbjIV.first >= 1000)
        {
            for(auto i : vbjIV.second)
            {
                if(!alreadyIn[i])
                {
                    vertexOrder.push_back(i);
                    alreadyIn[i] = true;
                }
            }
        }
    }

    // set base to beginning of rigid vertices
    num_non_rigid_ = base_sliding_ = base_rigid_ = vertexOrder.size();
    // now all slidingjoint shrunken skin vertices
    for(usint j = 0; j < skeleton_.sliding_joints_.size(); j++)
    {
        usint jo = skeleton_.sliding_joints_[j].j;
        for(usint i = 0; i < sjIndices[jo].size(); i++)
        {
            usint ind = sjIndices[jo][i];
            if(!alreadyIn[ind])
            {
                vertexOrder.push_back(ind);
                alreadyIn[ind] = true;
            }
        }
    }
    num_sliding_ = vertexOrder.size() - base_sliding_;

    // now all non sj in vbj order (first all bones then all joints)
    for(auto vbjIV : vbj_to_index)
    {
        if(vbjIV.first < 1000)
        {
            for(auto i : vbjIV.second)
            {
                if(!alreadyIn[i])
                {
                    vertexOrder.push_back(i);
                    alreadyIn[i] = true;
                }
            }
        }
    }
    num_rigid_ = vertexOrder.size() - base_rigid_;
    num_non_rigid_ += vertexOrder.size();

    // finally all ignored in bone order
    base_ignored_ = vertexOrder.size();
    for(auto vbjIV : ign_to_index)
    {
        for(auto i : vbjIV.second)
        {
            vertexOrder.push_back(i);
            alreadyIn[i] = true;
        }
    }
    num_ignored_ = vertexOrder.size() - base_ignored_;

    if(vertexOrder.size() != skin_.num_vertices)
    {
        std::cerr << "\n\nunexpected behavior in resortVertices! NumVO: " << vertexOrder.size() << " NumV: " << skin_.num_vertices << "\n\n" << std::endl;
    }

    resort_data(vertexOrder, vertexIsIgnored);

    num_simulated_skin_ = skin_.num_vertices - num_ignored_;
    base_rigid_ += num_simulated_skin_;
    base_sliding_ = base_rigid_;
    base_simple_rigid_ = base_rigid_ + num_sliding_;
    num_simple_rigid_ = num_rigid_ - num_sliding_;
    base_ignored_ = base_rigid_ + num_rigid_;

    if(num_ignored_ > 0)
    {
        Mat3X simulated_vertices;
        simulated_vertices.resize(3,vertices_.cols() - 2*num_ignored_);
        simulated_vertices.leftCols(num_simulated_skin_) = vertices_.leftCols(num_simulated_skin_);
        simulated_vertices.rightCols(num_simulated_skin_) = vertices_.block(0,skin_.num_vertices,3,num_simulated_skin_);

        Mat3X ignored_vertices = vertices_.block(0, num_simulated_skin_, 3, num_ignored_);

        vertices_.resize(3,simulated_vertices.cols() + ignored_vertices.cols());
        vertices_.leftCols(simulated_vertices.cols()) = simulated_vertices;
        vertices_.rightCols(num_ignored_) = ignored_vertices;
        orig_vertices_ = vertices_;

        IndexVector rest_indices;
        IndexVector simulated_indices;
        simulated_indices.clear();
        for(usint f = 0; f < skin_.sim_indices.size(); f+=3)
        {
            int f0 = skin_.sim_indices[f];
            int f1 = skin_.sim_indices[f + 1];
            int f2 = skin_.sim_indices[f + 2];

            if(f0 < num_simulated_skin_ && f1 < num_simulated_skin_ && f2 < num_simulated_skin_)
            {
                simulated_indices.push_back(f0);
                simulated_indices.push_back(f1);
                simulated_indices.push_back(f2);
            }
            else
            {
                rest_indices.push_back((f0 < num_simulated_skin_) ? f0 : f0 + num_simulated_skin_);
                rest_indices.push_back((f1 < num_simulated_skin_) ? f1 : f1 + num_simulated_skin_);
                rest_indices.push_back((f2 < num_simulated_skin_) ? f2 : f2 + num_simulated_skin_);
            }
        }

        skin_.all_indices = simulated_indices;
        skin_.all_indices.insert(skin_.all_indices.end(), rest_indices.begin(), rest_indices.end());

        skin_.num_vertices = num_simulated_skin_;
        skin_.sim_indices = simulated_indices;
        vertex_normals_.resize(3,vertices_.cols());
        face_normals_.resize(3,skin_.all_indices.size()/3);

        collision_tet_basepoints_ = orig_collision_tet_basepoints_.leftCols(skin_.num_vertices);
        orig_collision_tet_basepoints_ = collision_tet_basepoints_;
    }
    else
    {
        skin_.all_indices = skin_.sim_indices;
    }

    if(use_sliding_joints_)
    {
        base_sliding_references_ = vertices_.cols();
        num_sliding_references_ = sj_refs.cols();
        vertices_.resize(3, orig_vertices_.cols() + sj_refs.cols());
        vertices_.leftCols(orig_vertices_.cols()) = orig_vertices_;
        vertices_.rightCols(sj_refs.cols()) = sj_refs;
        skin_sliding();
        orig_vertices_ = vertices_;

        for(int i = 0; i < sj_refs.cols()/30; i++)
        {
            int t[3] = {skeleton_.sliding_joints_[i].b0 + 1, skeleton_.sliding_joints_[i].b1 + 1, (int)skeleton_.joint_positions_.cols() + skeleton_.sliding_joints_[i].jointindex};
            for(int j = 0; j < 30; j++)
                transform_indices_.push_back(t[j/10]);
        }
    }

    // setup normals
    setup_normals(base_ignored_ + num_ignored_, skin_.all_indices);

    // prepair anchor constraints for the boundary of an ignored region
    if(num_ignored_ > 0)
    {
        additional_anchor_bones_.resize(skin_.num_vertices, 0);
        for(int i = 0; i < num_simulated_skin_; i++)
        {
            for(auto n: skin_.neighbors_[i])
            {
                if(vertexIsIgnored[n] && additional_anchors_.find(i) == additional_anchors_.end())
                {
                    additional_anchors_.insert(i);
                    additional_anchor_bones_[i] = vertexIsIgnored[n];
                    break;
                }
            }
        }
    }

    // change transformindices
    for(auto &i : transform_indices_)
    {
        if(i >= skeleton_.joint_positions_.cols())
        {
            i = skeleton_.vol_joints_[i - skeleton_.joint_positions_.cols()].stickIndex_ + skeleton_.joint_positions_.cols();
        }
        else if(i > 0)
            i = skeleton_.joints_[i]->parent_->index_;
    }
    coltet_transform_indices_.resize(collision_tet_basepoints_.cols());
    for(usint i = 0; i < coltet_transform_indices_.size(); i++)
    {
        coltet_transform_indices_[i] = skeleton_.joints_[skin_.assoc_bone[i] + 1]->parent_->index_;
    }

}

void Mesh::resort_data(IndexVector &vertexOrder, IndexVector &vertexIsIgnored)
{
    //compute old_to_new and resort vertices
    old_to_new_.resize(vertexOrder.size(),0);

    Mat3X vertices_no;
    vertices_no.resize(3,2*skin_.num_vertices);
    Mat3X ColTetShrinkV_no;
    ColTetShrinkV_no.resize(3,skin_.num_vertices);
    IndexVector vert_ig_no(vertexIsIgnored.size(),0);
    IndexVector assocB(skin_.num_vertices,0);
    std::vector<std::pair<Skeleton::Correspondence,int>> corres_no(shrinking_correspondences_.size(), std::make_pair(Skeleton::NONE, -1));
    std::vector<int> transform_no(transform_indices_.size(), -1);
    std::vector<SJ_Interpolation_Parameters> sjparam2(skin_.num_vertices);

    for(usint i = 0; i < skin_.num_vertices; i++)
    {
        usint nextVertex = vertexOrder[i];
        vertices_no.col(i) = vertices_.col(nextVertex);
        vertices_no.col(skin_.num_vertices + i) = vertices_.col(skin_.num_vertices + nextVertex);

        ColTetShrinkV_no.col(i) = orig_collision_tet_basepoints_.col(nextVertex);

        old_to_new_[nextVertex] = i;
        vert_ig_no[i] = vertexIsIgnored[nextVertex];
        assocB[i] = skin_.assoc_bone[nextVertex];
        corres_no[i] = shrinking_correspondences_[nextVertex];
        transform_no[i] = transform_indices_[nextVertex];
        if(use_sliding_joints_)
            sjparam2[i] = sj_parameters_[nextVertex];
    }
    vertices_ = vertices_no;
    orig_vertices_ = vertices_no;
    collision_tet_basepoints_ = ColTetShrinkV_no;
    orig_collision_tet_basepoints_ = ColTetShrinkV_no;
    vertexIsIgnored = vert_ig_no;
    skin_.assoc_bone = assocB;
    shrinking_correspondences_ = corres_no;
    transform_indices_ = transform_no;
    sj_parameters_ = sjparam2;

    IndexSet iNeighbors;
    for(auto in : ignored_neighbors_)
    {
        iNeighbors.insert(old_to_new_[in]);
    }
    ignored_neighbors_ = iNeighbors;

    //resort indices
    for(usint i = 0; i < skin_.sim_indices.size(); i++)
    {
        usint ind = skin_.sim_indices[i];
        skin_.sim_indices[i] = old_to_new_[ind];
    }

    sj_parameters_.resize(num_sliding_);
    for(int i = 0; i < num_sliding_; i++)
    {
        sj_parameters_[i] = sj_parameters_[i + base_sliding_];
    }

    find_edges_and_neighbors();
}

void Mesh::compute_masses(float totalmass)
{
    vertex_masses_.clear();
    vertex_masses_.resize(skin_.num_vertices,0.0);
    float sum = 0;

    for(usint t = 0; t < tets_.indices.size()/4; t++)
    {
        Mat33 edges;
        for(int i = 0; i < 3; i++)
        {
            edges.col(i) = vertices_.col(tets_.indices[4*t + 1 + i]) - vertices_.col(tets_.indices[4*t]);
        }
        float tetMass = std::abs(edges.determinant()/6.0);

        for(int i = 0 ; i < 4; i++)
        {
            usint ti = tets_.indices[4*t + i];
            if(ti < skin_.num_vertices)
            {
                vertex_masses_[ti] += tetMass/4.0;
                sum += tetMass/4.0;
            }
        }
    }

    for(auto &m :vertex_masses_)
    {
        m *= totalmass/sum;
    }

    // decrease mass of specific regions in character to make them less wobbely
    if(skeleton_.is_character_mesh_)
    {
        for(int i = 0; i < num_simulated_skin_; i++)
        {
            std::string name = skeleton_.name_by_joint_[skin_.assoc_bone[i]];
            if(name.find("ankle") != name.npos || name.find("skullbase") != name.npos || name.find("vc7") != name.npos)
            {
                vertex_masses_[i] *= 1e-3;
            }
        }
    }
}

}
