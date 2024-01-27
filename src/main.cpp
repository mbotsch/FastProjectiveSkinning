//=============================================================================
// Copyright (C) 2019 The FastProjectiveSkinning developers
//
// This file is part of the Fast Projective Skinning Project.
// Distributed under a GPL license, see LICENSE.txt for details.
//=============================================================================

#include "Viewer_preparation.h"
#include "Viewer_skinning.h"

//=============================================================================

int main(int argc, char **argv)
{
    std::string skin_file_lr, skel_file, skin_file_hr, us_file, ini_filename;
    if(argc < 2)
    {
        std::cerr << "Fast Projective Skinning needs at least\n"
                  << "an .off file containing the skin you want to simulate!\n" << std::endl;
        return 2;
    }
    else
    {
        // if you want to rebuild your skeleton
        bool rebuild = false;
        std::string input(argv[1]);
        for(int i = 1; i < argc; i++)
        {
            input = argv[i];
            if(input == "--rebuild")
            {
                rebuild = true;
            }
            if(input.find(".ini") != input.npos)
            {
                ini_filename = input;
            }
            if(input.find(".skel") != input.npos)
            {
                skel_file = input;
            }
            std::cout << input << std::endl;
        }

        if(argc == 2 || rebuild)
        {
            input = argv[1];
            if(input.find(".off") != input.npos)
            {
                // start skeleton builder
                skin_file_lr = input; // todo: convert obj to off with pmp?

                Preparation_Viewer skel_builder("Skeleton Builder", 800, 600);
                skel_builder.skel_filename_ = skel_file;
                skel_builder.load_mesh(skin_file_lr.c_str());
                std::cout << std::endl;

                // build
                skel_builder.run();

                if(!skel_builder.ini_filename_.empty())
                    ini_filename = skel_builder.ini_filename_;
                else
                    return 2;
            }
        }
        else if(ini_filename.empty())
        {
            // parse all files seperately
            skin_file_lr = std::string(argv[1]);
            skel_file = std::string(argv[2]);
            if(skin_file_lr.find(".off") == skin_file_lr.npos)
            {
                std::cerr << "Error: " << skin_file_lr << " is no .off file." << std::endl;
            }
            if(skel_file.find(".skel") == skel_file.npos || skel_file.find(".skel2") == skel_file.npos)
            {
                std::cerr << "Error: " << skel_file << " is no .skel file." << std::endl;
            }

            if(argc == 5)
            {
                skin_file_hr = std::string(argv[3]);
                if(skin_file_hr.find(".off") == skin_file_hr.npos)
                {
                    std::cerr << "Error: " << skin_file_hr << " is no .off file." << std::endl;
                }

                us_file = std::string(argv[4]);
                if(us_file.find(".txt") == us_file.npos || us_file.find(".binus") == us_file.npos)
                {
                    std::cerr << "Error: " << us_file << " is no upsampling file." << std::endl;
                }
            }
        }
    }


  Skinning_Viewer viewer("Fast Projective Skinning", 800, 600);
  viewer.init(ini_filename, skin_file_lr.c_str(),skel_file.c_str(), skin_file_hr.c_str(), us_file.c_str());
  return viewer.run();

}

//=============================================================================
