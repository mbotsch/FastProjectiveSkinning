# Fast Projective Skinning

This is an implementation of the [Fast Projective Skinning](https://graphics.uni-bielefeld.de/downloads/publications/2019/mig19.pdf) approach from Martin Komaritzan and Mario Botsch. To get started, just clone the
repository recursively:

    $ git clone --recursive https://github.com/mbotsch/FastProjectiveSkinning.git

## Dependencies

If you are just interested in the CPU-Version, simply build this project and run it. If you are interestend in the GPU-Version, you will have to install the CUDA libraries. The code was tested for CUDA versions 9.2 and 10.1 but others should also work. Try one of CUDA's examples to make sure it works properly.

## Configure and build:

    $ cd FastProjectiveSkinning && mkdir build && cd build && cmake .. && make

This will automatically build the project and its dependecies.

## Running

First, try one of our simple examples:
	
	$ ./skinning ../data/cylinder/cylinder.ini

By pressing '?' you can see the options for keyboard input. You can also use the GUI. By pressing 'space' and 'a' you start the simulation and a simple animation. For some models, we have multiple animations. Use keys 1-9 to select them. If no animation is available, this will animate the selected joint instead.

By holding ctrl + left mouse, you can select, drag and drop joints around. Additional options can be tuned in the `defines.h` file.


### Examples

We provide 4 examples:

1. A simple cylinder, that can be tested via:

    $ ./skinning ../data/cylinder/cylinder.ini

2. A simple T-shaped mesh, that can be tested via:

	$ ./skinning ../data/tbone/tbone.ini

3. The armadillo model from the [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) decimated to 3 percent of the original resolution for simulation and using the full resolution for visualization via upsampling. It can be tested via:

	$ ./skinning ../data/armadillo/armadillo.ini

4. A human male model from [free3D.com](https://free3d.com/3d-model/male-base-mesh-6682.html), slightly modified and decimated to 20 percent of the original resolution for simulation and using the full resolution for visualization via upsampling. It can be tested via:

	$ ./skinning ../data/maleChar/maleChar.ini


### .ini Files

.ini files are simple text files containing the path from executable directory to the needed data files. 5 Files are possible:

1. SIMMESH - path to the simulated mesh in .off format
2. SKELETON - path to the embedded skeleton in .skel format
3. VISMESH - (optional) path to a second mesh for visualisation in .off format. Needed, if you want to use the upsampling
4. UPSAMPLING - (optional) path to an .txt file conteining the upsampling weights. Needed, if you want to use the upsampling.
5. ANIMATION - (optional) path to an .anim file containing an animation of your mesh compatible to the skeleton followed by the filename without ending (used, if you want to load multiple animations)

instead of using an .ini file you can also use the files (in the same order as above) as command line arguments directly.

### Building your Skeleton (.skel files)

If you have just a mesh that you want so simulate, you can use our programm to build a skeleton file in a very simple way. Just give it the skin-mesh as argument, like:

	$ ./skinning ../data/cylinder/cylinder.off

Again, your options will be explained by pressing '?'. 

For a quick start: Holding ctrl + left click selects a jont, by clicking and dragging, you can move it around. Pressing 'j' will add a new joint to the selected. In that way, you can build a skeleton very simple and save it to a .skel file. Make sure, that the skeleton is inside of your mesh.

You can also directly create or edit the file with any text editor. Format is:

	[number of joints]
	x- y- z-location name_of_root_joint root
	x- y- z-location name_of_next_joint name_of_its_parent
	...

You can also edit an existing mesh by using the --rebuild argument, like

	$ ./skinning path/to/mesh.off path/to/skeleton.skel --rebuild

### Decimation and Upsampling

In the skeleton build application explained above you can also decimate your mesh and compute the upsampling weights. Those will be automatically stored to a file. You can also directly create an .ini file here.

If you already have a mesh and a skeleton but want to decimate the mesh, use

	$ ./skinning path/to/mesh.off path/to/skeleton.skel --rebuild

or

	$ ./skinning path/to/ini.ini --rebuild
 
We recommend using at least 20 Neighbors for upsampling and about 3000-5000 simulated vertices for good and fast results.

### Collisions

The collision handling can be toggled via the GUI. If you want to handle collisions, you should have a powerful CPU or use the GPU version. If you don't want to handle collisions at all, you can turn them off completely in the `defines.h` file. This will boost performance a bit. 

Collisions can cause crashs if not used carefully. This will be fixed in future.


## License

This code is available under [GPL](LICENSE).

In cases where the constraints of the Open Source license prevent you from using FastProjectiveSkinning, please contact us.


