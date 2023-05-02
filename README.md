# CFIL: Coupled Flow Imitation Learning
This repository contains the source code for "A Coupled Flow Approach to Imitation Learning": https://arxiv.org/abs/2305.00303.

Accepted at the International Conference on Machine Learning (ICML) 2023.

# Environment Creation
A full description for reproducing our exact environment:

note, a few lines may seem superfluous, but this will help avoid mujoco-py errors.
1. Download mujoco200 http://www.roboti.us/download.html
    1. wget https://www.roboti.us/download/mujoco200_linux.zip 
    2. mkdir ~/.mujoco
    3. unzip mujoco200_linux.zip -d ~/.mujoco;
	2. add mjkey.txt to the ~/.mujoco
		1. A free universal key is now available here: http://www.roboti.us/license.html (until october 18 2031)
	3. add path in .bashrc 
2. conda create -n cfil python=3.7.6
3. conda activate cfil
4. pip install gym==0.15.7
5. I had errors simply pip installing mujoco-py
6. What worked instead was:
	1. git clone https://github.com/openai/mujoco-py
		2. cd mujoco-py
		3. pip install -r requirements.txt
		4. pip install -r requirements.dev.txt
		5. pip install mujoco-py==2.0.2.8
7. mujoco_py is often the main cause for issues so do an intermediate test:
	1. cd ..
		1. cd out of mujoco-py folder, otherwise test won't import correct mujoco-py of 2.0.2.8
    2. python 
    3. import mujoco_py
        1. Should not err!
8. conda install openmpi-mpicc
9. cd CFIL/spinningup
10. pip install -e .
11. conda install -c conda-forge mpi4py
12. pip install protobuf==3.20.0
13. pip install --upgrade numpy
	1. Ignore the requirements error.
14. DONE!
	1. Now you may run the bash scripts. Note, they currently assume a cluster of 4 gpu's.
	2. For plotting the results use the jupyter notebook
		1. First
			1. conda install -c anaconda jupyter

# Third-Party Code Credits

Some of the code above is from the following repositories:

- [spinningup](https://github.com/openai/spinningup): We use their implementation of SAC.
- [normalizing_flows](https://github.com/kamenbliznashki/normalizing_flows):  We use their implementation of MAF.
