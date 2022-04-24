# Basic Autonomous Driving Framework
This documentation describes how an end-to-end neural network for autonomous driving is
implemented -- use ROS2 nodes to receive sensor data from LGSVL Simulator,
apply a deep learning neural network to train the model, use the trained model to send
control commands using ROS2 nodes back to LGSVL Simulator, and drive a car autonomously. This framework
is tested to run in Ubuntu(Linux). It is assumed that you have some experience in using
Linux, Git, ROS, Docker, web browsers, and Python.

This project is the modified version of [ROS2 End-to-End Lane Following Model with LGSVL
Simulator](https://www.lgsvlsimulator.com/docs/lane-following/) which is in turn inspired by [NVIDIA's
End-to-End Deep Learning Model for Self-Driving
Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/).

## Complete End-to-End Architecture
![here](https://github.com/dr563105/basicad_framework/blob/main/autonomous_driving_framework_architecture.png)

At the highest-level, the architecture consists of four modules: sensor module, data collection,
pre-processing and training module, and finally evaluation module. 

## Features
* Training mode: Manually drive the vehicle and collect data
*  Autonomous Mode: The vehicle drives itself based on Lane Following model trained from the collected data
* ROS2-based
    * Time synchronous data collection node
    * Deploying a trained model in a node
* Data preprocessing for training
    * Data normalization and manipulation
    * Splitting data into training set and test set
    * Writing/Reading data in HDF5 format
* Deep Learning model training: Train a model using Keras with TensorFlow backend

### Prerequisites
* Docker CE
* Nvidia-docker
* GPU(at least 8GB)
* ROS2. This project uses ROS2 Dashing.
* Tensorflow, Keras
* Python3
* LGSVL Simulator


#### Clone the repository:
In the terminal at appropriate location run this command - 
```
git clone https://github.com/dr563105/basicad_framework.git
```
#### Pull the latest Docker image
Then in the terminal run this command -
```
docker pull lgsvl/lanefollowing:latest
```
#### Run the simulator
* Download the zipped modified simulator(filename: NewSimLFMay201.tar.gz) from [my personal google
drive](https://drive.google.com/file/d/14uIQBWeLkpzo5qYN1Thi1ka_OnYPITTC/view?usp=sharing). 
* Untar it and place the folder(NewSimLFMay201) along with the other folders from the repository.
* Navigate to `NewSimLFMay201->simulator->build`.
* Run simulator executable.
* Click `Open Browser`
* On the first login, create an account. Login. It should automatically download the
  default maps, vehicle, and simulation configurations.
* Navigate to `Vehicles`tab and create a new setup by clicking `Add new` button.
* Name the vehicle and use
  [this](https://assets.lgsvlsimulator.com/4eb6f2f8c293b00c4fed413a844cf3e4ffe7015d/vehicle_Jaguar2015XE)
  as vehicle url. Click `Submit`.
* Then click the wrench icon adjacent to the newly created vehicle profile. Make the
  `Bridge Type` as ROS2. Copy the contents from [here](https://github.com/dr563105/basicad_framework/blob/main/sensor_parameters.json) to the `Sensors` field. Click
  `Submit`. Add/remove as necessary. Please note that the camera sensors transmit images
  in 1920x1080 resolution. So they are huge and can bottleneck port bandwidth.
* Move to the `Simulations` tab, click `Add new` button and a popup menu will appear. In
  this menu's `general` tab, give a simulation name. 
* In the `Maps & Vehicles` tab, check `interactive mode`. Choose `SanFrancisco` as map.
* Next in the `Select Vehicles`, select the vehicles tab configuration name from the dropdown and write `localhost:9090` as IP address.
Other tabs can be left empty for the moment. Click `Submit`. 
* Select the configured simulation and click `play` button. If there are no errors, the
  simulation should execute and in the simulation application we can see a vehicle spawned
  in SF map. Click the `play` button to make the LGSVL simulator ready.

#### Build ROS2 packages
* Go to ROS2 workspace `testoutLF->lanefollowing->ros2_ws`.
* Execute the build command in the terminal
```
docker-compose up build_ros
```
* You should see two more folders -- build and log created. 

#### Running the collect script
While in the `ros2_ws`, run this command in the terminal
```
docker-compose up collect
```
* You should see the `rosbridge` is now connected. This can be verified by going to the
  settings in the LGSVL simulator app.
* The script inside `collect_script` is copied into the `~/ros2_ws/lane_following/collect.py`. The topic names of
  the nodes must match the topics mentioned in the vehicles tab of LGSVL WebUI.
* To exit, simply use `ctrl+c` in the terminal.
 
#### Evaluation
* To evaluate, the necessary evaluation script present inside `evaluation_script` must be
copied into `~/ros2_ws/lane_following/drive.py`.
* The trained models and their architectures are available inside `models_images_and_files` folder in
the repo. 
* Take the `*.h5` hdf5 file and place it inside the folder `ros2_ws/src/lane_following/model/`. It contains the trained models accessed by
the evaluation script. 
* The evaluation is run by executing this command in the terminal
```
docker-compose up drive_visual
```
* A small window will pop up. Now, go back to LGSVL simulator application and gently
  accelerate using the `Up` arrow key. The vehicle will then autonomously navigate. Toggle
  traffic and change weather conditions as necessary. 
* To exit, simply use `ctrl+c` in the terminal.

#### Pre-processing and Training
* For pre-processing, the scripts present inside `preprocessing_script` are used. For this
  step,  a data folder is required. Because of data collected is of large size, this is
  not provided.
* Similarly, for training, scripts inside `training_script` are used. 
* For convenience and CUDA dependencies, both the scripts are advised to be executed
  outside the docker. 

### Useful Links/Tips
* Robotic Operating System[(ROS)](https://www.ros.org/about-ros/)
* [LGSVL simulator](https://www.lgsvlsimulator.com/)  
* [Docker CE](https://docs.docker.com/engine/install/ubuntu/) and also its post installation.
* [steps](https://docs.docker.com/engine/install/linux-postinstall/)
* [Nvidia docker](https://github.com/NVIDIA/nvidia-docker)
* Use [Conda](https://www.anaconda.com/products/individual) environment for training.
  Conda automatically installs Tensorflow and CUDA dependencies and saves the headache of
  choosing which CUDA version is compatible with a particular version of tensorflow. 
* Always set ROS2 environment when running ROS related commands. `source
  /opt/ros/dashing/setup.bash` and from the ros2ws ` source install/setup.bash` for local
  setup.
* Thesis report folder contains the pdf of the report.
* Further instructions can be found [here](https://www.lgsvlsimulator.com/docs/create-ros2-ad-stack/) 
and [here](https://www.lgsvlsimulator.com/docs/lane-following/).
