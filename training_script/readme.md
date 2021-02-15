The training script takes the preprocessed data from `~/basicad_framework/data/hdf5/ds3`,
loads them to memory, and splits them using Sklearn library into train and test
dataset(80:20). 

The trained models are stored inside `~/basicad_framework/models` in hdf5 format. Use
these models to evaluate using LGSVL simulator. Copy them to
`~/basicad_framework/testoutLF/lanefollowing/ros2_ws/src/lane_following/model` and rename
as `model.h5`.
