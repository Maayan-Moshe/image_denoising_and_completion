# Image denoising and completion neural network with tensor flow [![GitHub license](https://img.shields.io/github/license/nea/MarkdownViewerPlusPlus.svg)](https://github.com/nea/MarkdownViewerPlusPlus/blob/master/LICENSE.md) [![GitHub (pre-)release](https://img.shields.io/badge/release-0.8.2-yellow.svg)](https://github.com/nea/MarkdownViewerPlusPlus/releases/tag/0.8.2) [![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.me/insanitydesign)

This is an example of neural network for image denoising and completion. 
The network architecture is novel but inspired by existing convolution neural networks (CNN) for image segmentation. 
Other sources of inspiration for the architecture are res-Net and Multigrid from numerical analysis.

The network architecture is configurable via a configuration file.

The project also contains a general framework for training and visualizing any type of CNN.

## Features
* Configuration file to allow control on the training and some of the architecture. 
Also allows control on the data paths and training parameters (such as rate, optimizer, etc.).
* Module to get prediction on already trained network to visualize results.

## Prerequisites
For my development I used windows with python 3.6 and numpy and TensorFlow. 
However the code should be compatible to any operating system and python version as long as TensorFlow version is above 1.6.
* Python 3.6.
* TensorFlow 1.8.0.
* numpy 1.4.
* To visualize results you need matplotlib 2.2.2.
* Project folder should be added to Python Path.

## Configuration

### Training configuration
The configuration file is named "training_configuration.json". Selected fields which are included by the configuration file:
* train_params
	* initial_learning_rate - Each time the validation error increases the training rate 
	reduce by factor of two until it reaches the minimum training rate.
	* minimum_learning_rate
	* state_fname - The network can be initilized from existing state, if state fname in None then it uses default initilization.
	* state_folder - The folder in which the network parameters is saved to.
	* summaries_dir - Here the summeries results like histograms and graph are stored (for TensorBoard).
	* saved_state_fname - not including the date and selected network parameters.
	* num_iterations
	* summary_name
	* validation_rate_step - The validation error is checked once in rate steps.
* graph_params
	* module_path - Which grapg it will prepare the module must contain a function called prepare_graph.
	* cost - Which cost function to use.
	* loss_producer - Responsible for summing the residuals.		
* train_data_params
	* file_path - The file containing the data
	* batch_size
	* feeder - We can feed the data in different ways.
* validation_data_params
	* file_path
	* sample_size
	* feeder
	
### Results configuration
The configuration file is named "results_configuration.json". Selected fields which are included by the configuration file:
* state_params
	* state_fname - The file which contains the network data.
	* state_folder
* graph_params
	* module_path - different graphs for different usages are contained in different modules.
	* image_shape
	* max_z_mm - Maximum value of predicted pixel mostly 256.
	* min_z_mm - Minimum value of predicted pixel mostly 0.
* data_params
	* file_path - The file containing the data
	* sample_size
	* feeder - How we read the data from the file.

## Results
By running the file variable_reader.py with the state file "real_data_multiscale_26_May_2018_06_48.ckpt" we produced sample results for this trained network.
Now we ran the file results_plotter.py to visualize the results.

Two examples are presented below, corrupted images of numbers 4 and 9 were entered to the network and we manage to retrieve images very similar to the original images.
Where we corrupted the images only with pepper noise (black pixels) of randomly selected ~30% of the pixels. The cost is L2 distance between the predicted image to the true one.

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_0.png "")

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_1.png "")

If we corrupt the images with salt pepper noise (with ~51% corrrupted images) and use L2 distance 
we get the following results. One can see the smear of the images compared to the original images.

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_salt_pepper_l2_0.png "")

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_salt_pepper_l2_1.png "")

If we corrupt the images with salt pepper noise (with ~51% corrrupted images) and use L1 distance 
we get the following results. One can see less smear of the images compared to the original images but not as good shape reconstruction.

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_salt_pepper_l1_0.png "")

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/results_salt_pepper_l1_1.png "")


## Graph structure
The current graph that is defiend by the training and results configuration has the general structure displayed below.
The graph image was produced by TensorBoard using the command C:\WINDOWS\system32>tensorboard --logdir=C:\gitrep\image_denoising_and_completion\results\summaries_images\train, 
where the summaries are located.

![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/graph_general_structure.PNG "")

The general structure of the network is similar to [U net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), 
in the sense that we start with a full scale image (in our case 28x28xnum_channels) and then reduce it by convolutional layers and then expand the image again to its original size.
Each reduction reduce the image by a factor of two in the first and second dimensions and each expansion increases back the image by a factor of two in the first two dimensions.
The lower limit for reduction is 2 in the small dimension so for our case of 28 pixels original size we have 4 reduction\expansion layers.
Each reduction layer output is used for the next reduction layer input but also as an input for the same size expansion layer and also the twice larger expansion layer.

The general structure of a reduction layer is displayed below. 
As an input the layer takes as an input image from the layer below it and same size tensor in the first two dimensions but several channels in the channels dimensions.
Each 2x2 window in the input image we average by weights calculated by layer conv5 and conv6 in the example below.
The weights then are normalized so that the sum of the weights would be one if the sum of the weights is greater than one and is not normalized if the sum of weights in smaller than one.
After reducing the image by averaging each 2x2 window we add some additional image that is calculated by conv7, conv8 and conv9 in the example below.
The output of the layer is used as an input to the next reduction image but also to the expansion layers of the same size and twice its size (in each of the first two dimensions).
![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/reduction_layer.PNG "")

The general structure of an expansion layer is displayed below. 
As an input the layer takes as an input image from the layer below it and same size tensor in the first two dimensions but several channels in the channels dimensions.
Also it takes the output of the reduction layers of the same size and half the size.
To make the smaller images (from the previous layer and the smaller reduction layer) of the same size of the new layer we use a constant interpolation operator called expanding_data.
The weights of the interpolation operator are learned but they are constant for all the layers, to save parameters.
Then we average the images from the previous layers but with weights which are calculated seperatley for each pixel.
The final layer is addiotion layer similar to the one used in [res-Net](https://arxiv.org/abs/1512.03385).
![alt text](https://github.com/Maayan-Moshe/image_denoising_and_completion/blob/master/images/expansion_layer.PNG "")

