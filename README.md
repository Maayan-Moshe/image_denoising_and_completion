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