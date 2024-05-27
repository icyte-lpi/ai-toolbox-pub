###############################################################################
#                        ICyTE - LPI - AI Toolbox                             #
# module name:                                                                #
#     net_visualization                                                       #
#                                                                             #
# module description:                                                         #
#     This module contains functions for net visualizations.                  #
#                                                                             #
# authors of the toolbox:                                                     #
#     Agustín Amalfitano                                                      #
#     Diego Comas	                			                       	      #
#     Juan Iturriaga    		                                   		      #
#                                                                             #  
# colaborators:                                                               #
#     Luciana Simón Gonzalez                                			      #
#     Gustavo Meschino			                                   		      #
#     Virginia Ballarin			                                		      #
#     Franco Ercoli				                                       	      #
#                                                                             #
# *LPI-ICyTE-CONICET-UNMDP                                                    #
#                                                                             #
###############################################################################

# --------------------------IMPORTS---------------------------------------------
import tensorflow as tf

# ------------------------------------------------------------------------------
def normalize_fmaps(fmap):
    """
     This function normalizes a feature map with values in [0, 255].

    --Inputs:
       
        fmap = A TENSOR with [1, number_rows, number_columns, output_number]
               with the features maps resulting of consulting an CNN.

    --Outputs:

        fmap_normalized = A normalized version of "fmap".

        output_number = Tre number of output in the feature map.

        square_shape = The required side for an array to reorder the 
                       feature maps.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries
    import math

    # Obtaining the size of the "feature_maps":
    lens = fmap.shape
    output_number = lens[3]
    
    # Compute the shape for an array containing the maps:
    square_shape = math.ceil(math.sqrt(output_number))

    # Normalization:
    minimum, maximum = fmap.min(), fmap.max()
    fmap_normalized = 255 * (fmap - minimum) / (maximum - minimum)

    # Returning outputs:
    return(fmap_normalized, output_number, square_shape)

# ------------------------------------------------------------------------------
def save_fmaps_as_png(feature_maps, ixs, image_name, result_path):
    """
     This function saves single images for all the feature maps.

    --Inputs:
       
        feature_maps = A TENSOR with 
                       [1, number_rows, number_columns, output_number]
                       with the features maps resulting of consulting an CNN.

        ixs = A LIST with the numbers of the layers to set as outputs.

        image_name = A STRING with the name of the image used for defining the 
                     feature map.

        result_path = A STRING with the path for saving results.

    --Outputs:

        None

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries
    import cv2
    from files import create_folder

    # Loop through the feature maps and display them (normalized):
    for fmap, block_number in zip(feature_maps, ixs):
        # Create another subdirectory to diferenciate between feature maps
        subdir_image = result_path +  "/" + "layer_" + str(block_number)
        create_folder(subdir_image)

        # Normalizing the feature map:
        fmap, output_number, square = normalize_fmaps(fmap)

        # Loop through all filters in each block:
        ix = 1
        while ix <= output_number:
            # Save the output of the particular block
            name = image_name + "_" + str(block_number) + "_id_" + str(ix) + ".png"
            cv2.imwrite(subdir_image + "/" + name, fmap[0, :, :, ix-1])
            ix += 1
    
    # Returning outputs:
    return()

# ------------------------------------------------------------------------------
def save_fmaps_as_mosaicos(feature_maps, ixs, image_name, result_path):
    """
     This function saves a feature map as a mosaic.

    --Inputs:
       
        feature_maps = A TENSOR with 
                       [1, number_rows, number_columns, output_number]
                       with the features maps resulting of consulting an CNN.

        ixs = A LIST with the numbers of the layers to set as outputs.

        image_name = A STRING with the name of the image used for defining the 
                     feature map.

        result_path = A STRING with the path for saving results.

    --Outputs:

        None

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries
    from matplotlib import pyplot

    # Loop through the feature maps and display them (normalized):
    for fmap, block_number in zip(feature_maps, ixs):

        # Normalizing the feature map:
        fmap, output_number, square = normalize_fmaps(fmap)
        
        # Loop through all filters in each block:
        ix = 1
        while ix <= output_number:
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray', vmin=0, vmax=255) # I make sure that it respects the gray levels of the normalization
            ix += 1
        
        # Guardo la salida del bloque en particular
        name_mosaic = "mosaic_" + image_name + "_" + str(block_number) + ".png"
        
        # Show the figure
        pyplot.savefig(result_path + "/" + str(name_mosaic))
        pyplot.close()
    
    # Returning outputs:
    return()

# ------------------------------------------------------------------------------
def info_dic(result_path, image_name, dic):
    """
     This function saves an info.mat file inside of each image folder.

    --Inputs:
       
        image_name = A STRING with the name of the image used for defining the 
                     feature map.

        result_path = A STRING with the path for saving results.

        dic = The DICTIONARY to be saved.

    --Outputs:

        None

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries
    from scipy.io import savemat

    # Saving:
    savemat(result_path + "/" + str(image_name) + "_info.mat", dic)

    # Returning outputs:
    return()

# ------------------------------------------------------------------------------
def layers_dic(feature_maps, dic, result_path, image_name):
    """
     This function saves a feature map as MATLAB files.

    --Inputs:
       
        feature_maps = A TENSOR with 
                       [1, number_rows, number_columns, output_number]
                       with the features maps resulting of consulting an CNN.

        dic = The DICTIONARY with information used as baseline for saving.

        result_path = A STRING with the path for saving results.

        image_name = A STRING with the name of the image used for defining the 
                     feature map.

    --Outputs:

        feature_maps_normalized = A normalized version of the feature maps in 
                                  unit16.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries
    from scipy.io import savemat
    import numpy
    import copy
    
    # Inicializing:
    minimums = []
    maximums = []
    feature_maps_normalized = []
    feature_maps_normalized = numpy.empty((len(feature_maps),), dtype=numpy.object)

    # Loop:
    for i in range(len(feature_maps)):
        # I get the minimums and maximums of the block (filter) in which I am:
        minimum_layer, maximum_layer = feature_maps[i].min(), feature_maps[i].max()
        
        # I save the extremes:
        minimums.append(minimum_layer)
        maximums.append(maximum_layer)
        
        # Normalize:
        aux = (2^16-1) * (feature_maps[i] - minimum_layer) / (maximum_layer - minimum_layer)
        feature_maps_normalized[i]=numpy.uint16(aux)
        
    # Cloning the DICTIONARY:
    dic2 = copy.deepcopy(dic)

    # Adding new keys:
    dic2["feature_maps"] = feature_maps_normalized
    dic2["minimums_layers"] = minimums
    dic2["maximums_layers"] = maximums
    
    # Saving:
    savemat(result_path + "/" + str(image_name) + "_feature_maps.mat", dic2)

    # Returning outputs:
    return(feature_maps_normalized)

# ------------------------------------------------------------------------------
@tf.function
def gradient_ascent_step(img, filter_index, learning_rate, feature_extractor):
    """
    This function computes gradient ascent. It simply computes the gradients 
    of the loss above with regard to the input image, and updates the update 
    image so as to move it towards a state that will activate the target 
    filter more strongly.

    --Inputs:
       
        img = A IMAGE to be used as input for computing loss.

        filter_index = The index of the filter in the output layer to be 
                       analyzed.

        learning_rate = The learning rate value for the maximization.

        feature_extractor = A Keras model set as feature extractor (a output 
                            is defined in a convolutional layer)

    --Outputs:

        loss = The value of the loss computed with "compute_loss".

        img = The updated image.

    *Extrated from Keras: 
            https://keras.io/examples/vision/visualizing_what_convnets_learn/.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries:
    from deep_nets import compute_loss

    # Compute loss:
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    
    # Compute gradients.
    grads = tape.gradient(loss, img)
    
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads

    # Returning outputs:
    return loss, img

# ------------------------------------------------------------------------------
def initialize_image(img_width, img_height):
    """
     This function initialize an IMAGE with random values from a uniform 
     distribution in [0, 1].

    --Inputs:
       
        img_width = The number of columns in the image.

        img_width = The number of rows in the image.

    --Outputs:

        img = A TENSOR with dimension [1, img_width, img_height, 3].

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Initializing with values in [0, 1]:
    img = tf.random.uniform((1, img_width, img_height, 3))
    
    # Here we scale our random inputs to [-0.125, +0.125]
    img = (img - 0.5) * 0.25
    
    # Returning outputs:
    return img

# ------------------------------------------------------------------------------
def maximize_filter(filter_index, feature_extractor, iterations=30, learning_rate=10):
    """
    This function obtains the image that maximizes a specific filter in a Neural 
    model using gradient maximization.

    --Inputs:
       
        filter_index = The index of the filter in the output layer to be 
                       maximized.

        feature_extractor = A Keras model set as feature extractor (a output 
                            is defined in a convolutional layer)

        iterations = The number of iterations for the maximization. By 
                     default is 30.

        learning_rate = The learning rate value for the maximization. 
                        By default is 10.
    
    --Outputs:

        img = A NUMPY ARRAY with the image that maximizes the output specified, 
              with [input_shape(0), input_shape(1), 3] dimensions.

        img2 = A NUMPY ARRAY with the image that maximizes the output specified, 
               with [1, input_shape(0), input_shape(1), 3] dimensions.

    *Extrated from Keras: 
            https://keras.io/examples/vision/visualizing_what_convnets_learn/.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries:
    import numpy as np

    # Inicializing:
    input_shape = feature_extractor.input_shape
    img_width = input_shape(0)
    img_height = input_shape(1)
    img = initialize_image(img_width, img_height)
    
    # Iterate:
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate, feature_extractor)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    img2 = np.expand_dims(img, axis=0)
    
    # Returning outputs:
    return loss, img, img2

# ------------------------------------------------------------------------------
def deprocess_image(img):
    """
     This function converts the resulting input image of "gradient_ascent_step" 
     back to a displayable form, by normalizing it, center-cropping it, 
     and restricting it to the [0, 255] range.

    --Inputs:
       
        img = A NUMPY ARRAY with the image that maximizes the output specified, 
              with [input_shape(0), input_shape(1), 3] dimensions.

    --Outputs:

        img_converted = A NUMPY ARRAY with the image in a displayable format.

    *Extrated from Keras: 
            https://keras.io/examples/vision/visualizing_what_convnets_learn/.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  
    
    # Libraries:
    import numpy as np
    
    # Normalize array: center on 0., ensure variance is 0.15
    img_converted -= img.mean()
    img_converted /= img.std() + 1e-5
    img_converted *= 0.15

    # Clip to [0, 1]
    img_converted += 0.5
    img_converted = np.clip(img, 0, 1)

    # Convert to RGB array
    img_converted *= 255
    img_converted = np.clip(img_converted, 0, 255).astype("uint8")

    # Returning outputs:
    return img_converted

# ------------------------------------------------------------------------------
def feature_visualization_layer(model, id_layer, list_outputs, experiment_name, result_folder):
    """
     This function performs feature-visualization on a specific filter. The 
     results are saved accordingly.

    --Inputs:

        model = A Neural Network OBJECT generated using Keras. It should be 
                formatting for accepting RGB input images. If the original 
                model does not accept them, you can solve it using the 
                "rgb_to_gray" and the "reshape_input_model" functions. 
       
        id_layer = A LIST with the number of layer to "visualize". If it is 
                   "-1", then the classification layer will be analyzed.

        list_outputs =  A LIST containing the names of the filters to 
                        analyze. Please note that if the classification layer 
                        is being analyzed the this list should contain the 
                        names of the classes.

        experiment_name = A STRING with the name of the experiment for identification.

        result_folder = A STRING with the path for saving results.

    --Outputs:

        None.

    *It is based on the toolbox "xplique":
            https://github.com/deel-ai/xplique.

    """
    
    # Author: Diego Comas
    # Revised by: -  
    
    # Libraries:
    import matplotlib.pyplot as plt
    from xplique.features_visualizations import Objective, maco
    from xplique.plots.image import plot_maco

    # Creating a list for iterating between "id_outputs" and "list_outputs":
    classes = [(id_output, list_outputs[id_output]) for id_output in range(len(list_outputs))]

    # Iterating between outputs
    for output_id, output_name in classes:
        # Create the objective, "id_layer" is the  layer to observe and the "output_id" are the ids of the filters!
        obj_logits = Objective.neuron(model, id_layer, output_id)
        print(obj_logits)
        print(output_name)
        img, alpha = maco(obj_logits, nb_steps=128, values_range = (-1, 1))
        plot_maco(img, alpha)
        plt.title(output_name)

        # Saving the results:
        file_name = experiment_name + '_' + output_name + '.png'
        plt.savefig(result_folder + file_name)
        plt.show()

# ------------------------------------------------------------------------------
def grad_cam_explainer(model, list_methods, list_images, img_paths, labels_in_model, img_size, experiment_name, result_folder):
    """
     This function performs grad-cam based explainers from a model and a 
     set of images.

    --Inputs:

        model = A Neural Network OBJECT generated using Keras. It should be 
                formatting for accepting RGB input images. If the original 
                model does not accept them, you can solve it using the 
                "rgb_to_gray" and the "reshape_input_model" functions. 
       
        list_methods = A LIST with the names (STRING) of the explainers to 
                       be used. The explainer are implemented:
                       "GradCAM", "GradCAMPP", "Saliency", "DeconvNet", 
                       "GradientInput", "GuidedBackprop".

        list_images =  A LIST containing the names of the images relative 
                       to "img_paths".

        img_paths = A STRING with the base folder for reading the images 
                    in "list_images".

        labels_in_model = A LIST with the names of the labels identified 
                          in the model.

        img_size = A LIST with the size proper for the input to the model.

        experiment_name = A STRING with the name of the experiment for identification.

        result_folder = A STRING with the path for saving results.

    --Outputs:

        None.

    *It is based on the toolbox "xplique": 
            https://github.com/deel-ai/xplique.

    """
    
    # Author: Diego Comas
    # Revised by: -  
    
    # Libraries:
    from xplique.attributions import GradCAM, GradCAMPP, Saliency, DeconvNet, GradientInput, GuidedBackprop
    from xplique.plots import plot_attributions
    import numpy as np
    import matplotlib.pyplot as plt

    # Defining arbitrary parameters (common for all methods):
    parameters_1 = {
        "model": model,
        "output_layer": -1,
        "batch_size": 16,
        "conv_layer": None,
    }

    parameters_2 = {
        "model": model,
        "output_layer": None,
        "batch_size": 16,
    }

    # Defining explainers according to list_methods":
    explainers = {}
    for method in list_methods:
        if method == "GradCAM":
            explainers["GradCAM"] = GradCAM(**parameters_1)
        elif method == "GradCAMPP":
            explainers["GradCAMPP"] = GradCAMPP(**parameters_1)
        elif method == "Saliency":
            explainers["Saliency"] = Saliency(**parameters_2)
        elif method == "DeconvNet":
            explainers["DeconvNet"] = DeconvNet(**parameters_2)
        elif method == "GradientInput":
            explainers["GradientInput"] = GradientInput(**parameters_2)
        elif method == "GuidedBackprop":
            explainers["GuidedBackprop"] = GuidedBackprop(**parameters_2)

    # Iteraring on "list_images" for genetaring the GRAD CAM MAPS:
    for id_image in range(len(list_images)):
        # Obtaining names and labels for the images:
        image_full_name = img_paths + list_images[id_image]
        image_short_name = list_images[id_image]
        
        # Preparing the image for the model:
        image_array = np.expand_dims(tf.keras.preprocessing.image.load_img(image_full_name, target_size=(img_size[0],img_size[1],img_size[2])), 0)
        image_array = np.array(image_array, dtype=np.float32) / 255.0

        # Iterating on the labels in the model:
        for label in labels_in_model:
                # Obtaining number of labels:
                num_labels = len(labels_in_model)

                # Obtaining the output neuron for the class "label":
                id_label = labels_in_model.index(label)

                # Expanding dimensions properly:
                y = np.expand_dims(tf.keras.utils.to_categorical(id_label, num_labels), 0)

                # iterate on all methods
                for method_name, explainer in explainers.items():
                    # compute explanation by calling the explainer
                    explanation = explainer.explain(image_array, y)

                    # visualize explanation with plot_explanation() function
                    plot_attributions(explanation, image_array, img_size=5, cmap='cividis', cols=1, alpha=0.6)

                    # Defining the figure name:
                    figure_file_name = experiment_name + '_' + method_name + '_' + image_short_name + '_' + label + '.png'
                    
                    # Saving: 
                    plt.savefig(result_folder + figure_file_name)
                    print(figure_file_name)
                    plt.close()

# ------------------------------------------------------------------------------