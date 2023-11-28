###############################################################################
#                         ICyTE-LPI-Deep Toobox                               #
# module name:                                                                #
#     LPI_deep_NetVisualization                                               #
#                                                                             #
# module description:                                                         #
#     This module contains all the required function for net visualizations.  #
#                                                                             #
# authors of the toolbox:                                                     #
#     Agustín Amalfitano                                                      #
#     Diego Comas				                       						  #
#     Franco Ercoli				                       						  #
#     Juan Iturriaga    		                       						  #
#                                                                             #  
# colaborators:                                                               #
#     Luciana Simón Gonzalez                        						  #
#     Virginia Ballarin			                       						  #
#     Gustavo Meschino			                       						  #
#                                                                             #
# versions:                                                                   #
#     module: 1.0 - 2023-04-30                                                #
#     toolbox 1.0 - 2023-XX-XX                                                #
#                                                                             #
# *LPI-ICyTE-CONICET-UMDP                                                     #
#                                                                             #
###############################################################################

# ******************************************************************************
# ------------------------------LIST OF VERSIONS--------------------------------
# Version |   Date   |         Authors      | Description
# -------- ---------- ---------------------- -----------------------------------
#
#    1.0   04/30/2023  Diego Comas            First version.
#                      Agustín Amalfitano 
#                      Franco Ercoli
#
# ******************************************************************************

# ------------------------------LIST OF FUNCTIONS-------------------------------
# Functions              |   Date    |    Authors          |   Description
# ------------------------------------------------------------------------------
#
# normalize_fmaps         04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         normalize a feature 
#                                                           maps with values in 
#                                                           [0, 255].
#
# save_fmaps_as_png       04/21/2023   Agustín Amalfitano / This function saves 
#                                       Diego Comas         single images for 
#                                                           all the feature
#                                                           maps.
#
# save_fmaps_as_mosaics   04/21/2023   Agustín Amalfitano / This function saves 
#                                       Diego Comas         a feature map as a 
#                                                           mosaic.
#
# info_dic                04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         saves an info.mat 
#                                                           file inside of each 
#                                                           image folder.
#
# layers_dic              04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         saves a feature map 
#                                                           as MATLAB files.
#
# gradient_ascent_step    04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         compute gradient 
#                                                           ascent.
#
# initialize_image        04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         initialize an IMAGE 
#                                                           with random values 
#                                                           from a uniform 
#                                                           distribution in 
#                                                           [0, 1].
#
# maximize_filter         04/21/2023   Agustín Amalfitano / This function obtain 
#                                       Diego Comas         the image that 
#                                                           maximizes an specific 
#                                                           filter in a Neural 
#                                                           model using 
#                                                           gradient 
#                                                           maximization.
#
# deprocess_image         04/21/2023   Agustín Amalfitano / This function 
#                                       Diego Comas         convert the resulting 
#                                                           input image of 
#                                                           "gradient_ascent_step" 
#                                                           back to a displayable 
#                                                           form.
#
# save_mosaic             04/21/2023   Agustín Amalfitano /
#                                       Diego Comas
#
# layer_i_analysis        04/21/2023   Agustín Amalfitano /
#                                       Diego Comas
#
# ------------------------------------------------------------------------------

# --------------------------IMPORTS---------------------------------------------
import tensorflow as tf

# ------------------------------------------------------------------------------
def normalize_fmaps(fmap):
    """
     This function normalize a feature maps with values in [0, 255].

    --Inputs:
       
        fmap = A TENSOR with [1, number_rows, number_columns, output_number]
               with the features maps resulting of consulting an CNN.

    --Outputs:

        fmap_normalized = A normalized version of "fmap".

        output_number = Tre number of output in the feature map.

        square_shape = The required side for an array to reorder the 
                       feature maps.

    """

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

    # Libraries
    import cv2
    from LPI_deep_Files import create_folder

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
     This function saves an info.mat file inside of each image folder

    --Inputs:
       
        image_name = A STRING with the name of the image used for defining the 
                     feature map.

        result_path = A STRING with the path for saving results.

        dic = The DICTIONARY to be saved.

    --Outputs:

        None

    """

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
    This function compute gradient ascent. It simply computes the gradients 
    of the loss above with regard to the input image, and update the update 
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

    # Libraries:
    from LPI_deep_Networks import compute_loss

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

    # Initializing with values in [0, 1]:
    img = tf.random.uniform((1, img_width, img_height, 3))
    
    # Here we scale our random inputs to [-0.125, +0.125]
    img = (img - 0.5) * 0.25
    
    # Returning outputs:
    return img

# ------------------------------------------------------------------------------
def maximize_filter(filter_index, feature_extractor, iterations=30, learning_rate=10):
    """
    This function obtain the image that maximizes an specific filter in a Neural 
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
     This function convert the resulting input image of "gradient_ascent_step" 
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
def save_mosaic(rows, n_columns, filters_layer_i, filter_index,
                caja_roja, fmap_i_im_j, image_names,
                ix=1, save_outputs=False, mosaic_index=-1):

    """
   
        TO BE REVISED!!!!
    """
    
    # Libraries:
    from matplotlib import pyplot, patches
    import traceback


    output_ij = fmap_i_im_j[0, :, :, filter_index]
    # ixim es el indice que arma la image_name, empieza siempre desde 1
    ixim = 1
    for _ in range(rows):
        first = True
        try:
            for _ in range(n_columns):
                if (ix <= filters_layer_i):
                    # save if filter separately

                    # specify subplot and turn of axis
                    ax = pyplot.subplot(rows + 1, n_columns, ixim)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if ixim <= n_columns:
                        ax.set_title(str(ix - 1),fontsize='small')

                    # eje y con label rotado -45 y color a eleccion
                    if first:
                        pyplot.ylabel(str(ix - 1),rotation=-45, fontsize='small', color="black")

                    if ix == filter_index + 1:   

                        # remarcar la image_name con un rectangulo
                        ##################################
                        autoAxis = ax.axis()
                        rec = patches.Rectangle((autoAxis[0]+caja_roja,autoAxis[2]-1),(autoAxis[1]-autoAxis[0])-caja_roja,(autoAxis[3]-autoAxis[2])+caja_roja, color="red", fill=False,lw=3)
                        rec = ax.add_patch(rec)
                        rec.set_clip_on(False)
                        ##################################


                    # plot filter channel in grayscale cmap="gray", "jet" a color

                    if (ixim - 1) % 2 == 0:
                        pyplot.imshow(fmap_i_im_j[0, :, :, ix-1], cmap='gray')
                    else:
                        pyplot.imshow(fmap_i_im_j[0, :, :, ix-1], cmap='gray')
                ix += 1
                ixim += 1
                if first:
                    first = False
        except:
            traceback.print_exc()
            print("no se condice la cantidad de image_namees analizadas con el n de filtros")
            continue
            # show the figure
    # pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=.1, hspace=0.1)
    # pyplot.show()
    if mosaic_index >= 0:
        mosaic_name_i = image_names + "mosaic_" + str(filter_index) + "_" + str(mosaic_index) + ".png"
    else:
        print(mosaic_index)
        mosaic_name_i = image_names + "mosaic_" + str(filter_index) + ".png"
        
    output_name_i = image_names + "output" + str(filter_index) + ".png"
    
    pyplot.savefig(mosaic_name_i, dpi=1000)
    pyplot.close()
    #########################


    if save_outputs:
        # solucion
        #https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
        pyplot.imshow(output_ij, interpolation='nearest')
        #pyplot.gray()  # escala de grises
        pyplot.savefig(output_name_i, dpi=1000)
        pyplot.close()

    # Returning outputs:
    return()

# ------------------------------------------------------------------------------
def layer_i_analysis(layer_name, result_folder, model):

    """
   
        TO BE REVISED!!!!
    """

    # Libraries:
    from LPI_deep_Files import create_folder

    # create subfolder i
    layer_subfolder_i = result_folder + layer_name
    create_folder(layer_subfolder_i)

    image_names = layer_subfolder_i + "/" + model + "-filters-" + layer_name + "_"
    print(image_names)

    # Returning outputs:
    return()

# ------------------------------------------------------------------------------