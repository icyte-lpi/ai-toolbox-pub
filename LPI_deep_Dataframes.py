###############################################################################
#                         ICyTE-LPI-Deep Toobox                               #
# module name:                                                                #
#     LPI_deep_Dataframes                                                     #
#                                                                             #
# module description:                                                         #
#     This module contains all the required function for using DataFrames.    #
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
#     module: 1.0 - 2023-05-14                                                #
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
#    1.0   04/21/2023  Diego Comas            First version.
#                      Agustín Amalfitano 
#                      Franco Ercoli
#
# ******************************************************************************

# ------------------------------LIST OF FUNCTIONS-------------------------------
# Functions              |   Date    |    Authors          |   Description
# ------------------------------------------------------------------------------
# balance_df              04/05/2023   Diego Comas /         Particiona un DATAFRAME
#                                       Agustín Amalfitano   en TRAIN, VALIDATION y
#                                                            TEST
#
# hold_out_df             04/05/2023   Diego Comas /         Define los iteradores
#                                       Agustín Amalfitano   de ENTRENAMIENTO,
#                                                            VALIDACIÓN Y TEST
#
# filter_df               04/05/2023   Diego Comas /         Filtra un DATAFRAME
#                                       Agustín Amalfitano
#
# create_iter_df          04/05/2023   Diego Comas /         Define iteradores a
#                                       Agustín Amalfitano   partir de DataFrames
#
# read_csv                04/05/2023   Diego Comas /         Obtener un DATAFRAME a
#                                       Agustín Amalfitano   partir de un CSV
#
# walk_into_subdirs       05/15/2023   Diego Comas /         This function walk  
#                                       Agustín Amalfitano   thought the 
#                                                            subdirectories looking 
#                                                            for files to create 
#                                                            a DATAFRAME.
#
# save_df                 05/15/2023   Diego Comas /         This function saves 
#                                       Agustín Amalfitano   a PANDAS DATAFRAME 
#                                                            in a "csv" file.
#
# ------------------------------------------------------------------------------

# --------------------------IMPORTS---------------------------------------------
# Reserved.

# ------------------------------------------------------------------------------
def balance_df(data_frame, mode, id_files="images", id_labels="labels"):
    """
     This function performs dataset balancing from a DATAFRAME.

    --Inputs:
       
        data_frame = The dataframe from PANDAS.

        mode = A STRING indicating how to perform balancing:
                "min" --> Adjust to the frequency of the minority class.
                "max" --> Adjust to the frequency of the majority class.

        id_files = The name of the field where the names of the files are.

        id_labels = The name of the field where the labels are.

    --Outputs:

        data_frame_balanced = A balanced version of the input DATAFRAME.

    """

    # Libraries:
    import numpy as np
    import pandas as pd
    from math import ceil

    # Find the distinct values and frequencies:
    labels, frequencies = np.unique(data_frame[id_labels].values, return_counts=True)

    # Check the mode for balancing:
    if (mode == "min"):
        # Adjust to the frequency of the minority class:
        number_samples = min(frequencies)
    else:
        # Adjust to the frequency of the majority class:
        number_samples = max(frequencies)

    # --Sampling:
    # Inicialization:
    data_frames_Auxiliar = [[] for id_Label in range(len(labels))]
    # Loop:
    for id_Label in range(len(labels)):
        # Analize the state of the label:
        n_samples = frequencies[id_Label]
        label = labels[id_Label]
        # Check for kind of balancing:
        if (n_samples >= number_samples):
            data_frames_Auxiliar[id_Label] = data_frame[data_frame[id_labels] == label].sample(
                number_samples)
        else:
            # I have to check:
            n_iteration = ceil(number_samples / n_samples)
            # Look forward:
            Auxiliar = [[] for id_iteration in range(n_iteration)]
            for id_iteration in range(n_iteration):
                if (id_iteration < (n_iteration-1)):
                    # I am not in the last one:
                    Auxiliar[id_iteration] = data_frame[data_frame[id_labels] == label]
                else:
                    #  I am in the last one:
                    if ((number_samples % n_samples) == 0):
                        Auxiliar[id_iteration] = data_frame[data_frame[id_labels] == label]
                    else:
                        Auxiliar[id_iteration] = data_frame[data_frame[id_labels] == label].sample(
                            number_samples % n_samples)
            # Join:
            data_frames_Auxiliar[id_Label] = pd.concat(Auxiliar)

    # Join everything:
    data_frame_balanced = pd.concat(data_frames_Auxiliar)

    # Returning outputs:
    return data_frame_balanced

# ------------------------------------------------------------------------------
def hold_out_df(data_frame, proportions, result_name,
                                  id_files="images", id_labels="labels", balancing_criterium="no"):
    """
     This function make a partition in a DATAFRAME to form TRAIN, VALIDATION 
     and TEST data considering HOLD-OUT.

    --Inputs:
       
        data_frame = The dataframe from PANDAS.

        proportions = A LIST with the proportions to be used in HOLD-OUT 
                      [TRAIN, VALIDATION, TEST]. If validation is not used, 
                      then complete with 0.

        result_name = The full-name for a "xlsx" file for saving the information of the data 
                      in each sub-set. 

        id_files = The name of the field where the names of the files are.

        id_labels = The name of the field where the labels are.

        balancing_criterium = A STRING defining the kind of balancing to perform:
                "min" --> Adjust to the frequency of the minority class.
                "max" --> Adjust to the frequency of the majority class.
                "no" --> None balancing is used.      

    --Outputs:

        df_train = A DATAFRAME with the training data.

        df_validation = A DATAFRAME with the validation data.

        df_test = A DATAFRAME with the test data.

    """

    # Libraries:
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # --Obtaining the DATAFRAME of TEST:
    df_Auxiliar, df_test = train_test_split(data_frame,
                                            test_size=proportions[2],
                                            stratify=list(data_frame[id_labels]))

    # -Obtaining TRAINING and VALIDATION:
    if proportions[1] == 0:
        df_train_Auxiliar = df_Auxiliar
        df_validation = []
        df_flag = False
    else:
        # Obtaining TRAINING:
        df_train_Auxiliar, df_validation = train_test_split(df_Auxiliar,
                                                            test_size=proportions[1]/(
                                                                1-proportions[2]),
                                                            stratify=list(df_Auxiliar[id_labels]))
        df_flag = True

    # Ckeking for balancing:
    if (balancing_criterium == "no"):
        df_train = df_train_Auxiliar
    else:
        df_train = balance_df(
            df_train_Auxiliar, balancing_criterium, id_files, id_labels)

    # Cheking for saving files:
    if not result_name:
        # Save list of file to a "xlsx":
        writer = pd.ExcelWriter(result_name + '_SETS.xlsx')
        df_train.to_excel(writer, sheet_name='TRAIN')
        if df_flag:
            df_validation.to_excel(writer, sheet_name='VALIDATION')
        df_test.to_excel(writer, sheet_name='TEST')
        writer.save()

    # Returning outputs:
    return df_train, df_validation, df_test

# ------------------------------------------------------------------------------
def findin_df(data_frame, id="labels"):
    """
     This function find distinct values in a specific field of a DATAFRAME.

    --Inputs:
       
        data_frame = The DATAFRAME from PANDAS.

        id = The name of the field to look for distinct values. By default: 
             "labels".
    
    --Outputs:

        labels = A LIST of STRING with the names of the found labels.

        frequencies = A LIST containing number of sample for each label.

    """

    # Libraries:
    import numpy as np

    # Find the distinct values and frequencies:
    labels, frequencies = np.unique(data_frame[id].values, return_counts=True)

    # Returning outputs:
    return labels, frequencies

# ------------------------------------------------------------------------------
def filter_df(data_frame, id, value_id):
    """
     This function filter a DATAFRAME.

    --Inputs:
       
        data_frame = The dataframe from PANDAS.

        id = The name of the field to look for filtering.

        value_id = The value in the field to perform the filer.

    --Outputs:

        data_frame_filtered = A filtered version of the input DATAFRAME.

    """

    # Perform the filtering:
    data_frame_filtered = data_frame[data_frame[id] == value_id]

    # Returning outputs:
    return data_frame_filtered

# ------------------------------------------------------------------------------
def sampling_df(data_frame, id="labels", maximum_samples=0):
    """
     This function sampling a DATAFRAME.

    --Inputs:
       
        data_frame = The dataframe from PANDAS.

        id = The name of the field to look for sampling.

        maximum_samples = (Optional) The maximum number of samples to select. 
                          If it is 0 then no maximum number of samples is defined.

    --Outputs:

        data_frame_sampled = A sampled version of the input DATAFRAME.

    """

    # Libraries:
    import numpy as np
    import pandas as pd

    # --Find number of classes and frequencies:
    names, frequencies = findin_df(data_frame, id)

    # --Sampling:
    data_frames_aux = []
    for id_name in range(len(names)):
        num_samples = frequencies[id_name]
        name = names[id_name]
        if maximum_samples == 0:
            # Making the sampling:
            data_frames_aux.append(
                data_frame[data_frame[id] == name].sample(num_samples))
        else: 
            # Copying the sampling:
            data_frames_aux.append(
                data_frame[data_frame[id] == name])
    
    # Join all the parts:
    data_frame_sampled = pd.concat(data_frames_aux)

    # Returning outputs:
    return data_frame_sampled

# ------------------------------------------------------------------------------
def create_iter_df(data_frame, preproc_function, image_path, img_size, size_batch,
                                   augmentation_type=0, color_mode="rgb", id_files="images",
                                   id_labels="labels", list_classes=[], shuffle_flag=False):
    """
     This function sampling a DATAFRAME.

    --Inputs:

        data_frame = The dataframe from PANDAS.

        preproc_function = Preprocessing function to use, according to the network 
                           with which we work.

        image_path = A STRING with the base path where the images referred to in 
                     the DATAFRAMES are.

        img_size = A LIST defining the size of the input images.

        size_batch = A NUMBER indicating the size of the image batches.

        augmentation_type = A number indicating the type of augmentation to perform:
                               0 --> Do not perform augmentation.
                               1 --> Perform type 1 augmentation.

        color_mode = A STRING indicating the color mode to be used: "grayscale", 
                     "rgb", "rgba".

        id_files = The name of the field where the names of the files are.

        id_labels = The name of the field where the labels are. Default is "labels".

        list_classes = A LIST of class names in STRING form.

        shuffle_flag = A BOOLEAN that sets the shuffle condition each time the iterator object is called.
                       Note that it should be FALSE for TEST iterators.

    --Outputs:

        iterator = An ITERATOR OBJECT defined from the DATAFRAME.

    """

    # Libraries:
    from keras.preprocessing import image

    # I analyze the condition of "augmentation":
    if (augmentation_type == 1):
        # I perform augmentation of type 1:
        datagen = image.ImageDataGenerator(preprocessing_function=preproc_function,
                                           rotation_range=5,  # random rotation
                                           width_shift_range=0.05,  # random horizontal shift
                                           height_shift_range=0.05,  # random vertical shift
                                           # [1-zoom_range, 1+zoom_range]
                                           zoom_range=0.15,
                                           shear_range=0.1,
                                           fill_mode='reflect'
                                           )
    else:
        # I do not perform augmentation:
        datagen = image.ImageDataGenerator(
            preprocessing_function=preproc_function)

    # Making the iterator:
    iterador = datagen.flow_from_dataframe(dataframe=data_frame,
                                           directory=image_path,
                                           x_col=id_files,
                                           y_col=id_labels,
                                           batch_size=size_batch,
                                           shuffle=shuffle_flag,
                                           color_mode=color_mode,
                                           class_mode="categorical",
                                           target_size=img_size,
                                           classes=list_classes
                                           )

    # Returning outputs:
    return iterador

# ------------------------------------------------------------------------------
def read_csv(name_CSV, id_files, id_labels, separator=',', additional_filters=[]):
    """
     This function reads a "csv" file and returns a PANDAS DATAFRAME.

    --Inputs:

        name_CSV = A STRING with the full name of the CSV.

        id_files = The name of the field where the names of the files are.

        id_labels = The name of the field where the labels are. Default is "labels".

        separator = STRING with the symbol that delimits the separation in the CSV. 
                    Default is ",".

        additional_filters = A LIST of STRING with additional fields to read from the file.


    --Outputs:

        df_data = A DATAFRAME, sorted by "id_files".

    """
    
    # Libraries:
    import pandas as pd

    # Making the list of fields for reading:
    aux_list = []
    aux_list.append(id_files)
    aux_list.append(id_labels)
    if additional_filters:
        for element in additional_filters:
            aux_list.append(element)

    # Reading the DATAFRAME:
    df_data = pd.read_csv(name_CSV,
                          delimiter=separator, usecols=aux_list)

    # Returning outputs:
    return df_data

# ------------------------------------------------------------------------------
def walk_into_subdirs(current_path, find_partition, find_additional_info=[]):
    """
     This function walk thought the subdirectories looking for files to create 
     a DATAFRAME.

    --Inputs:

        current_path = A STRING with the path to look in for files.

        find_partition = A BOOLEAN indicating if to identify the partition as 
                         the folders current_path. If it is not the case, we 
                         expect the folders in the "current_path" be the labels 
                         and each of it have files. 

        find_additional_info = A STRING containing a separator to identify
                               something in the first part of the file name. 
                               When defined this information is added to a field 
                               called "sublabel" jointly with the "label".

    --Outputs:

        dataframe = The DATAFRAME generated. The first row is the name of the 
                    columns.

    """
    # Libraries:
    import os

    # Initialize dataframe list:
    dataframe = []

    # Initializing names of columns:
    columns = "name,label,"
    if not find_additional_info:
        columns = columns + "sublabel,"
    if not find_partition:
        columns = columns + "partition,"
    columns = columns + "rel_path"
    dataframe.append(columns)

    # Define the list of dirs inside "current_path":
    list_first_dic = [f.name for f in os.scandir(current_path) if f.is_dir()]

    # Walk the subdirs
    for path, subdirs, files in os.walk(current_path):
        # iterate thought each file in subdirs
        for name in files:
            # Inicializing the line in the DATAFRAME:
            add_to_dataframe = name

            # List with the path and the name:
            path_and_name = (os.path.join(path, name))
            
            # Split the STRING to get differents atributes:  
            split_str = path_and_name.split("/")

            # Reading the label as the folder previous to the files: 
            label = split_str[-2]
            # Adding the label:
            add_to_dataframe = add_to_dataframe + "," + label
        
            # Cheking for finding additional conditions:
            if not find_additional_info:
                # I have to identify a condition defined in the file name:
                aux = name.split(find_additional_info)

                # I defined the sublabel:
                sublabel = label + "_" + aux[-2]

                # Adding the sublabel:
                add_to_dataframe = add_to_dataframe + "," + sublabel

            # Cheking for finding partition:
            if find_partition:
                # I have to identify the partition defined in the folders containing in the main level:
                partition = split_str[-3]

                # Adding the "partition":
                add_to_dataframe = add_to_dataframe + "," + partition

            # Defining relative path from "current_path":
            rel_path = split_str[-len(split_str):]
            rel_path = '/'.join(rel_path)

            # Adding the "partition":
            add_to_dataframe = add_to_dataframe + "," + rel_path

            # Save the list only if in the subdir and not in 
            # the same dir of the script:
            if find_partition:
                if split_str[-3] in list_first_dic:
                    dataframe.append(add_to_dataframe)
            else:
                if split_str[-2] in list_first_dic:
                    dataframe.append(add_to_dataframe)

    # Returning outputs:
    return(dataframe)
    
# ------------------------------------------------------------------------------   
def save_df(df_name, dataframe):
    """
     This function saves a PANDAS DATAFRAME in a "csv" file.

    --Inputs:

        df_name = A STRING with the full name for the CSV name.

        dataframe = The PANDAS DATAFRAME to be save.

    --Outputs:

        None.

    """

    # Opening the file and writing in it:
    with open(df_name, "w") as f:
        for elem in dataframe:
            f.write(elem + "\n")

# ------------------------------------------------------------------------------
