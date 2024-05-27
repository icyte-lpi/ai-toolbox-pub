###############################################################################
#                        ICyTE - LPI - AI Toolbox                             #
# module name:                                                                #
#     deep_nets                                                               #
#                                                                             #
# module description:                                                         #
#     This module contains functions for creating, freezing, training, and    #
#     validation of Deep Neural Networks.                                     #
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
import os

# ------------------------------------------------------------------------------
def create_cnn(model_name, number_classes, classifier_scheme=1):
    """
     This function generates a CNN using Keras.

    --Inputs:
       
        model_name = A STRING containing the name of the model to be created.
                     The implemented CNN are:
                     For transfer-learning:
                            'ResNet50', 'VGG16', 'VGG19', 'ResNet50V2',
                            'InceptionV3', 'MobileNet', 'MobileNetV2', 
                            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                            'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                            'EfficientNetB6', 'EfficientNetB7'

                     CNN from scratch: Consult program code.

        number_classes = The numbers of classes for the classification.

        classifier_scheme = A number indicating the structure of the classification 
                            phase of the CNN. By default "1".
                '1' --> GlobalAveragePooling2D + Dense(512, activation='relu') + Dropout(0.5) + 
                            + Dense(50, activation='relu') + Dropout(0.5) + softmax
    --Outputs:

        net = The OBJECT NET created with KERAS according to the parameters.

        preproc_function = The preprocessing function to apply for using the 
                           generated CNN.

        images_size = Returns the size of the input image, according to the CNN.

        transfer_flag = A BOOLEAN indicating if the CNN is for transfer-learning.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    from keras import layers
    from keras.models import Model

    # List of implemented transfer-learning-based CNN:
    list_transfer_nets = ['ResNet50', 'VGG16', 'VGG19', 'ResNet50V2',
                            'InceptionV3', 'MobileNet', 'MobileNetV2', 
                            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                            'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                            'EfficientNetB6', 'EfficientNetB7']
    transfer_flag = False
    for element in list_transfer_nets:
        if (element == model_name):
            transfer_flag = True

    # --CNN creation:
    # Cheking if I am in "transfer-learning":
    if transfer_flag:
        
        # I am in, so I create the proper model:
        if model_name == "ResNet50":
            # We use "ResNet50":
            from keras.applications.resnet import preprocess_input, ResNet50
            images_size = (224, 224)
            base_model = ResNet50(weights='imagenet',
                                   include_top=False)
        
        elif model_name == "VGG16":
            # We use "VGG16":
            from keras.applications.vgg16 import preprocess_input, VGG16
            images_size = (224, 224)
            base_model = VGG16(weights='imagenet',
                                include_top=False)
        
        elif model_name == "VGG19":
            # We use "VGG19":
            from keras.applications.vgg19 import preprocess_input, VGG19
            images_size = (224, 224)
            base_model = VGG19(weights='imagenet',
                                include_top=False)
        
        elif model_name == "ResNet50V2":
            # We use "ResNet50V2"
            from keras.applications.resnet_v2 import preprocess_input, ResNet50V2
            images_size = (224, 224)
            base_model = ResNet50V2(weights='imagenet',
                                     include_top=False)
        
        elif model_name == "InceptionV3":
            # We use "InceptionV3"
            from keras.applications.inception_v3 import preprocess_input, InceptionV3
            images_size = (224, 224)
            base_model = InceptionV3(weights='imagenet',
                                      include_top=False)
        
        elif model_name == "MobileNet":
            # We use "MobileNet"
            from keras.applications.mobilenet import preprocess_input, MobileNet
            images_size = (224, 224)
            base_model = MobileNet(weights='imagenet',
                                    include_top=False)
        
        elif model_name == "MobileNetV2":
            # We use "MobileNetV2"
            from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
            images_size = (224, 224)
            base_model = MobileNetV2(weights='imagenet',
                                      include_top=False)
            
        elif model_name == "EfficientNetB0":
            # We use "EfficientNetB0"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB0
            images_size = (224,224)
            base_model = EfficientNetB0(weights='imagenet',
                                    include_top = False)
                                    
        elif model_name == "EfficientNetB1":
            # We use "EfficientNetB1"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB1
            images_size = (224,224)
            base_model = EfficientNetB1(weights='imagenet',
                                    include_top = False)
                                    
        elif model_name == "EfficientNetB2":
            # We use "EfficientNetB2"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB2
            images_size = (224,224)
            base_model = EfficientNetB2(weights='imagenet',
                                    include_top = False)
                                    
        elif model_name == "EfficientNetB3":
            # We use "EfficientNetB3"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB3
            images_size = (224,224)
            base_model = EfficientNetB3(weights='imagenet',
                                    include_top = False)
                                    
        elif model_name == "EfficientNetB4":
            # We use "EfficientNetB4"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB4
            images_size = (224,224)
            base_model = EfficientNetB4(weights='imagenet',
                                    include_top = False)                                     
                                    
        elif model_name == "EfficientNetB5":
            # We use "EfficientNetB5"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB5
            images_size = (224,224)
            base_model = EfficientNetB5(weights='imagenet',
                                    include_top = False)
                                    
        elif model_name == "EfficientNetB6":
            # We use "EfficientNetB6"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB6
            images_size = (224,224)
            base_model = EfficientNetB6(weights='imagenet',
                                    include_top = False)
        
        elif model_name == "EfficientNetB7":
            # We use "EfficientNetB7"
            from keras.applications.efficientnet import preprocess_input, EfficientNetB7
            images_size = (224, 224)
            base_model = EfficientNetB7(weights='imagenet',
                                         include_top=False)

        # Freezing:
        for layer in base_model.layers:
            layer.trainable = False

        # Classifier:
        if (classifier_scheme == 1):
            # Flatten and pooling:
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            # MLP:
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(50, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            pnetictions = layers.Dense(number_classes, activation='softmax')(x)

        # Ensambling:
        net = Model(inputs=base_model.input, outputs=pnetictions)

        # Saving preprocessing function:
        preproc_function = preprocess_input

    else:
        # CNN from scratch!
        if model_name == "Adhoc1":
            # Scheme 1!
            images_size = (224, 224)

            # Input:
            input_layer = layers.Input(shape=(224, 224, 3), name='Input')

            # First block:
            x = layers.Conv2D(filters=32,  # 32 filters of 11x11
                              kernel_size=(11, 11),
                              activation='relu',
                              name='Conv2D_1')(input_layer)  # convolucion
            x = layers.MaxPool2D(
                pool_size=2, name='MaxPooling2D_1')(x)  # pooling

            # Second block:
            x = layers.Conv2D(filters=54,  # 54 filters of 5x5
                              kernel_size=(5, 5),
                              activation='relu',
                              name='Conv2D_2')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_2')(x)

            # Third block:
            x = layers.Conv2D(filters=128,  # 128 filters of 3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_3')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_3')(x)

            # Fouth block:
            x = layers.Conv2D(filters=246,  # 246 filters of 3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_4')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_4')(x)

            # Fifth block:
            x = layers.Conv2D(filters=492,  # 492 filters of 3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_5')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_5')(x)

            # Fully connected
            x = layers.GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)

            x = layers.Dense(2048,
                             activation='relu',
                             name='Dense_1')(x)
            x = layers.Dropout(0.4, name='Dropout_1')(x)

            x = layers.Dense(4012,
                             activation='relu',
                             name='Dense_2')(x)
            x = layers.Dropout(0.2, name='Dropout_2')(x)

            output = layers.Dense(number_classes,
                                  activation='softmax',
                                  name='Output')(x)

            # Creating the model:
            net = Model(input_layer, output)

            # Saving preprocessing function:
            preproc_function = None

        elif model_name == "Adhoc2":
            # Scheme 1!
            images_size = (200, 200)
            # Armo la capa de entrada:
            input_layer = layers.Input(
                shape=(images_size[0], images_size[1], 1), name='Input')

            # First block:
            x = layers.Conv2D(filters=32,  # 32 filters of  11x11
                              kernel_size=(11, 11),
                              activation='relu',
                              name='Conv2D_1')(input_layer)  # convolucion
            x = layers.MaxPool2D(
                pool_size=2, name='MaxPooling2D_1')(x)  # pooling

            # Second block:
            x = layers.Conv2D(filters=54,  # 54 filters of  5x5
                              kernel_size=(5, 5),
                              activation='relu',
                              name='Conv2D_2')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_2')(x)

            # Third block:
            x = layers.Conv2D(filters=128,  # 128 filters of  3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_3')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_3')(x)

            # Fourth block:
            x = layers.Conv2D(filters=246,  # 246 filters of  3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_4')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_4')(x)

            # Fifth block:
            x = layers.Conv2D(filters=492,  # 492 filters of  3x3
                              kernel_size=(3, 3),
                              activation='relu',
                              name='Conv2D_5')(x)
            x = layers.MaxPool2D(pool_size=2, name='MaxPooling2D_5')(x)

            # Fully connected:
            x = layers.GlobalAveragePooling2D(name='GlobalAveragePooling2D')(x)

            x = layers.Dense(2048,
                             activation='relu',
                             name='Dense_1')(x)
            x = layers.Dropout(0.4, name='Dropout_1')(x)

            x = layers.Dense(4012,
                             activation='relu',
                             name='Dense_2')(x)
            x = layers.Dropout(0.2, name='Dropout_2')(x)

            output = layers.Dense(number_classes,
                                  activation='softmax',
                                  name='Output')(x)

            # Creating the model:
            net = Model(input_layer, output)

            # Saving preprocessing function:
            preproc_function = None

    # Returning outputs:
    return net, preproc_function, images_size, transfer_flag

# ------------------------------------------------------------------------------
def train_net(net, epochs, TRAIN_iterator, VALIDATION_iterator,
                 optimizer='sgd', learning_rates=[], momentum_optimizer=[], loss_function='categorical_crossentropy',
                 metric='categorical_accuracy', flag_info=True, stopping=[],
                 id_finetuning=-1, result_name=[], flag_save_net=False):
    """
     This function trains a Neural Network using the method fit de Keras.

    --Inputs:

        net = The OBJECT NET created with KERAS or using the function 
              "create_cnn".

        epochs = A LIST with values number of epchos for training. If fine-tuning 
                 is applied it is expected more than one value.

        TRAIN_iterator = An ITERATOR OBJECT defined from a DATAFRAME containing 
                         the data of the TRAIN dataset.

        VALIDATION_iterator = An ITERATOR OBJECT defined from a DATAFRAME containing 
                              the data of the VALIDATION dataset.

        optimimizer = A STRING indicating the loss function to be used. Implemented: "adam", "sgd".

        learning_rates = A LIST with a values of the learning rates for training. If 
                         fine-tuning is applied it is expected more than one value.
                   
        momentum_optimizer = A LIST with a values of the momemtum for the optimizer. If fine-tuning 
                             is applied it is expected more than one value.

        loss_function = A STRING indicating the loss function to be used. We implement all 
                        the available in Keras.

        metric = A STRING indicating the metric function to be used. We implement all 
                 the available in Keras.

        flag_info = A BOOLEAN indicating if information about the  training 
                    should be shown.

        stopping = A number indicating the early-stopping to be applied. 0 = no 
                   early-stooping. A complete list of the implemented early-stoppings 
                   schemes can be consulted in "create_early_stopping".

        id_finetuning = A number indicating the layer from which the net is defreezing 
                        for fine-tuning. By default: -1 = all the not BN layer.

        result_name = A STRING indicating the full name for the result files (including graphs).
                      If empty none result is save.

        flag_save_net = A BOOLEAN indicating if the trained networks must be saved 
                        in H5 format.

    --Outputs:

        net = The OBJECT NET trained.

        history = The training history from the FIT method.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    from keras import optimizers
    import pandas as pd
    from graphs import show_history

    # --Training:

    # Cheking for validation data:
    if not VALIDATION_iterator:
        validation_data = None
    else:
        validation_data = VALIDATION_iterator

    # Inicialization:
    histories = [[] for id_stage in range(len(epochs))]

    # Showing info:
    print("------TRAINING")

    for id_stage in range(len(epochs)):

        # I analyze if I am in a stage other than the first:
        if (id_stage > 0):

            # Showing info:
            print("------FINE-TUNING")

            # As each net is changing the denomination of its names and block, we are going to put the limitation by layer index!
            stage_counter = 0

            # I unfreeze the net:
            for layer in net.layers:

                if (stage_counter >= id_finetuning):
                    # I have to train her!
                    # Analyze I am in a BATCH_NORMALIZATION layer that should not be adjusted in FINE-TUNING:
                    if '_bn' in layer.name:
                        # I'm in a BATCH_NORMALIZATION --> force it not to train:
                        layer.trainable = False
                    else:
                        # I'm not in a BATCH_NORMALIZATION --> unfreeze:
                        layer.trainable = True
                else:
                    # Freeze that layer:
                    layer.trainable = False

                # increment the counter:
                stage_counter = stage_counter + 1

        # I build the optimizer and compile the model:
        if optimizer == 'adam':
            objet_optimizer = optimizers.Adam(
                learning_rate=learning_rates[id_stage])
        else:
            objet_optimizer = optimizers.SGD(
                learning_rate=learning_rates[id_stage], momentum=momentum_optimizer[id_stage])

        # Compilation:
        net.compile(optimizer=objet_optimizer,
                    loss=loss_function,
                    metrics=metric)

        # Calling FIT:
        history = net.fit(x=TRAIN_iterator,
                            verbose=flag_info,
                            epochs=epochs[id_stage],
                            steps_per_epoch=None,
                            validation_data=validation_data,
                            validation_steps=None,
                            callbacks=stopping)
        # Save history:
        histories[id_stage] = pd.DataFrame(history.history)

    # Check if I want to save the performance results:
    if result_name:
        if flag_save_net:
            # Save NET:
            net.save(result_name + '.h5')

        # Save histories:
        hist_df = pd.concat(histories)
        hist_df.to_excel(result_name + '_history.xlsx')

        # Generating and saving ghraps:
        show_history(hist_df, metric, result_name + '_history.png')

    print("---------------------")

    # Returning outputs:
    return net, history

# ------------------------------------------------------------------------------
def eval_net(net, TEST_iterator, additional_measures, test_batch, negative_case_id=[]):
    """
     This function evaluates a neural network against the test set and returns 
     the validation measures.

    --Inputs:

        net = The net OBJECT with the net to evaluate generated using Keras.

        TEST_iterator = An ITERATOR OBJECT generated with one of the FLOW methods.

        additional_measures = A LIST of STRING with the names of additional measures 
                              to evaluate. The following are implemented: 
                              'FPR', 'FNR', 'Precision', 'Exhaustivity', 
                              'F1-Score', 'FPR+FNR'.

        negative_case_id = A LIST with the ids (the numbers) of the negative case 
                           in the TRAINING and in the TEST. By default it is empty.

     --Outputs:

        ypred = The net object, with the trained net.

        measure_vector = A LIST with the values of the similarity measures analyzed. 
                         Element [0] will be accuracy, element [1] will be error, 
                         and then there will be the values of the additional
                         measures considering the order indicated.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    import numpy as np
    from sklearn.metrics import confusion_matrix, accuracy_score

    # Inicializing:
    measure_vector = [[] for id_measure in range(len(additional_measures)+2)]
    
    # I evaluate the net for the TEST set:
    vector_pred = net.predict(x=TEST_iterator,
                          batch_size=test_batch,
                          verbose=0)

    # Defining id of labels:
    Y_1 = np.argmax(vector_pred, axis=-1)

    # Obtaining id of labels in test:
    Y_2 = np.array(TEST_iterator.classes)

    # Analizing if I am in BINARY classification:
    if (len(negative_case_id) != 0):
        # I am in binary classification, I analyze the mapping condition (so that the label 0 is the normal one):
        if (negative_case_id[0] == 1) & (negative_case_id[1] == 1):
            # I must invert both:
            ypred = [abs(value-1) for value in Y_1]
            yvalidation = [abs(value-1) for value in Y_2]
        elif (negative_case_id[0] == 1) & (negative_case_id[1] == 0):
            # I must invert the obtained:
            ypred = [abs(value-1) for value in Y_1]
            yvalidation = Y_2
        elif (negative_case_id[0] == 0) & (negative_case_id[1] == 1):
            # I must invert the gold-standard:
            ypred = Y_1
            yvalidation = [abs(value-1) for value in Y_2]
        else:
            ypred = Y_1
            yvalidation = Y_2
    else:
        # I'm not into the binary case, so I'm copying it directly (yes I'll take into account the mapping for the names)
        ypred = Y_1
        yvalidation = Y_2

    # I calculate ACCURACY, error and CONFUSION MATRIX:
    accuracy = accuracy_score(y_true=yvalidation, y_pred=ypred)
    error = 1 - accuracy
    confussion_matrix = confusion_matrix(y_true=yvalidation, y_pred=ypred)

    # Guardo:
    measure_vector[0] = accuracy
    measure_vector[1] = error

    # Analyze if I have to find something else:
    aux_id = 2
    for measure in additional_measures:
        # Analyze the measure to find:
        if (measure == 'FPR'):
            # I must compute "FPR":
            FPR = confussion_matrix[0, 1] / \
                (confussion_matrix[0, 1] + confussion_matrix[0, 0])
            measure_vector[aux_id] = FPR
            aux_id = aux_id + 1
        elif (measure == 'FNR'):
            # I must compute "FNR":
            FNR = confussion_matrix[1, 0] / \
                (confussion_matrix[1, 0] + confussion_matrix[1, 1])
            measure_vector[aux_id] = FNR
            aux_id = aux_id + 1
        elif (measure == 'Precision'):
            # I must compute "Precision":
            Precision = confussion_matrix[1, 1] / \
                (confussion_matrix[0, 1] + confussion_matrix[1, 1])
            measure_vector[aux_id] = Precision
            aux_id = aux_id + 1
        elif (measure == 'Exhaustivity'):
            # I must compute "Exhaustividad":
            Exhaustividad = confussion_matrix[1, 1] / \
                (confussion_matrix[1, 0] + confussion_matrix[1, 1])
            measure_vector[aux_id] = Exhaustividad
            aux_id = aux_id + 1
        elif (measure == 'F1-Score'):
            # I must compute "F1-Score":
            Precision = confussion_matrix[1, 1] / \
                (confussion_matrix[0, 1] + confussion_matrix[1, 1])
            Exhaustividad = confussion_matrix[1, 1] / \
                (confussion_matrix[1, 0] + confussion_matrix[1, 1])
            F1_Score = 2 * Precision * Exhaustividad / \
                (Precision + Exhaustividad)
            measure_vector[aux_id] = F1_Score
            aux_id = aux_id + 1
        elif (measure == 'FPR+FNR'):
            # I must compute "FPR+FNR":
            FPR = confussion_matrix[0, 1] / \
                (confussion_matrix[0, 1] + confussion_matrix[0, 0])
            FNR = confussion_matrix[1, 0] / \
                (confussion_matrix[1, 0] + confussion_matrix[1, 1])
            SUMA = FNR + FPR
            measure_vector[aux_id] = SUMA
            aux_id = aux_id + 1

    # Returning outputs:
    return ypred, measure_vector, confussion_matrix

# ------------------------------------------------------------------------------
def create_early_stopping(early_stopping_type):
    """
     This function generates a specific EarlyStopping function.

    --Inputs:
       
        early_stopping_type = A number indicating the type of Early Stopping 
                              to define: 
            0 --> Early stopping is not applied. 
            1 --> Early stopping is applied with monitor "val_accuracy" 
                  and patience "15".
            2 --> Early stopping is applied with monitor "val_accuracy" 
                  and patience "25".

    --Outputs:

        stopping = The OBJECT stopping function. An empty list is returned 
                   if early_stopping_type = 0.

        flag_stopping = A BOOLEAN indicating whether or not EarlyStopping 
                        is used. "False" is returned if 
                        early_stopping_type = 0.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    from keras.callbacks import EarlyStopping

    # Inicializing:
    stopping = []

    # I analyze which one I have to build, based on predetermined criteria:
    if (early_stopping_type == 1):
        stopping = EarlyStopping(
            monitor="val_accuracy",  # variable to monitor
            min_delta=0,
            patience=15,  # Number of epochs without improvement to wait
            verbose=0,
            mode="max",  # criteria {"auto", "min", "max"}.
            baseline=None,
            restore_best_weights=True,
        )
    elif (early_stopping_type == 2):
        stopping = EarlyStopping(
            monitor="val_accuracy",  # variable to monitor
            min_delta=0,
            patience=25,  # Number of epochs without improvement to wait
            verbose=0,
            mode="max",  # criteria {"auto", "min", "max"}.
            baseline=None,
            restore_best_weights=True,
        )

    # Analyze the flag condition:
    if (early_stopping_type != 0):
        flag_stopping = True
    else:
        flag_stopping = False

    # Returning outputs:
    return stopping, flag_stopping

# ------------------------------------------------------------------------------
def get_conv_indexes(model):
    """
     This function goes through the layers and saves the index of the last 
     conv layer of each block.

    --Inputs:
       
        model = A Neural Network OBJECT generated using Keras.

    --Outputs:

        ixs = A LIST with the numbers of the convolutional layers in "model".

        conv_layers_names = A LIST with the names of the convolutional layers 
                            in "model".

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries:
    import sys

    # Inicializing:
    ixs = []
    conv_layers_names = []

    # Loop:
    len_layers = len(model.layers)
    for i in range(len_layers):
        # Extracting the layer name:
        layer = model.layers[i]
        if "conv" in layer.name and "out" in layer.name:
            try:
                next_layer = model.layers[i + 1]
                ixs.append(i)
                conv_layers_names.append(layer.name)
            except:
                # if by chance the last layer is convolutional (has no sent)
                # but it can happen on a homemade model, so I made this brake.
                print("The last layer is a convolutional layer, adjust your model")
                sys.exit()

    # Returning outputs:
    return(ixs, conv_layers_names)

# ------------------------------------------------------------------------------
def compute_loss(input_image, filter_index, feature_extractor):
    """
     This function computes the "loss" as the mean of the activation of a 
     specific filter in our target layer. To avoid border effects, 
     we exclude border pixels.

    --Inputs:
       
        input_image = A IMAGE to be used as input for computing loss.

        filter_index = The index of the filter in the output layer to be 
                       analyzed.

        feature_extractor = A Keras model set as feature extractor (a output 
                            is defined in a convolutional layer)

    --Outputs:

        filter_activation_reduced = A TENSOR with the value of loss computed 
                                    as the mean of elements across 
                                    dimensions of a tensor. .

    *Extrated from Keras: 
            https://keras.io/examples/vision/visualizing_what_convnets_learn/.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  
    
    # Libraries:
    import tensorflow as tf

    # Extract activation:
    activation = feature_extractor(input_image)

    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]

    # Reducing tensor:
    loss_computed = tf.reduce_mean(filter_activation)

    # Returning outputs:
    return loss_computed

# ------------------------------------------------------------------------------
def load_saved_model(model_name):
    """
     This function load a Neural Network from a H5 file.

    --Inputs:
       
        model_name = The full name of the H5 file to be loaded.

    --Outputs:

        model = A Neural Network OBJECT with the content of the H5 file.

        error_flag = A BOOLEAN indicating if an error ocurred during 
                     loading.

    """
    
    # Author: Diego Comas 
    # Revised by: - 

    # Libraries:
    from keras.models import load_model
    
    # Reading the file:
    try:
        model = load_model(model_name)
        error_flag = False
    except:
        model = []
        error_flag = True    
    
    # Returning outputs:
    return(model,error_flag)

# ------------------------------------------------------------------------------
def evaluate_image(image_name, model):
    """
     This function evaluates a Neural Network for a specific image.

    --Inputs:
       
        image_name = The full name of the image to be evaluated.

        model = A Neural Network OBJECT of Keras. It is supposed each required 
                output is previusly set.

    --Outputs:

        feature_maps = A TENSOR OBJECT with the feature maps.

        label_assigned = The label assigned to the image during the evaluation.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  
 
    # Libraries
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np

    # Get the size of the first layer (the size of the image at the net input):
    lens = model.layers[0].input_shape
    width = lens[0][1]
    heigth = lens[0][2]
    
    # Cheking if the image is GRAY of RGB:
    if lens[0][3]==1:
        # It is GRAY:
        img = load_img(image_name , grayscale=True, target_size=(width,heigth))
    else:
        # It is RGB:
        img = load_img(image_name , target_size=(width,heigth))
    
    # Convert the image to an array:
    img = img_to_array(img)

    # Expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(img, axis=0)
    
    # Get feature maps for all indexs:
    feature_maps = model.predict(img)
    
    # Obtaining the label assigned to the image:
    label_assigned = np.argmax(feature_maps[-1], axis=-1)

    # Deleting the label:
    del feature_maps[-1]

    # Returning outputs:
    return(feature_maps, label_assigned)

# ------------------------------------------------------------------------------
def set_net_outputs(model, ixs):
    """
    This function sets specific indexes to as outputs.

    --Inputs:
       
        model = A Neural Network OBJECT generated using Keras.

        ixs = A LIST with the numbers of the layers to set as outputs.

    --Outputs:

        model_changed = The Neural Network OBJECT with the changes in the 
                        outputs.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries:
    from keras.models import Model

    # Adding the final output with the label as part of the outputs in query:
    ixs.append(-1) 

    # Loop:
    outputs = [model.layers[i].output for i in ixs]

    # Final set:
    model_changed = Model(inputs=model.inputs, outputs=outputs)

    # Returning outputs:
    return(model_changed)

# ------------------------------------------------------------------------------
def implementing_hold_out(data, labels, id_files, id_labels, dict_params, image_folder, additional_measures, experiment_folder=[]):
    """
     This function implements hold-out on a CNN defined by means "dict_params".

    --Inputs:
       
        data = A DATAFRAME containing the data to be used for training and validation.

        labels = A LIST with the names of the labels to be used.

        id_files = A STRING with the name of the columns in "data" containing the names 
                   of the files. Short names is used or path relative to "image_folder" 
                   is required.

        id_labels = A STRING with the name of the columns in "data" containing the labels 
                    of the files.

        dict_params = A DICTIONARY containing all the parameters for the test using 
                      the next keys (the specified are the required):
                    "experiment_name" --> A STRING with the name for the experiment.
                    "net_used" --> A STRING with the name of the network to be used. 
                                   A complete list of the implemented models can be 
                                   consulted in "create_cnn".
                    "flag_info" --> A BOOLEAN indicating if information about the 
                                    training should be shown.
                    "flag_saving_net" --> A BOOLEAN indicating if the trained networks 
                                          must be saved in H5 format.
                    "classifier" --> A LIST with a number indicating the classification 
                                     scheme to be used. A complete list of the implemented 
                                     classification schemes can be consulted in "create_cnn".
                    "holdout"  --> A LIST with the proportions to be used in HOLD-OUT 
                                   [TRAIN, VALIDATION, TEST]. If validation is not used, 
                                   then complete with 0.
                    "augmentation" --> A LIST with a number indicating the augmentation to 
                                       be applied. 0 = none. A complete list of the implemented 
                                       early-stoppings schemes can be consulted in "train_net".
                    "extern_balancing" --> A STRING indicating the way balancing is performed
                                           in before partitioning. "min" = matching to the
                                           minority class. 
                                           "max" = matching to the mayority class. "no" = none.
                    "train_balancing" --> A STRING indicating the way balancing is performed 
                                          in TRAINING. "min" = matching to the minority class.
                                          "max" = matching to the mayority class. "no" = none.
                    "maximum_samples"  --> A LIST with the value of maximum samples per 
                                           class. 0 = no limitation. 
                    "epochs" --> A LIST with values number of epchos for training. 
                                 If fine-tuning is applied it is expected more than one 
                                 value.
                    "color" --> A STRING indicating the color mode to be used: "grayscale", 
                                "rgb", "rgba".
                    "iterations" --> A LIST with the number of iterations of TRAINING-TEST.
                    "optimimizer" --> A STRING indicating the loss function to be used. 
                                      A list of the optimimizers implemented can be 
                                      consulted in "train_net".
                    "finetuning" --> A LIST with a number indicating the layers from which the 
                                     net is defreezed.
                    "learning_rates" --> A LIST with a values of the learning rates for training. 
                                         If fine-tuning is applied it is expected more than one 
                                         value.
                    "momentum_optimizer" --> A LIST with a values of the momemtum for the optimizer. 
                                             If fine-tuning is applied it is expected more than one 
                                             value.
                    "loss_function" --> A STRING indicating the loss function to be used. 
                                          A list of the functions implemented can be 
                                          consulted in "train_net". 
                    "metric_function" --> A STRING indicating the metric function to be used. 
                                          A list of the functions implemented can be 
                                          consulted in "train_net". 
                    "train_batch" --> A LIST containing the size of the BATCH for training.
                    "validation_batch" --> A LIST containing the size of the BATCH for 
                                           validation.
                    "test_batch" --> A LIST containing the size of the BATCH for test.
                    "early_stopping" --> A LIST with a number indicating the early-stopping 
                                         to be applied. 0 = no early-stooping. A complete 
                                         list of the implemented early-stoppings schemes can 
                                         be consulted in "create_early_stopping".

        image_folder = A STRING with the path to the files. It is a general folder which 
                       can contain additional folders included in the information of 
                       "id_files".

        additional_measures = A LIST with the names of the validation measures to be 
                              used, in addition to ACC and error. The implemented 
                              validation measures should be consulted in the 
                              function "eval_net".

        experiment_folder = A STRING with the path for saving results. If empty no results 
                            will be saved.
    
    --Outputs:

        time_counters = A LIST with the times (in seconds) used for the implementation.
        
        matrix_measures = A NUMPY ARRAY with the resulting measures for the implementation.

    """

    # Author: Diego Comas 
    # Revised by: - 
    
    # Libraries:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from dataframes import sampling_df, balance_df, hold_out_df, create_iter_df
    from settings import toolbox_version
    from texts import list_into_str, save_list_to_text
    from files import create_folder

    # Define list of measures:
    number_of_measures = len(additional_measures)+2

    # Define the number of iterations:
    number_of_iterations = dict_params['iterations'][0]

    # Detect number of classes:
    number_classes = len(labels)
    
    # Cheking if we want to save results:
    if experiment_folder:

        # Trying to create the folder "results" inside "experiments":
        result_folder = os.path.join(experiment_folder, 'Results')
        create_folder(result_folder)

        # Inicialization EXCEL file:
        excel_file_name = result_folder + '/Validation.xlsx'

        writer_final = pd.ExcelWriter(excel_file_name)

        # Inicialize EXCEL content:
        columns_names = []

        columns_names.append('Measure')
        for iter in range(number_of_iterations):
            columns_names.append('iteration #' + format(iter+1))
        columns_names.append('MEAN')
        columns_names.append('STD')

        # Inicialize matrixes:
        matrix_measures = np.zeros((number_of_measures, number_of_iterations+2))
        confussion_matrix = np.zeros((number_classes, number_classes, number_of_iterations))

    # Inicialization of TIMES vector:
    time_counters = [[] for iter in range(number_of_iterations+1)]

    # Making STOPPING function:
    function_stopping, flag_stopping = create_early_stopping(dict_params['early_stopping'])

    # Iterations:
    for iter in range(number_of_iterations):

        # Show message:
        print("*************************************************************")
        print("***" + dict_params['experiment_name'])
        print("*****Iteration #", iter + 1, " de ", number_of_iterations,  "...")

        # Start counting time:
        startTime = datetime.now()

        # Checking if first iter:
        if (iter == 0):
            # Save starting time:
            start_Time = startTime

            # Define name specific for results in this iter:
            aux_result_name = result_folder + \
                                    '/Data_iter_' + str(iter+1)

        # Inicialization of the CNN:
        net, preprocess_input, images_size, flag_transfer = create_cnn(dict_params['net_used'], number_classes, dict_params['classifier'][0])

        # Applying max samples if needed:
        if (dict_params['maximum_samples'] != 0):
            data_filtered = sampling_df(data, id_labels, dict_params['maximum_samples']) 
        else:
            data_filtered = data

        # Partition and balancing:
        if dict_params['extern_balancing'] != 'no':
            data_balanced = balance_df(data_filtered, dict_params['extern_balancing'], id_files, id_labels)
        else:
            data_balanced = data_filtered

        # Making partition::
        TRAIN_Data, VAL_Data, TEST_Data = hold_out_df(data_balanced, dict_params['holdout'],
                                                                            aux_result_name,
                                                                            id_files, id_labels,
                                                                            dict_params['train_balancing'])
        # Checking VALIDATION data (if it is not defined, then I will use the test data for monitoring training):
        if (dict_params['holdout'][1] == 0):
            VAL_Data = TEST_Data

        # -TRAINING:
        # Iterators:
        TRAIN_iterator = create_iter_df(TRAIN_Data, preprocess_input, image_folder, images_size,
                                                        dict_params['train_batch'][0], dict_params['augmentation'][0], dict_params['color'], id_files,
                                                        id_labels, labels, Condicion_Shuffle=True)

        VALIDATION_iterator = create_iter_df(VAL_Data, preprocess_input, image_folder, images_size,
                                                                dict_params['validation_batch'][0], 0, dict_params['color'], id_files,
                                                                id_labels, labels, Condicion_Shuffle=False)

        TEST_iterator = create_iter_df(TEST_Data, preprocess_input, image_folder, images_size,
                                                        dict_params['test_batch'][0], 0, dict_params['color'], id_files,
                                                        id_labels, labels, Condicion_Shuffle=False)

        # Train:
        net, history = train_net(net, dict_params['epochs'], TRAIN_iterator, VALIDATION_iterator,
                                 dict_params['optimimizer'], dict_params['learning_rates'], dict_params['momentum_optimizer'], dict_params['loss_function'],
                                 dict_params['metric_function'], dict_params['flag_info'], function_stopping, dict_params['finetuning'],
                                        aux_result_name, dict_params['flag_saving_net'])

         # -Evaluation of the net:
         # Get the class mapping you did in TRAIN and in TEST:
         # (note that regardless of contributing the list of classes to FLOW_FROM_DATAFRAME,
         # once it finds the distinct tags, it sorts them alphanumerically --> the condition
         # of normality will not necessarily have the value "0" in the labels and the value will have to be inverted
         # logic to find the validation measures).
        id_Negative_Case_TRAIN = TRAIN_iterator.class_indices[labels[0]]
        id_Negative_Case_TEST = TEST_iterator.class_indices[labels[0]]

        # Evaluate TEST performance:
        ypred, measures_values, confussion_matrix = eval_net(net, TEST_iterator,
                                                                additional_measures,dict_params['test_batch'],
                                                                negative_case_id=[id_Negative_Case_TRAIN, id_Negative_Case_TEST])

        # Save results:
        matrix_measures[:, iter] = measures_values

        # Results matrixes:
        matrixes = pd.DataFrame(confussion_matrix)
        matrixes.to_excel(writer_final, sheet_name='Confusion_Iter_' +
                            str(iter+1), float_format="%.4f", index=None)

        # Save final time:
        endTime = datetime.now()

        # Save time counters:
        time_counters[iter] = str(endTime - startTime)

        print("\nAccuracy Test, iter ", iter + 1, " of ",
                number_of_iterations,  ": %.1f%%" % (100.0 * measures_values[0]))

    print("*************************************************************")

    # Computing total time (on all the iterations):
    time_counters[number_of_iterations] = str(endTime - start_Time)

    # Defining MEAN and STD:
    matrix_measures[:, -
                        2] = (np.mean(matrix_measures[:, :number_of_iterations], axis=1))
    matrix_measures[:, -
                        1] = (np.std(matrix_measures[:, :number_of_iterations], axis=1))
    
    # ---Saving results if required:
    if experiment_folder: 

        # Defining dictionary:
        Data_EXCEL = {
            columns_names[id_column+1]: matrix_measures[:, id_column] for id_column in range(number_of_iterations+2)}
        Data_EXCEL['Measure'] = ['ACC', 'error'] + additional_measures
        # Defining dataframe:
        array = pd.DataFrame(Data_EXCEL, columns=columns_names)
        array.to_excel(writer_final, sheet_name='Results',
                            float_format="%.4f", index=None)
        # Saving:
        writer_final.save()

        # ---Copying TXT file with the parameters:
        dict_params['labels_used'] = list_into_str(labels)
        
        # Inicialization:
        text_params_file_final = [
            [] for iter in range(len(list_keys)+3)]
        # Cargo el código de versión:
        text_params_file_final[0] = 'script_version: ' + dict_params['script_version']
        text_params_file_final[1] = 'toolbox_version: ' + toolbox_version()
        text_params_file_final[2] = 'net_used: ' + dict_params['net_used']
        text_params_file_final[3] = 'labels: ' + dict_params['labels_used']

        # Remove already written parameters:
        del (dict_params['net_used'])
        del (dict_params['script_version'])
        list_keys = list(dict_params)

        # Copying rest of parameters:
        list_keys = list(dict_params)

        for iter in range(len(list_keys)-1):
            if isinstance(dict_params[list_keys[iter]], list):
                string = list_into_str(dict_params[list_keys[iter]])
            elif isinstance(dict_params[list_keys[iter]], bool):
                string = str(dict_params[list_keys[iter]])
            else:
                string = dict_params[list_keys[iter]]
            text_params_file_final[iter + 4] = (list_keys[iter]+': ' + string).rstrip()

        # Saving times in the parameters file:
        string = list_into_str(time_counters)
        text_params_file_final[len(list_keys)+3] = 'time_used: ' + string
        
        # Saving file:
        save_list_to_text(text_params_file_final, experiment_folder + '/' + 'Test_Parameters.txt')

    # Returning outputs:
    return time_counters, matrix_measures

# ------------------------------------------------------------------------------
def evaluate_class_models(data_df, dic_par, labels, id_files, id_labels, image_folder, result_folder):
    """
     This function implements the evaluation of classification models from a DATAFRAME.

    --Inputs:
       
        data_df = A DATAFRAME containing the data to be used for training and validation.

        dict_params = A DICTIONARY containing all the parameters for the test using 
        the next keys:

        labels = A LIST with the names of the labels to be used.

        id_files = A STRING with the name of the columns in "data" containing the names 
                   of the files. Short names is used or path relative to "image_folder" 
                   is required.

        id_labels = A STRING with the name of the columns in "data" containing the labels 
                    of the files.

        dict_params = A DICTIONARY containing all the parameters for the test using 
                      the next keys:
                    "experiment_name" --> A STRING with the name for the experiment.
                    "nets" --> A LIST containing STRINGS with the names of the models 
                               to be tested. A complete list of the implemented models 
                               can be consulted in "create_cnn".
                    "flag_info" --> A BOOLEAN indicating if information about the 
                                    training should be shown.
                    "flag_saving_net" --> A BOOLEAN indicating if the trained networks 
                                          must be saved in H5 format.
                    "classifier" --> A LIST with a number indicating the classification 
                                     scheme to be used. A complete list of the implemented 
                                     classification schemes can be consulted in "create_cnn".
                    "holdout"  --> A LIST with the proportions to be used in HOLD-OUT 
                                   [TRAIN, VALIDATION, TEST]. If validation is not used, 
                                   then complete with 0.
                    "augmentation" --> A LIST with a number indicating the augmentation to 
                                       be applied. 0 = none. A complete list of the implemented 
                                       early-stoppings schemes can be consulted in "train_net".
                    "extern_balancing" --> A STRING indicating the way balancing is performed
                                           in before partitioning. "min" = matching to the
                                           minority class. 
                                           "max" = matching to the mayority class. "no" = none.
                    "train_balancing" --> A STRING indicating the way balancing is performed 
                                          in TRAINING. "min" = matching to the minority class.
                                          "max" = matching to the mayority class. "no" = none.
                    "maximum_samples"  --> A LIST with the value of maximum samples per 
                                           class. 0 = no limitation. 
                    "epochs" --> A LIST with values number of epchos for training. 
                                 If fine-tuning is applied it is expected more than one 
                                 value.
                    "color" --> A STRING indicating the color mode to be used: "grayscale", 
                                "rgb", "rgba".
                    "iterations" --> A LIST with the number of iterations of TRAINING-TEST.
                    "optimimizer" --> A STRING indicating the loss function to be used. 
                                      A list of the optimimizers implemented can be 
                                      consulted in "train_net".
                    "finetuning" --> A LIST with a number indicating the layer from which it 
                                     freezes. It should inform a initial layer per model. 
                    "learning_rates" --> A LIST with a values of the learning rates for training. 
                                         If fine-tuning is applied it is expected more than one 
                                         value.
                    "momentum_optimizer" --> A LIST with a values of the momemtum for the optimizer. 
                                             If fine-tuning is applied it is expected more than one 
                                             value.
                    "loss_function" --> A STRING indicating the loss function to be used. 
                                          A list of the functions implemented can be 
                                          consulted in "train_net". 
                    "metric_function" --> A STRING indicating the metric function to be used. 
                                          A list of the functions implemented can be 
                                          consulted in "train_net". 
                    "train_batch" --> A LIST containing the size of the BATCH for training.
                    "validation_batch" --> A LIST containing the size of the BATCH for 
                                           validation.
                    "test_batch" --> A LIST containing the size of the BATCH for test.
                    "early_stopping" --> A LIST with a number indicating the early-stopping 
                                         to be applied. 0 = no early-stooping. A complete 
                                         list of the implemented early-stoppings schemes can 
                                         be consulted in "create_early_stopping".

        image_folder = A STRING with the path to the files. It is a general folder which 
                       can contain additional folders included in the information of 
                       "id_files".

        result_folder = A STRING with the path for saving results. If empty no results 
                        will be saved.
    
    --Outputs:

        time_counters_total = A LIST containing the times (in seconds) used for the 
                              implementation. Each elements corresponds to a model.
        
        matrix_measures = A LIST containing NUMPY ARRAYS with the resulting measures 
                          for the implementation. Each elements corresponds to a model.

    """
    # Author: Diego Comas 
    # Revised by: - 

    # Libraries:
    from datetime import date
    from files import create_folder

    # Detect number of classes:
    number_classes = len(labels)

    # Defines proper validation measures:
    if (number_classes==2):
        additional_measures = ['FPR', 'FNR', 'Precision', 'Exhaustivity', 'F1-Score', 'FPR+FNR']
    else:
        additional_measures = []
    
    # Defines part of the experiment name:
    if (number_classes==2):
        prefix_name_1 = labels[0] + '_vs_' + labels[1]
    else: 
        prefix_name_1 = 'multilabel'

    # Define experiment date:
    date = date.today()
    formated_date = format(date.day) + '-' + \
        format(date.month) + '-' + format(date.year)
    
    # Initializing counter:
    model_counter = -1

    # Inicializing the measures:
    matrix_measures_total = []
    time_counters_total = []

    # Iterate on the models to evaluate:
    for model in dic_par['nets']:

        # Incrementating counter:
        model_counter = model_counter + 1

        # Define the name for the experiments (the folder where we save all the results):
        experiment_name = formated_date + '_' + dic_par['experiment_name'] + '_' + dic_par['case'] + '_' + prefix_name_1 + '_' + model

        # Trying to create the folder "experiments" inside "results":
        if result_folder:
            experiment_folder = os.path.join(result_folder, experiment_name)
            create_folder(experiment_folder)
            
        # Checking if we defined "id_finetuning":
        try:
            id_finetuning = dic_par['finetuning'][model_counter]
        except:
            id_finetuning = -1

        # Copying dictionary with parameters for this particular test:
        dict_params=dic_par.copy()
        
        # Remove not required parameters:
        del (dict_params['nets'])

        # Changing keys accordantly:
        dict_params['net_used'] = model
        dict_params['finetuning'] = [id_finetuning]
        
        # --ITERATION TRAINING-TEST:
        time_counters, matrix_measures = implementing_hold_out(data_df, labels, id_files, id_labels, dict_params, image_folder, additional_measures, experiment_folder)

        # Download the values of times and measures:
        time_counters_total.append(time_counters)
        matrix_measures_total.append(matrix_measures)

    # Returning outputs:
    return time_counters_total, matrix_measures_total

# ------------------------------------------------------------------------------
def reshape_input_model(base_model, new_shape, flag_print):
    """
    This function allows to change the input shape of a model.

    --Inputs:
       
        base_model = A Neural Network OBJECT generated using Keras.

        new_shape = A LIST with the 3 numbers of the new shape 
                    [number_rows, number_columns, number_chanels].

    --Outputs:

        new_model = The Neural Network OBJECT with the changes in the 
                    input shape.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    from keras.models import Model
    
    # I obtain the strcuture of the model:
    model_config = base_model.get_config()

    # Changing the input shape accordintly:
    model_config['layers'][0]['config']['batch_input_shape'] = (None, new_shape[0], new_shape[1], new_shape[2])

    # Cheating the new model with all the of the previous:
    new_model = Model.from_config(model_config)
    
    # Checking if new shape should be show
    if flag_print:
        new_model.summary()

    # Returning outputs:
    return new_model

# ------------------------------------------------------------------------------
def rgb_to_gray(inputs):
    """
    This function create a layer for converting RGB to GRAYSCALE using 
    [0.333, 0.333, 0.333] as weights.

    --Inputs:
       
        inputs = The inputs to the layer.

    --Outputs:

        gray_output = The output of the conversion layer.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    import tensorflow as tf

    # Obtaing the output: 
    gray_output = tf.reduce_sum(inputs * tf.constant([0.333, 0.333, 0.333], shape=(1, 1, 1, 3)), axis=-1, keepdims=True)

    # It is implemented the typical conversion from RGB to GRAY SCALE using [0.333, 0.333, 0.333] as weights. 
    return gray_output

# ------------------------------------------------------------------------------