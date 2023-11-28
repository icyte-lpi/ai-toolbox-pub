###############################################################################
#                         ICyTE-LPI-Deep Toobox                               #
# module name:                                                                #
#     LPI_deep_Files                                                          #
#                                                                             #
# module description:                                                         #
#     This module contains functions for hardware settings.                   #
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
#     module: 1.0 - 2023-04-25                                                #
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
#    1.0   04/25/2023  Diego Comas            First version.
#                      Agustín Amalfitano 
#                      Franco Ercoli
#
# ******************************************************************************

# ------------------------------LIST OF FUNCTIONS-------------------------------
# Functions              |   Date    |    Authors          |   Description
# ------------------------------------------------------------------------------
#
# toolbox_version          04/25/2023   Diego Comas          Return toolbox 
#                                                            version. 
#
# memory_limit             04/25/2023   Diego Comas          Limit the memory to   
#                                                            be used by GPUs.
#
# ------------------------------------------------------------------------------

# --------------------------IMPORTS---------------------------------------------
# Reserved.

# ------------------------------------------------------------------------------
def toolbox_version():
    """
     This function return the version of the ICyTE-LPI-Deep toobox.

    --Inputs:
       
        None.
    
    --Outputs:

        toolbox_version = A STRING containing the version of the toolbox.

    """

    toolbox_version = '1.0'

    # Returning outputs:
    return toolbox_version

# ------------------------------------------------------------------------------
def memory_limit(limit):
    """
     This function limit the memory to be used by GPUs.

    --Inputs:
       
        limit = A number expressed in GB which is the required limit.
    
    --Outputs:

        flag_error = A BOOLEAN flag defining the error status.

        gpus = A LIST containing the names of the physical GPUs in the PC.

    """
    # So that it doesn't take up all the memory of the server, but uses a limit
     # particular.
     # (extracted from the official TensorFlow documentation)
     # IMPORTANT: Keep in mind that you should strongly reduce the batch sizes!

    # Libraries:
    import tensorflow as tf

    # We pass the limit to MB
    maximum = limit * 1024  
    # List of existing physical devices:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # We limit the memory to be used:
        try:
            tf.config.set_logical_device_configuration(gpus[0],
                                                    [tf.config.LogicalDeviceConfiguration(memory_limit=maximum)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            # Save flag:
            flag_error = True
        except RuntimeError as e:
            # If an error occurred during setting
            # (for instance if the device was created after inicializing the GPU):
            print(e)
            # Save flag:
            flag_error = True
    else:
        # Save flag:
        flag_error = True

    # Returning outputs:
    return flag_error, gpus

# ------------------------------------------------------------------------------