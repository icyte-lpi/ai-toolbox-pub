###############################################################################
#                       ICyTE - LPI - AI Toolbox                              #
# module name:                                                                #
#     settings                                                                #
#                                                                             #
# module description:                                                         #
#     This module contains functions for hardware settings.                   #
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
# Reserved.

# ------------------------------------------------------------------------------
def toolbox_version(repo_path):
    """
     This function return the version of the ICyTE - LPI - AI Toolbox informing 
     the HASH of the repository.

    --Inputs:
       
        repo_path = A STRING containing the path to the toolbox repo.
    
    --Outputs:

        toolbox_hash = A STRING containing the version of the toolbox.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    import git
    
    # We obtain the version as reposity hash:
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        toolbox_hash = repo.head.object.hexsha[:7]
    except git.InvalidGitRepositoryError:
        toolbox_hash = "No Git Repo"

    # Returning outputs:
    return toolbox_hash

# ------------------------------------------------------------------------------
def memory_limit(limit):
    """
     This function limits the memory to be used by GPUs.

    --Inputs:
       
        limit = A number expressed in GB which is the required limit.
    
    --Outputs:

        flag_error = A BOOLEAN flag defining the error status.

        gpus = A LIST containing the names of the physical GPUs in the PC.

    """

    # Author: Diego Comas 
    # Revised by: -

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
            flag_error = False
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