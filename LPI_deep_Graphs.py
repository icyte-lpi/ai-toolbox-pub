###############################################################################
#                         ICyTE-LPI-Deep Toobox                               #
# module name:                                                                #
#     LPI_deep_Graphs                                                         #
#                                                                             #
# module description:                                                         #
#     This module contains all the functions developed for doing graphs.      #
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
#     module: 1.0 - 2023-04-05                                                #
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
#    1.0   04/05/2023  Diego Comas            First version.
#                      Agustín Amalfitano 
#                      Franco Ercoli
#
# ******************************************************************************

# ------------------------------LIST OF FUNCTIONS-------------------------------
# Functions              |   Date    |    Authors          |   Description
# ------------------------------------------------------------------------------
# show_history             04/05/2023   Diego Comas /        Show history of
#                                       Agustín Amalfitano   training.
#
# ------------------------------------------------------------------------------

# --------------------------IMPORTS---------------------------------------------
# Reserved.

# ------------------------------------------------------------------------------
def show_history(history, metric_name, file_name=0):
    """
     This function show the history graphs from a training history.

    --Inputs:
       
        history = A object with the training history in the format of the method 
                  fit of KERAS.

        metric_name = A STRING with the name of the metric to be used, in 
                      addition to "loss".

        file_name = A STRING with the full name for saving the figure and graphs. 
                    If it is 0, then no saving is performed.
    
    --Outputs:

        None.

    """

    # Libraries:
    import matplotlib.pyplot as pyplot

    # Reading training parameters:
    epochs = range(1, len(history["loss"]) + 1)
    pyplot.figure(1, figsize=(20, 16))
    pyplot.clf()

    # Graph of "loss":
    pyplot.subplot(221)
    pyplot.plot(epochs, history["loss"], "m", label="Training loss")
    pyplot.plot(epochs, history["val_loss"], "c", label="Validation loss")
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Loss")
    pyplot.legend()

    # Graph of the metric informed in "metric_name":
    pyplot.subplot(222)
    pyplot.plot(epochs, history[metric_name], "m",  label="Training " + metric_name)
    pyplot.plot(epochs, history["val_" + metric_name], "c", label="Validation" + metric_name)
    pyplot.xlabel("Epochs")
    pyplot.ylabel(metric_name)
    pyplot.legend()

    # We save the figure if necessary:
    if file_name!=0:
        pyplot.savefig(file_name)

# ------------------------------------------------------------------------------