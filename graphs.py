###############################################################################
#                        ICyTE - LPI - AI Toolbox                             #
# module name:                                                                #
#     graphs                                                                  #
#                                                                             #
# module description:                                                         #
#     This module contains functions for doing graphs.                        #
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
def show_history(history, metric_name, file_name=0):
    """
     This function shows the history graphs from a training history.

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
    
    # Author: Diego Comas 
    # Revised by: -

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