###############################################################################
#                       ICyTE - LPI - AI Toolbox                              #
# module name:                                                                #
#     texts                                                                   #
#                                                                             #
# module description:                                                         #
#     This module contains functions for managing text files and text data.   #
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
def save_list_to_text(lines_list, file_name):
    """
     This function saves a LIST in a text file.

    --Inputs:
       
        lines_list = A LIST containing the text to be save in a file.
        
        file_name = A STRING with the full name of the file to be save.
    
    --Outputs:

        flag_error = A BOOLEAN flag defining the error status.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Try to save
    try:
        with open(file_name, 'w') as f:
            for line in lines_list:
                f.write(line)
                f.write('\n')
        # Lower error flag:
        flag_error = False
    
    except:
        # Raise error flag:
        flag_error = True

    # Returning outputs:
    return flag_error

# ------------------------------------------------------------------------------
def str_into_list(string):
    """
     This function converts a STRING to a LIST.

    --Inputs:
       
        string = A STRING to be converted to a LIST using "," as separator.
    
    --Outputs:

        list = A LIST containing the parts of the STRING.

    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  
    
    # Cheking not to be empty:
    if string == "":
        # It is empty!
        list = ""
        
    else:
        # It is not empty:
        list = list(string.split(","))
        list = [int(x) for x in list]
    
    # Returning outputs:
    return list

# ------------------------------------------------------------------------------
def list_into_str(list):
    """
     This function convert a LIST to a STRING.

    --Inputs:
       
        list = A LIST containing the parts of the STRING to define.
    
    --Outputs:

        string = A STRING obtained from the LIST using ", " as separator.

    """
 
    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Initialize an empty string:
    string = ""
 
    # Traverse in the string:
    string = ', '.join([str(elem) for elem in list])

    # Returning outputs:
    return string

# ------------------------------------------------------------------------------