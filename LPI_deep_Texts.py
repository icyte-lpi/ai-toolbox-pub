###############################################################################
#                         ICyTE-LPI-Deep Toobox                               #
# module name:                                                                #
#     LPI_deep_Texts                                                          #
#                                                                             #
# module description:                                                         #
#     This module contains all the required functions for managing text       #
#     files.                                                                  #
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
# Functions              |   Date    |    Authors           |   Description
# ------------------------------------------------------------------------------
#
# save_list_to_text        04/05/2023   Diego Comas          Save list to text 
#                                        Agustín Amalfitano  file.
#
# str_into_list            04/21/2023   Diego Comas          Convert a STRING
#                                       Agustín Amalfitano   to a LIST.
#
# list_into_str            04/25/2023   Diego Comas          This function 
#                                                            convert a LIST to 
#                                                            a STRING.
#
# ------------------------------------------------------------------------------

# --------------------------IMPORTS---------------------------------------------
# Reserved.

# ------------------------------------------------------------------------------
def save_list_to_text(lines_list, file_name):
    """
     This save a LIST in a text file.

    --Inputs:
       
        lines_list = A LIST containing the text to be save in a file.
        
        file_name = A STRING with the full name of the file to be save.
    
    --Outputs:

        flag_error = A BOOLEAN flag defining the error status.

    """

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
     This function convert a STRING to a LIST.

    --Inputs:
       
        string = A STRING to be converted to a LIST using "," as separator.
    
    --Outputs:

        list = A LIST containing the parts of the STRING.

    """
    
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
 
    # Initialize an empty string:
    string = ""
 
    # Traverse in the string:
    string = ', '.join([str(elem) for elem in list])

    # Returning outputs:
    return string

# ------------------------------------------------------------------------------