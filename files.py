###############################################################################
#                        ICyTE - LPI - AI Toolbox                             #
# module name:                                                                #
#     files                                                                   #
#                                                                             #
# module description:                                                         #
#     This module contains functions developed for managing files and         #
#     folders.                                                                #
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
def dict_conversion(dict, conversion):
    """
     This function performs conversion of format in specified keys of 
     a DICTIONARY.

    --Inputs:
       
        dict = A DICTIONARY which requires format conversion in some of its 
               keys. In all the cases the values are STRING separated by 
               commas ', '.

        conversion = A DICTIONARY caontaining the keys and the format for 
                     conversion as STRINGS.
                     "list" --> The content is converted to a LIST.
                     "boolean" --> The content is converted to a BOOLEAN.
                     "float" --> The content is converted to a LIST
                                 with FLOATS.
                     "int" --> The content is converted to a LIST
                               with INTS.
    
    --Outputs:

        dict = The DICTIONARY with converted keys.

    """

    # Author: Diego Comas 
    # Revised by: -

    # Check for conversion of parameters values:
    if conversion:
        list_keys = list(conversion.keys())
        for key in list_keys:
            value = conversion[key]
            if value=="list":
                dict[key] = list(dict[key].split(', '))
            elif value=="boolean":
                dict[key] = eval(dict[key])
            elif value=="float":
                dict[key] = [float(x) for x in list(dict[key].split(', '))]
            elif value=="int":
                dict[key] = [int(x) for x in list(dict[key].split(', '))]

    # Returning outputs:
    return dict

# ------------------------------------------------------------------------------
def read_params_file(file_name, separator, conversion = {}):
    """
     This function read a TXT file containing parameters and return a DICTIONARY 
     of parameters.
    
    --Inputs:
    
        file_name = A STRING with the full name of the parameters file.
        
        separator = A STRING indicating the separator used in the file. For instance: ": ".

        conversion = A DICTIONARY caontaining the keys and the format for 
                conversion from STRINGS.
                "list" --> The content is converted to a LIST.
                "boolean" --> The content is converted to a BOOLEAN.
                "float" --> The content is converted to a LIST
                            with FLOATS.
                "int" --> The content is converted to a LIST
                        with INTS.
    
    --Outputs:
        
        dic_Parameters = A DICTIONARY with keys equal to the text before separator 
                         (one per line) and the content equal to the text after the 
                         separator. If "conversion" is defined, the specified keys are 
                         converted accordantly.
        
        lines = A LIST with the lines in the text file.
        
    """
    
    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Reading the file and returning a dictionary:
    dic_Parameters = {}
    with open(file_name, "r") as f:
        lines = f.readlines()
        # Converting lines to a dictionary:
        for line in lines:
            list_1 = line.split(separator)
            dic_name_i = list_1[0]
            dic_value_i = list_1[1]
            # Removing the ENTERS:
            dic_Parameters[dic_name_i] = dic_value_i.rstrip()

    # Check for conversion of parameters values:
    if conversion:
        dic_Parameters = dict_conversion(dic_Parameters, conversion)

    # Returning outputs:
    return dic_Parameters, lines

# ------------------------------------------------------------------------------
def file_to_list(filename):
    """
     This function reads a TXT file a returns the content in a LIST (an element 
     per line).

    --Inputs:
    
        file_name = A STRING with the full name of the parameters file.
        
    --Outputs:
        
        lines = A LIST with the lines in the text file.
        
    """
    
    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Open file and pass each line to a element in a LIST:
    with open(filename, "r") as f:
        aux = f.readlines()
    lines = [str(x.strip()) for x in aux]

    # Returning outputs:
    return lines

# ------------------------------------------------------------------------------
def input_files(inputfile):
    """
    This function returns a list of files with particular conditions.

    --Inputs:
    
        inputfile = A STRING with the path and parameters (concatenated). For 
                    instance: "...folder/*.txt" returns all the files in 
                    "...folder/" ending in ".txt". 
        
    --Outputs:
        
        list_of_files = A LIST with the names of the files which meet the 
                        condition.
        
    """

    # Author: Diego Comas 
    # Revised by: -
    
    # Libraries:
    import glob
    
    # Function for sorting values:
    def numericalSort(value):
        import re
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    # Generationg the list of files ordered numerically:
    list_of_files = sorted((glob.glob(inputfile)), key=numericalSort)

    # Returning outputs:
    return list_of_files

# -----------------------------------------------------------------------------
def check_key(dic, key):
    """
     This check if a KEY is defined in a DICTIONARY. It is useful for checking 
     parameters read from a TXT file.

    --Inputs:
    
        dic = A DICTIONARY.

        key = A STRING containing the name of the key to search.
        
    --Outputs:
        
        flag_key = A BOOLEAN indicating if the key exists in the DICTIONARY.

        key_content = The value of the key in the DICTIONARY, if it exists.    
     
    """

    # Author: Diego Comas 
    # Revised by: -

    # Initializing outputs:
    key_content = []
    flag_key = False

    # Searching for the key:
    if key in dic.keys():
        # Preparing outputs:
        key_content = dic[key]
        flag_key = True

    # Returning outputs:
    return flag_key, key_content

# -----------------------------------------------------------------------------
def images_paths_and_names(full_path):
    """
     This function returns the full path of all the files in "full_path" and 
     their names.

    --Inputs:
    
        full_path = A STRING with the path for searching files.
        
    --Outputs:
        
        file_paths = A LIST with the full-path for the files found in 
                     "full_path".

        file_names = A LIST with the names of the the files found in 
                     "full_path".
        
    """

    # Author: Agustín Amalfitano 
    # Revised by: Diego Comas  

    # Libraries:
    import os

    # Inicializing:
    file_paths = []
    file_names = []

    # We find the images using the OS library:
    with os.scandir(full_path) as entries:
        for entry in entries:
            img_i = full_path + "/" + entry.name
            file_paths.append(img_i)
            img_name_i = entry.name
            file_names.append(img_name_i.split(".")[0])

    # Returning outputs:
    return(file_paths, file_names)

# -----------------------------------------------------------------------------
def create_folder(path_create):
    """
     This function creates a folder.

    --Inputs:
    
        path_create = A STRING with the full name of the folder to create.
        
    --Outputs:
        
        existing_flag = A BOOLEAN indicating if the folder already exists..
        
    """

    # Author: Diego Comas 
    # Revised by: -

    # Libraries:
    import os

    # Checking if it is already exists!
    try:
        os.stat(path_create)
        existing_flag = True
    except:
        os.mkdir(path_create)
        existing_flag = False

    # Returning outputs:
    return existing_flag

# -----------------------------------------------------------------------------