# -*- coding: utf-8 -*-

##    Description    eTOXlab model template
##                   
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2013-2016 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

from model import model

class imodel(model):
    def __init__ (self, vpath):
        model.__init__(self, vpath)

        ##########################################################################################################
        ##
        ## General settings
        ##
        ##    General characteristics of the model
        ##
        ##########################################################################################################
        self.buildable    = True           # True if the model needs to be trained (built) before it can be used
                                           # for prediction
                                           
        self.quantitative = True           # True for quantitative dependent variables, False for qualitative
                                           # (e.g. 1 or 0) endpoints 
        
        self.confidential = False          # True for building the model without storing any information about
                                           # the training series stuctures
                                           
        self.identity     = False          # If set to True, the structure of any query compound is compared with 
                                           # those in the training series, returning the annotated value
                                           
        self.experimental = False          # If set to True, and the input file contains a field with the label
                                           # described by 'SDFileExperimental' this value is returned
                                           
        ##########################################################################################################
        ##
        ## Input setting
        ##                                       
        ##    Labels of SDFile fields recognized by the program in the input file. These have the following
        ##    synthax. For example, if the SDFilenName field is 'name', the input file will contain this string
        ##    as shown:
        ##
        ##      > <name>
        ##      aspirin
        ##
        ##########################################################################################################
        self.SDFileName = 'name'           # Label of the SDFile field in the input file containing the name of
                                           # the molecule
                                           
        self.SDFileActivity = 'activity'   # Label of the SDFile field in the input file containing the value of
                                           # the dependent variable. Leave blank for non-supervided models (PCA)
                                           
        self.SDFileExperimental = ''       # Label of the SDFile field in the input file containing the value to
                                           # be returned, typically an experimentally determined value

        self.SDFileMetadata = []           # Label of any SDFile field that must be passed throug the normalization
                                           # filters. Typically they contain information that is used by models

        ##########################################################################################################        
        ##
        ## Normalization settings
        ##
        ##    Define how the input structures are normalized and transformed prior to being processed
        ##
        ##    These settings apply both to the stucture of the training series and query structures, which are 
        ##    submitted to exactly the same normalization workflow
        ##
        ##    * Licenses: the program moka is third party software and requires a license activation
        ##
        ##########################################################################################################
        self.norm = False                  # If True the structures are normalized. When set to False the values
                                           # of all the following settings are ignored
                                           
        self.normStand = True              # If True the structures are submitted to the structural normalization
                                           # program 'standardizer'
                                           
        self.normNeutr = True              # If True the ionization of the structures is adjusted to the pH
                                           # detailed by 'normNeutr_pH' using the program defined below
                                           
        self.normNeutrMethod = 'moka'      # Name of the program used to adjust the ionization of the structures
                                           # when the value of 'normNeutr' is set to True
                                           
        self.normNeutr_pH = 7.4            # Value of the pH used to adjust the ionization of the structures
                                           # when the value of 'normNeutr' is set to True
                                           
        self.norm3D = True                 # If True the structures are converted to 3D using the program 'corina'
                                           # irrespectivelly if the input structure is 2D or already 3D

        ##########################################################################################################
        ##
        ## Molecular descriptor settings
        ##
        ##    Define the program used to calculate the molecular descriptors (MD) from the input structures and
        ##    diverse parameters defining how these programs will carry out the molecular descriptor computation
        ##
        ##    * Licenses: the program pentacle is third party software and requires a license activation
        ##
        ##########################################################################################################
        self.MD = 'padel'                   # Program used to calculate the MD. Must be one of the following
                                            # values: 'padel', 'pentacle', 'adriana' or 'external'

        # padel relevant settings
        
        self.padelMD = ['-2d']              # Type of padel MD to be calculared. Must be a subset of the values
                                            # in the following list, separated by a comma: '-2d' or '-3d'
                                            
        self.padelMaxRuntime = None         # Timeout (in miliseconds) used by padel to compute a single molecule.
                                            # If the computation time excedes this value the MD are undefined
                                            
        self.padelDescriptor = None         # Name of a xml formated filed containing the names of the MD computed
                                            # by padel

        # pentacle relevant settings
        
        self.pentacleProbes = ['DRY','O',
                               'N1','TIP']  # Name of the molecular probes used by program pentacle to compute
                                            # the molecular descriptors. Must be a subset of the values in the
                                            # following list, separated by a comma: 'DRY', 'O', 'N1' or 'TIP'
                                            
        self.pentacleOthers = []            # Include here any command line parameter to be sent to the program
                                            # pentacle 


        # external relevant settings

        self.MDexternalFile  = None         # Name of a TSV with MD. The first colum must be a text string
                                            # used to associate the MD with a given compound
                                            
        self.MDexternalID    = None         # Name of the SDField containing the text string used in the TSV
                                            # file for associating the MD with the compound 
        
        self.MDexternalField = None         # Name of a SDFiled with a tab separated line containing MD for this
                                            # molecule. This could be used to pass MD for predicted compound, not
                                            # present in the TSV file used at model building time
        
        ##########################################################################################################
        ##
        ## Modeling settings
        ##
        ##    Define the modeling method used to build the predictive model and diverse parameters defining 
        ##    how this method will work
        ##
        ##########################################################################################################
        self.model = 'pls'                  # Name of the modeling method used to build the model. Must be one of
                                            # the following values: 'pls', 'pca', 'RF'

        # pls relevant settings
        
        self.modelLV = 2                    # Number of latent variables used to build the model
        
        self.modelAutoscaling = True        # If true, the molecular descriptors will be normalized to unit 
                                            # variance by dividing each value by the variable variance

        self.modelCutoff = 'auto'           # Applicable only for PLS-DA ('quantitative' is False). The cutoff
                                            # value used to asign 1 or 0 to the predictions.
                                            # When set to 'auto' the value is assigned automatically as the one
                                            # producing the best compromise between sensitivity and specificity
                                            
        self.selVar = False                 # If True the program will run the variable selection method
                                            # defined by 'selVarMethod'. This setting can extend significantly 
                                            # the time required to build a model (by many hours or even days)

        # variable selection relevant settings
        
        self.selVarMethod = 'golpe'         # Name of the variable selection method. Must be one the following
                                            # values: 'golpe' (to be extended in following versions)
                                            
        self.selVarLV = 2                   # Number of latent variables used to build the reduced models used
                                            # by the variable selection. Could be diferent than 'modelLV'
        
        self.selVarCV = 'LOO'               # Name of the cross-validation method used during the variable
                                            # selection. Must be one the following values: 'LOO' (to be extended
                                            # in following versions)
        
        self.selVarRun = 2                  # Number of maximumm sequential runs of the GOLPE algorithm to run
                                            # before stop. The algorithm will run only until the model predictive
                                            # ability (assesed by cross-validation) will no longer improve
        
        self.selVarMask = None              # Name of a file containing a previously computed mask of selected and
                                            # non selected variables

        # RF relevant settings

        self.RFestimators = 100              # number of trees in the forest
        
        self.RFfeatures = 'sqrt'             # The number of features to consider when looking for the best split.
                                             # Acceptable values are 'sqrt', 'log2' and 'none'
                                             # - 'sqrt': then max_features=sqrt(n_features)
                                             # - 'log2': then max_features=log2(n_features)
                                             # - 'none': then max_features=n_features

        self.RFclass_weight = 'balanced'     # None or Balanced
                                             # class_weight : dict, list of dicts, “balanced”
                                             # The “balanced” mode uses the values of y to automatically adjust weights inversely
                                             # proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

        self.RFtune = True                   # If true optimizes the values of RFestimators and RFfeatures and generates a
                                             # diagnostic plot

        self.RFrandom = False                # If True a random seed is used for boostrapping
                                             # RF: Qualitative and Quantitative
                                             # Only valid for Qualitative Models #  If True a random seed is used as aleatory, if False, random seed is fixed to 1226.       
                                             # int seed, RandomState instance, or None (default)
                                             # The seed of the pseudo random number generator to use when shuffling the d.ta for
                                             # probability estimation.

          
        ## Model Validation Settings
        
        self.ModelValidationCV = 'loo'      ##      ('kfold', 'gkfold', 'stkfold', 'logo', 'lpgo', 'loo', 'lpo', 'shufsplit', 'gshufplit', 'stshufsplit', 'psplit', 'tsplit')
        self.ModelValidationN = 2           ##       int, Only for n_splits or n_groups
        self.ModelValidationP = 1           ##       int, Only for n_samples e.g. LeavePOut(p)

                                            ##        kfold = KFold(n_splits=2, random_state=self.random_state, shuffle=False)              ### K-Folds cross-validator
                                            ##        gkfold = GroupKFold(n_splits=2)                                                       ### K-fold iterator variant with non-overlapping groups.
                                            ##        stkfold = StratifiedKFold(n_splits=2, random_state=self.random_state, shuffle=False)  ### Stratified K-Folds cross-validator
                                            ##        logo = LeaveOneGroupOut()                                                             ### Leave One Group Out cross-validator
                                            ##        lpgo = LeavePGroupsOut(n_groups=2)                                                    ### Leave P Group(s) Out cross-validator
                                            ##        loo = LeaveOneOut()                                                                   ### Leave-One-Out cross-validator
                                            ##        lpo = LeavePOut(2)                                                                    ### Leave-P-Out cross-validator
                                            ##        shufsplit = ShuffleSplit(n_splits=3, random_state=0, test_size=0.25, train_size=None) ### Random permutation cross-validator
                                            ##        gshufplit = GroupShuffleSplit(test_size=10, n_splits=100)                             ### Shuffle-Group(s)-Out cross-validation iterator
                                            ##        stshufsplit = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)       ### Stratified ShuffleSplit cross-validator
                                            ##        psplit = PredefinedSplit(test_fold=array([ 0,  1, -1,  1]))                           ### Predefined split cross-validator
                                            ##        tssplit = TimeSeriesSplit(n_splits=3)                                                 ### Time Series cross-validator

        self.ModelValidationLC = False       ##        Plot learning curve
        


        ##########################################################################################################
        ##
        ## View settings
        ##
        ##    Define the type of visual representation to be generated as well as some properties of the generated 
        ##    graphics, like the color, shape and dimension of the markers
        ##
        ##    Available graphic types:
        ##                                    
        ##    'pca'       builds a PCA model with the training series of the selected model and the same MDs. Then
        ##                represents the scores of the PC1 and PC2 in the horizontal and vertical axes
        ##
        ##    'property'  represents the training series of the selected model using the log P for the X axis and
        ##                the molecular weight (MW) for the Y axis, as computed using RDKit
        ##
        ##    'project'   projects the training series of the selected model on a reference dataset for which
        ##                it must have been generated a PCA model. The markers of each compound can be assigned
        ##                colors according to the distance to model (DModX)
        ##                
        ##
        ##########################################################################################################                                            
        self.viewType = 'property'          # Type of graphic to generate. Must be one of the following
                                            # values: 'pca', 'property' or 'project'
                                            
        self.viewBackground = False         # If True, the selected model is represented together with another
                                            # dataset, shown as reference, defined by the two following settings
                                            
        self.viewReferenceEndpoint = None   # Name of the reference dataset (the endpoint label)
        
        self.viewReferenceVersion = 0       # Version of the reference dataset

        self.plotPCAColor = 'red'           # Color of the marker 
        self.plotPCAMarkerShape = 'D'       # Shape of the marker (see 'matplotlib.markers' document for a list)
        self.plotPCAMarkerSize = 40         # Size of the marker
        self.plotPCAMarkerLine = 0          # Thickness of the marker border
        
        self.plotPRPColor = 'red'           # Color of the marker
        self.plotPRPMarkerShape = 'D'       # Shape of the marker (see 'matplotlib.markers' document for a list)
        self.plotPRPMarkerSize = 40         # Size of the marker
        self.plotPRPMarkerLine = 0          # Thickness of the marker border
        
        self.plotPRJColor = 'DModX'         # Color of the marker or 'DModX' to assign a color according to
                                            # the distance to model (DModX) of every compound
                                            
        self.plotPRJMarkerShape = 'o'       # Shape of the marker (see 'matplotlib.markers' document for a list)
        self.plotPRJMarkerSize = 50         # Size of the marker
        self.plotPRJMarkerLine = 1          # Thickness of the marker border
        
        self.plotBGColor = '#aaaaaa'        # Color of the marker
        self.plotBGMarkerShape = 'o'        # Shape of the marker (see 'matplotlib.markers' document for a list)
        self.plotBGMarkerSize = 20          # Size of the marker
        self.plotBGMarkerLine = 0           # Thickness of the marker border

        ##########################################################################################################
        ##
        ## Path to external programs
        ##
        ##    Define the absolute path to diverse external programs used by the program. If the software is
        ##    updated, the correct path must be introduced here
        ##
        ##########################################################################################################                                            
        self.mokaPath = '/opt/blabber/blabber4eTOX/'
        self.padelPath = '/opt/padel/padel218ws/'
        self.padelURL = 'http://localhost:9000/computedescriptors?params=' 
        self.pentaclePath = '/opt/pentacle/pentacle106eTOX/'
        self.adrianaPath = '/opt/AdrianaCode/AdrianaCode226/'
        self.corinaPath = '/opt/corina/corina3494/'
        self.javaPath = '/usr/java/jdk1.7.0_51/'
        self.RPath = '/opt/R/R-3.0.2/'
        self.standardiserPath = '/opt/standardiser/standardise20140206/'
