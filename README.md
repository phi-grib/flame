# Flame
Flame is a flexible framework supporting predictive modeling within eTRANSAFE (http://etransafe.eu) project. 

Flame will allow to:
- Develop easily machine-learning models, for example, QSAR-like models starting from annotated collections of chemical compounds stored in standard formats (SDFiles)
- Transfer new models into a production environment where they can be used by web services to predict the properties of new compounds

NOTE: this README is under construction, so please excuse errors and inaccuracies

## Main features
- Native support of most common machine-learning algorithms, including rich configuration options and facilitating the model optimization 
- Support for any standard formatted input: from a tsv table to a collection of compounds in SMILES or SDFile format. 
- Multiple interfaces adapted to the needs of different users: as a web service, for end-user prediction, a full featured GUI for model development, command line, integration in Jupyter notebooks, etc.
- Support for parallel processing
- Integration of models developed using other tools (e.g. R, KNIME)
- Support for inter-model communication: the output of a model can be used as input for other models
- Integrated model version management

## Quickstarting
Flame provides a simple command-line interface (Flame), which is useful for accessing its functionality and getting acquainted with its use.

Let's start creating a new model:

```sh
flame -c manage -a new -e MyModel
```

This creates a new entry in the model repository, and the development version of the model, populating these entries with default options.
The contents of the model repository are shown using the command

```sh
flame -c manage -a list
```

Building a model only requires entering an input file formatted for training one of the supported machine-learning methods. In the case of QSAR models, the input file can be a SDFile, where the biological property is annotated in one of the fields. 

The details of how Flame normalizes the structures, obtains molecular descriptors and applies the machine-learning algorithm are defined in a text file which now contains default options. These can be changed as we will describe latter, but for now let's use the defaults to obtain a RF model on a series of 100 compounds annotated with a biological property in the field <activity> 
	
```sh
flame -c build -e MyModel -f series.sdf
```	
After a few seconds the model is built, and a summary of the model quality is presented in the screen.
This model is immediately accessible for predicting the properties of new compounds. This can be done locally using the command:
```sh
flame -c predict -e MyModel -v 0 -f query.sdf
```	
And this will show the properties predicted for the compounds in the query SDFile 

In the above command we specified the model version used for the prediction. So far we only have a model in the development folder (version 0). This version will be overwritten every time we develop a new model for this endpoint. Let's imagine that we are very satisfied with our model and want to store it for future use. We can "publish" it with the command
```sh
flame -c manage -a publish -e MyModel
```	
This will create model version 1. We can list existing version for a given endpoint using the list command mentioned before
```sh
flame -c manage -a list
```	
Now, the output says we have a published version of model MyModel. 

Imagine that the model is so good you want to send to a company, so they can run predictions for confidential compounds that they cannot disclose to you. The model can be exported using the command
```sh
flame -c manage -a export -e MyModel
```	
This creates a very compact file with the extension .tgz in the local directory. It can be sent by e-mail or uploaded to a repository in the cloud from where the company can download it. In order to use it, the company can easily install the new model using the command
```sh
flame -c manage -a import -f MyModel.tgz
```	
And then the model is immediately operative and able to produce exactly the same predictions we obtain at the development environment  
## Flame commands

### Building a model

### Management commands

### Prediction commands

## Flame-app
Flame-app is a simple prediction web server, accessible using a web interface

## Technical details

## Examples of use with Jupyter Notebooks
