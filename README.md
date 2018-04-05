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
python flame.py -c manage -a new -e MyModel
```

This creates a new entry in the model repository, and the development version of the model, populating these entries with default options.
The contents of the model repository are shown using the command

```sh
python flame.py -c manage -a list
```

Building a model only requires entering an input file formatted for training one of the supported machine-learning methods. In the case of QSAR models, the input file can be a SDFile, where the biological property is annotated in one of the fields. 

The details of how Flame normalizes the structures, obtains molecular descriptors and applies the machine-learning algorithm are defined in a text file which now contains default options. These can be changed as we will describe latter, but for now let's use the defaults to obtain a RF model on a series of 100 compounds annotated with a biological property in the field <activity> 
	
```sh
python flame.py -c build -e MyModel -f series.sdf
```	
After a few seconds the model is built, and a summary of the model quality is presented in the screen.
This model is immediately accessible for predicting the properties of new compounds. This can be done locally using the command:
```sh
python flame.py -c predict -e MyModel -v 0 -f query.sdf
```	
And this will show the properties predicted for the compounds in the query SDFile 

In the above command we specified the model version used for the prediction. So far we only have a model in the development folder (version 0). This version will be overwritten every time we develop a new model for this endpoint. Let's imagine that we are very satisfied with our model and want to store it for future use. We can "publish" it with the command
```sh
python flame.py -c manage -a publish -e MyModel
```	
This will create model version 1. We can list existing version for a given endpoint using the list command mentioned before
```sh
python flame.py -c manage -a list
```	
Now, the output says we have a published version of model MyModel. 

Imagine that the model is so good you want to send to a company, so they can run predictions for confidential compounds that they cannot disclose to you. The model can be exported using the command
```sh
python flame.py -c manage -a export -e MyModel
```	
This creates a very compact file with the extension .tgz in the local directory. It can be sent by e-mail or uploaded to a repository in the cloud from where the company can download it. In order to use it, the company can easily install the new model using the command
```sh
python flame.py -c manage -a import -f MyModel.tgz
```	
And then the model is immediately operative and able to produce exactly the same predictions we obtain at the development environment  
## Flame commands

| Command | Description |
| --- | --- |
| -c/--command | Action to be performed. Acceptable values are *build*, *predict* and *manage* |
| -e/--endpoint | Name of the model which will be used by the command. This name is defined when the model is created for the fist time with the command *-c manage -a new* |
| -v/--version | Version of the model, typically an integer. Version 0 makes reference to the model development "sandbox" which is created automatically uppon model creation |
| -a/--action | Management action to be carried out. Acceptable values are *new*, *kill*, *publish*, *remove*, *export* and *import*. The meaning of these actions and examples of use are provided below   |
| -f/--infile | Name of the input file used by the command. This file can correspond to the training data (*build*) or the query compounds (*predict*) |
| -h/--help | Shows a help message in the screen |


### Management commands
| Command | Example | Description |
| --- | --- | ---|
| new | *python -c manage -a new -e NEWMODEL* | Creates a new entry in the model repository named NEWMODEL  |
| kill | *python -c manage -a kill -e NEWMODEL* | Removes NEWMODEL from the model repository. Use with extreme care, since the program will not ask confirmation and the removal will be permanent and irreversible  |
| publish | *python -c manage -a publish -e NEWMODEL* | Clones the development version, creating a new version in the model repository. Versions are assigned sequential numbers |
| remove | *python -c manage -a remove -e NEWMODEL -v 2* | Removes the version specified from the NEWMODEL model repository |
| list | *python -c manage -a list* | Lists the models present in the repository and the published version for each one. If the name of a model is provided, lists only the the published versions for this model  |
| export | *python -c manage -a export -e NEWMODEL* | Exports the model entry NEWMODE, creating a tar compressed file *NEWMODEL.tgz* which contains all the versions. This file can be imported by another flame instance (installed in a different host or company) with the *-c manage import* command |
| import | *python -c manage -a import -e NEWMODEL* | Imports file *NEWMODEL.tgz*, typically generated using command *-c manage -a export* creating model NEWMODEL in the local model repository |


## Flame-app
Flame-app starts a simple prediction web server, 

```sh
python flame-ws.py 
```	

To access the web graphical interface, open a web browers and enter the address *http://localhost:8080*

![Alt text](images/flame-gui.png?raw=true "web GUI")

Web API services available:

| URL | HTTP verb | Input data | Return data | HTTP status codes |
| --- | --- | --- | --- | --- |
| /info | GET | | application/json: info_message response | 200 |
| /dir | GET | | application/json: available_services response | 200 |
| /predict | GET | | multipart/form-data encoding: - model - filename | application/json: predict_call response | 200 or 500 for malformed POST message |

## Technical details

### Using Flame

Flame can be used in several ways:
- Using the web-GUI, starting the flame-ws.py web-service
- Using the flame command descrived above
- As a Python package, making direct calls to the high-level objects *predict*, *build* or *manage*
- As a Python package, making calls to the lower level objects *idata*, *apply*, *learn*, *odata*

Typically, Flame models are developed by modeling engineers, importing an appropriate training series and tunning the parameters defined in *parameters.yaml* file, either with and editor or with the model GUI, and then running the flame build command until optimum results are obtained. This task can also be carried out making calls to the objects from an interactive Python environment, like a Jupyter notebook

Advanced users can customize the models by editting the objects *idata_child*, *appl_child*, *learn_child* and *odata_child* present at the *model/dev* folder. These empty objects are childs of the corresponding objects called by flame, and it is possible to override any of the parents' method simply by copying and editing these whitin the files

### Flame models

* parameter.yaml and child objects *

