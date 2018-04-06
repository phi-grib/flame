# Flame

Flame is a flexible framework supporting predictive modeling within eTRANSAFE (http://etransafe.eu) project. 

Flame allows to:
- Develop easily machine-learning models, for example, QSAR-like models starting from annotated collections of chemical compounds stored in standard formats (SDFiles)
- Transfer new models into a production environment where they can be used by web services to predict the properties of new compounds

Flame is in active development and **no stable release has been produced so far**. Even this README is under construction, so please excuse errors and inaccuracies

## Installation

It is required to **install Anaconda** to use the Conda package and environment manager. Download Anaconda from [here](https://www.anaconda.com/distribution/).
Flame also needs the 3rd party module Standardiser. Please follow **install standardiser** instrucctions.


Download the repository:

```bash
git clone https://github.com/phi-grib/flame.git

```

Go to the repository directory 

```bash
cd flame
```

and create the **conda environment** with all the dependencies and extra packages (jupyter, matplotlib...):

```bash
conda env create -f environment.yml
```

Once the environment is created do:

```bash
source activate flame
```

to activate the environment.

Now install flame package with:

```bash
python setup.py install
```

or

```bash
pip install -e .
```

## Main features

- Native support of most common machine-learning algorithms, including rich configuration options and facilitating the model optimization 
- Support for any standard formatted input: from a tsv table to a collection of compounds in SMILES or SDFile format. 
- Multiple interfaces adapted to the needs of different users: as a web service, for end-user prediction, a full featured GUI for model development, command line, integration in Jupyter notebooks, etc.
- Support for parallel processing
- Integration of models developed using other tools (e.g. R, KNIME)
- Support for inter-model communication: the output of a model can be used as input for other models
- Integrated model version management

## Quickstarting

Flame provides a simple command-line interface (`flame.py`), which is useful for accessing its functionality and getting acquainted with its use.

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

The details of how Flame normalizes the structures, obtains molecular descriptors and applies the machine-learning algorithm are defined in a parameter file (*parameter.yaml*) which now contains default options. These can be changed as we will describe latter, but for now let's use the defaults to obtain a Random Forest model on a series of 100 compounds annotated with a biological property in the field <activity> 
	
```sh
python flame.py -c build -e MyModel -f series.sdf
```	
After a few seconds the model is built, and a summary of the model quality is presented in the screen.
This model is immediately accessible for predicting the properties of new compounds. This can be done locally using the command:
```sh
python flame.py -c predict -e MyModel -v 0 -f query.sdf
```	
And this will show the properties predicted for the compounds in the query SDFile 

In the above command we specified the model version used for the prediction. So far we only have a model in the development folder (version 0). This version will be overwritten every time we develop a new model for this endpoint. Let's imagine that we are very satisfied with our model and want to store it for future use. We can obtain a persistent copy of it with the command
```sh
python flame.py -c manage -a publish -e MyModel
```	
This will create model version 1. We can list existing version for a given endpoint using the list command mentioned before
```sh
python flame.py -c manage -a list
```	
Now, the output says we have a published version of model MyModel. 

Imagine that the model is so good you want to send it elsewhere, for example, a company which wants to obtain predictions for confidential compounds in their own computing facilities. The model can be exported using the command
```sh
python flame.py -c manage -a export -e MyModel
```	
This creates a very compact file with the extension .tgz in the local directory. It can be sent by e-mail or uploaded to a repository in the cloud from where the company can download it. In order to use it, the company can easily install the new model using the command
```sh
python flame.py -c manage -a import -e MyModel
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

Management commands deserve further description:

### Management commands

| Command | Example | Description |
| --- | --- | ---|
| new | *python -c manage -a new -e NEWMODEL* | Creates a new entry in the model repository named NEWMODEL  |
| kill | *python -c manage -a kill -e NEWMODEL* | Removes NEWMODEL from the model repository. **Use with extreme care**, since the program will not ask confirmation and the removal will be permanent and irreversible  |
| publish | *python -c manage -a publish -e NEWMODEL* | Clones the development version, creating a new version in the model repository. Versions are assigned sequential numbers |
| remove | *python -c manage -a remove -e NEWMODEL -v 2* | Removes the version specified from the NEWMODEL model repository |
| list | *python -c manage -a list* | Lists the models present in the repository and the published version for each one. If the name of a model is provided, lists only the the published versions for this model  |
| export | *python -c manage -a export -e NEWMODEL* | Exports the model entry NEWMODE, creating a tar compressed file *NEWMODEL.tgz* which contains all the versions. This file can be imported by another flame instance (installed in a different host or company) with the *-c manage import* command |
| import | *python -c manage -a import -e NEWMODEL* | Imports file *NEWMODEL.tgz*, typically generated using command *-c manage -a export* creating model NEWMODEL in the local model repository |


## Flame-app

Flame includes a simple prediction web server.

```sh
python flame-ws.py 
```	

To access the web graphical interface, open a web brower and enter the address *http://localhost:8080*

![Alt text](images/flame-gui.png?raw=true "web GUI")

Web API services available:

(in development)

| URL | HTTP verb | Input data | Return data | HTTP status codes |
| --- | --- | --- | --- | --- |
| /info | GET | | application/json: info_message response | 200 |
| /dir | GET | | application/json: available_services response | 200 |
| /predict | POST | multipart/form-data encoding: model and filename | application/json: predict_call response | 200, 500 for malformed POST message |

The exact synthax of the JSON object returned by predict will be documented in detail elsewhere.

## Technical details

### Using Flame

Flame was designed to be used in different ways, using diverse interfaces. For example:
- Using the web-GUI, starting the `flame-ws.py` web-service
- Using the `flame.py` command described above
- As a Python package, making direct calls to the high-level objects *predict*, *build* or *manage*
- As a Python package, making calls to the lower level objects *idata*, *apply*, *learn*, *odata*

The two main modeling tasks that must be supported by Flame are the *model development* and the use of the models for *prediction*. These are typically carried out by people with different expertise and in different environments. Flame was designed around this concept and allow to decouple completelly both tasks. Somebody can develop a model in a research environment which can be easily exported to be installed in a production environment to serve prediction services. Flame implements interfaces designed specifically for each task, even if they share exactly the same code, to guarantee compatibility and consistency. 

### Developing models

Typically, Flame models are developed by modeling engineers. This task requires importing an appropriate training series and defininig the model building workflow. 

Model building can be easily customized by editing the parameters defined in a command file (called *parameters.yaml*), either with a text editor or with the Flame modeling GUI (**in development**). Then, the model can be built using the `flame.py` build command, and its quality can be assessed in an iterative process which is repeated until optimum results are obtained. This task can also be carried out making calls to the objects mentioned above from an interactive Python environment, like a Jupyter notebook. A full documentation of the library can be obtained running Doxygen on the root directory.

Advanced users can customize the models by editting the objects *idata_child*, *appl_child*, *learn_child* and *odata_child* present at the *model/dev* folder. These empty objects are childs of the corresponding objects called by flame, and it is possible to override any of the parents' method simply by copying and editing these whitin the childs code files.

Models can be published to obtain persistent versions, usable for predicton in the same environment, or exported for using them in external production environments, as described above.


### Runnning models

Models built in Flame can be used for obtaining predictions using diverse methods. We can use the command mode interface with a simple call:
```sh
python flame.py -c predict -m MyModel -v 1 -f query.sdf
```
This allows to integate the prediction in scripts, or workflow tools like KNIME and Pipeline Pilot.

Also, the models can run as prediction web-services, using the provided flame-ws interface. These services can be consumed by the stand-alone web GUI provided and described above or connected to a more complex platform, like the one currently in development in eTRANSAFE project.


## Licensing

Flame was produced at the PharmacoInformatics lab (http://phi.upf.edu), in the framework of the eTRANSAFE project (http://etransafe.eu). eTRANSAFE has received support from IMI2 Joint Undertaking under Grant Agreement No. 777365. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and the European Federation of Pharmaceutical Industries and Associations (EFPIA). 

![Alt text](images/eTRANSAFE-logo-git.png?raw=true "eTRANSAFE-logo") ![Alt text](images/imi-logo.png?raw=true "IMI logo")

Copyright 2018 Manuel Pastor (manuel.pastor@upf.edu)

Flame is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License as published by the Free Software Foundation version 3**.

Flame is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Flame. If not, see <http://www.gnu.org/licenses/>.

