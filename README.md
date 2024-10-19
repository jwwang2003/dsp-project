# Digital Signal Processing Project


## Python!

### Python Virtual Environment

Install the python virtual environment package:
`pip install virtualenv`

#### Initialize a python enviroment in your project folder

`python<version> -m venv <virtual-environment-name>`

Example (based on linux):
```
mkdir projectA
cd projectA
python3.8 -m venv env
```

Here, we are creating a virtual environment using python3.8 and naming it `env`.

#### Activating the virtual environment

For Linux (Ubuntu/MacOS) users:
`source env/bin/activate`

For Windows users:
```
env/Scripts/activate.bat //In CMD
env/Scripts/Activate.ps1 //In Powershel
```

![](./assets/terminal-ss-0.png)
> Notice how after we activate the environment, there is an indicator "dsp-project" that will pop up showing that we are indeed in __our__ "environment"


## Create a `requirements.txt`

This creates a list of all the libraries used in the environment which is essential for making this project portable and makes it easy for other members of the project to stay in sync. This is kind of like a package manager (similar to Maven in Java or NPM in JavaScript).

`pip freeze > requirements.txt`

## To install the required libraries (from a `requirements.txt`)

`pip install -r requirements.txt`

## Deactivate the environment

`deactivate`
