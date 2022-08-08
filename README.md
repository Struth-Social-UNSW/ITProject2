# IT Project 2: Struth Social
This is the GitHub repository for the IT Project Team Struth Social's Fake News Detection Web Application.

- [Struth Social]
  - [Development Workflow](#development-workflow)
  - [Setting up the project (IMPORTANT)](#setting-up-the-project-important)
    - [For the frontend](#for-the-frontend)
    - [For the backend](#for-the-backend)
  - [Front-end](#front-end)
    - [Development commands](#development-commands)
    - [Technologies in use](#technologies-in-use)
  - [Back-end](#back-end)
    - [Development commands](#development-commands-1)
    - [Technologies in use](#technologies-in-use-1)
  - [The Team](#the-team)

## Development Workflow

Info about the workflow here:
-procedure for pushing new commits, review of pull requests etc.

Issues will be tracked on Trello


## Setting up the project (IMPORTANT)
- You will need Python 3.8 installed for both the backend and the frontend. Download the relevant version here: [https://www.python.org/downloads/](https://www.python.org/downloads/release/python-380/)
### For the frontend
1. In terminal or command prompt, navigate to Flask_App folder.
2. Install the following requirements using the following commands
  - pip install flask
  - pip install tweepy 
3. For MacOS:
  - export FLASK_APP=flaskapp.py
4. For Windows:
  - set FLASK_APP=flaskapp.py
5. Once exported, use:
  - flask run
6. This will create a locally hosted web app located at http://localhost:5000/
7. Web server can be terminated using CTRL + C
8. To run in debug mode:
  - export FLASK_DEBUG = 1
  - flask run
### For the backend
  - Inside the backend folder, open a terminal window. Then run the command: 
  - pip install -r requirements.txt

## Front-end

### Development commands
- ```CMD HERE```: Explain what the command does

### Technologies in use
- **Python 3.8**: Underlying language enabling the program to run
- **Flask 1.1.2**: Python framework for locally hosting web application
- **Tweepy 4.10.0**: Python library for accessing the Twitter API.

## Back-end

### Development commands
- ```CMD HERE```: Explain what the command does

### Technologies in use
- **Python 3.8**: Underlying language enabling the program to run
- **covid-twitter-bert**: Pretrained BERT model for analysing COVID-19 Twitter data
  - **tensorflow==2.2.0**: Tensorflow is the Deep Learning library which will enable to BERT model to function as expected
  - **tensorflow_hub**: A repository of trained ML/DL models
  - **tensorflow_addons==0.11.2**: A library containing useful tools for the Tensorflow package
  - **gin-config**: Lightweight config framework for Python, useful for the injection of multiple parameters in ML/DL tasks
  - **tqdm**: A package containing a graphical progress bar
  - **pandas**: Open source data analysis and manipulation tool
  - **scikit-learn**: Preprocessing and feature extraction tool
  - **google-cloud-storage**: Access path for use of TPUs in DL tasks
  - **spacy**: A commercial grade NLP package
  - **emoji**: Contains the codes for all current emojis
  - **unidecode**: Decodes unicode strings into ASCII characters
  - **cloud-tpu-client**: Interface for using TPUs
  - **sentencepiece**: Unsupervised text tokeniser/detokeniser

## The Team
Callum Pevere  (c.pevere@student.unsw.edu.au)
- Project Manager, Technical Lead, Technical Developer 

Liam Weber (l.weber@student.unsw.edu.au)
- Technical Developer, System Architect, Testing

Breydon Verryt-Reid (b.verrytreid@student.unsw.edu.au)
- Technical Developer, Quality Assurance, Testing  

Justin Macey (j.macey@student.unsw.edu.au) 
- Documentation Manager, Client Liaison, Technical Assistant, Testing  
