Problem Statement:

Can we predict whether this product is suitable for someone with this skin type?
Can we predict whether a given skincare/makeup product will be well-received (recommended/highly rated) 
by someone with a specific skin type — using only the product's own attributes and the customer's stated skin type 




To run a file with name file_name - type in terminal >> python file_name.py
To add a file to the github repo, use the commands >> git add file_name, then >> git commit -m, then >> git push origin main


Step 0 > Github and evniornment
 Set up a github repository on github where all the code from VS will be committed.
 Then set up an evironment in VSCode using the command >> conda create -p newenvt python== 4.1.1
 Then activate the enviorment using >> conda activate -newenvt
 Then create a readme file here
 Initialise git using >> git init 
 To add this readme file to gitbub repo, run the command >> git add readme.md, then >> git commit -m, then >> git push origin main

Step 1 > Create the gitingnore file and the requirements.txt file
Step 2 > Create an empty fiolder src and then the setup.py file in it along with init.py
Step 3 > Upon running the setup.py file, the ML_Project_Hardika.egg.info is created along with the build and dist files too.
Step 4 > Then a templayte.py file is created. This will define the structure of the project and will create the necessary folders and files for the project. Upon running the template.py file, it will crate all your files memtioned.
Step 5 > Create the logger.py file in the src folder so that instead of using scattered print() statements, your entire project writes timestamped, leveled messages (INFO, ERROR, WARNING) to a file automatically. 
Step 6 > Crate the exception.py file all exception handelling
Step 6 > Create the application.py file to call these functions of logging and execption and eventually other functoins in the pipeline too
Step 7 > In the components folder that was created in the template file, code the data ingestion pipeline in the data_ingestion.py file. This is the data ingestion component which will read the data from kaggle and split it into train and test datasets, that will be stored in the newly created artifacts folder as raw, train, and text files.
Step 8 > In the same way, create the data transformation .py file for EDA and feature scaling + encoding
Step 9 > To run step 8, we need changes to be made in Utils.py and application.py
Step 10 > 




