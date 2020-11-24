# MarketAuthenticator
This root folder contains 5 python files, 2 drivers, and 2 log files.  Each is described below:


**chromedriver.exe**:
the driver that allows webscraping with Chrome.  This must be in the same directory as the file which performs the webscraping.

**debug.log**:
the file that stores Chrome debugging outputs

**geckodriver.exe**:
the driver that allows webscraping with FireFox.  This must be in the same directory as the file which performs the webscraping.

**geckodriver.log**:
the file that stores FireFox webscraping logging outputs

Python files
-------------------
**https.py**:
responsible for direct communication with the alleged website through https or http, also does certificate chain validation and hostname checking

**scraper.py**:
main functionality of the data collector submodule, gathers 11 attributes about the alleged website using web scraping and by calling https.py

**neural_network_helper.py**:
contains functions that were used while gathering data to train the model, and functions that were used for preprocessing gathered data to be fed to the model.

**neural_network.py**:
main functionality of the data analyzer submodule, contains the code for the neural network, including training it and submitting an individual data vector to it to see the output

**main.py**:
allows user to input a url through the console which will be run through the data collector, and then the data analyzer.  The output is a confidence score of the models prediction that the input url is either legitimate or phishing


Installation from complete project zip file:
-------------------
1. Unzip the folder
2. Open the folder in pycharm
3. After pycharm finishes indexing, click "configure python interpreter" and set python 3.8 as your interpreter.  Instructions on how to do this are found here: https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#add_new_project_interpreter
4. Click on "add configuration" on the top right of the pycharm window.  Click on the + sign on the top left, select python, and set the script path to be main.py in the project root directory

Installation from github:
-------------------
1. create a local directory for this project
2. open git bash in the local directory and enter the following commands:
```
git init
git remote add origin https://github.com/achupanikath/MarketAuthenticator.git
git pull origin demo
```
3. open the local directory in pycharm
4. Go to file -> settings -> Project: -> python interpreter.  Click on the down arrow next to python interpreter, and click show all.  Then click the + sign.  In the new window that pops up, set your base interpreter to your python 3.8 executable, wherever it has been installed (if you don't have it, you will need to install it).  Then select ok.  A window should pop up that says "creating virtual environment".
5. Click on "add configuration" on the top right of the pycharm window.  Click on the + sign on the top left, select python, and set the script path to be main.py in the project root directory
6. Navigate to the terminal (located at the bottom of the pycharm window).  Enter the following commands, in order:
```
pip install selenium
pip install pyOpenSSL
pip install certifi
pip install requests
pip install numpy==1.18.5
pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas
pip install sklearn
pip install matplotlib
```
      
Usage:
-------------------
run main.py, and when the console prompts you for a url, type in a url and hit enter.  The data collector will run on the url, followed by the data analyzer, and the model's prediction will output to the console.

Contact
-------------------
for any questions please email jobrienweiss@umass.edu or apanikath@umass.edu
