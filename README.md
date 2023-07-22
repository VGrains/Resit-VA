# Resit-VA

Welcome to our Fire Analysis VA tool.

To run our dashboard, please run 'python dashboard.py' or 'python3 dashboard.py' in the terminal.

dashboard.py: the main dashboard file with the dash app layout and callbacks. The callback functions call to functions defined in data.py for a cleaner overview.

data.py: our 'backend' file. Here all the figures and tables are created and returned. Also the prediction task is performed here.

model.py: our predictive model file. Here we trained our models.

requirements.txt: the packages required to run our VA tool.

ANN_reg2.joblib: our saved neural network model.

SVM2.joblib: our saved support vector classifier model.

Wildfire_att_description.txt: a file with descriptions of the columns in the csv file.