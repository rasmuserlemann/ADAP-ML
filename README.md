The goal of this project is to develop a full-fledged machine learning package that will carry out analysis of the results generated from ADAP-BIG. Specific objectives of ADAP-ML is to provide both uni-variate and multi-variate statistical analysis/machine learning methods for detecting biomarkers. Interactive capabilities with the data through visualization is useful to have, but ADAP-ML should try to minimize the reliance on users to interact with the data to extract knowledge from the data.

For testing:
- cd repo directory and create a virtual environment

      $ virtualenv venv

- Install versions from requirements.txt by 

      $ python -m pip install -r requirements.txt

- Execute the JupyterNotebooks/main.py file with test data

      $ python main.py SCLC_study_output_filtered_2.csv
      
- The generated folder is in JupyterNotebooks under Analysis_DATE_UNIQUE_CODE