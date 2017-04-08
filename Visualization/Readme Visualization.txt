Steps to run the Visualization:

NOTE: Since the final output files are already present, the user can directly go to steps 2 and 3
to run the demo.

1. Run all the "../Logistic Regression/LogisticRegression.py", "../XGBoost/xgboost1.py", "../ANN/ANN.py"
files to get the required output files for visualization (for now there are output files from our test are already there in the folder "datafiles") 

2. All the required javascript and css files are present in the current folder as follows:

/Visualization
	/css
	  project.css
	/datafiles
	  Accuracy.csv
	  ANN.csv
	  countries.topo.json
	  LogisticRegression.csv
	  XGBoost.csv
	/js
	  checkBox.js
	  d3.v3.min.js
	  project.js
	  topojson.v1.min.js
	index.html
2. Serve the current folder (I used a python "SimpleHTTPServer" to serve this folder).

3. In our case entering "localhost:8000" in the browser made our visualization up and running.