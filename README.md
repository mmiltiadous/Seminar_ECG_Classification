This reposidory is accompanied by a report submitted for the course Seminar Advances in Deep Learning at Leiden University.

# ECG_Classification
Guidlines on how to run the code:

## CNN

For dataset with other rhythm run:  python3 cnn_2.py 

For dataset without other rhythm run:  python3 cnn_noother.py

The results for all the experiments mentioned in the report are in the results_cnn folder

## InceptionTime
The code is placed in inception_time_folder.

For dataset with other rhythm run:  python3 main.py InceptionTime

For dataset without other rhythm, uncomment lines 180-230 and comment lines 128-178 in utils.py file and run:  <br>
python3 main.py InceptionTime

The results for all the experiments mentioned in the report are in the results_inception folder

## TCGAN


# References
1. Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier,
Daniel F. Schmidt, Jonathan Weber, Geoffrey I. Webb, Lhassane Idoumghar, Pierre-
Alain Muller, and François Petitjean. 2020. InceptionTime: Finding AlexNet for
Time Series Classification. Data Mining and Knowledge Discovery (2020). https://github.com/hfawaz/InceptionTime
