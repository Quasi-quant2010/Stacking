# Emsenble Classification Framework with Stacking and Blending

# Reference
- This Framework is refered by https://github.com/AlexInTown/Santander

# Contribution
- We add an another method at 1-Level in Stacking.
- the Method is GBDT + Linear Classifier
- http://qiita.com/Quasi-quant2010/items/a30980bd650deff509b4

# Run

## Make Data Directory
- mkdir -p /home/username/data/sklearn/stacking/input
- mkdir -p /home/username/data/sklearn/stacking/output
- mkdir -p /home/username/data/sklearn/stacking/model
- mkdir -p /home/username/data/sklearn/stacking/graph
- mkdir -p /home/username/data/sklearn/stacking/stack_setting

## Data
- wget https://github.com/bluekingsong/simple-gbdt/tree/master/data/adult.data.csv

## Set config.xml
- you setup config.xml. Maybe, set the username in <data.path>/home/username/data/sklearn/stacking/</data.path>

## Run
- python main.py

## Output
- stack_setting  
you get the bset stacking and blending parameter
- graph
- output

# Require
- Sklearn
- Pandas
- Sklearn-Pandas
- Lasagne
- Seaborn
- Matplotliib
- if you have some error, you install the lack libraries by pip.
