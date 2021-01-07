# Readme

The following list of files make up my homework assignment:

param_sweep_outputs2101173902.csv
pds-01-DataExploration_Cards_v4.ipynb
pds-01-DataExploration_Cards_v4_cleared.ipynb
pds-02-PrepareDataSet_combine_draft_scores-v2.ipynb
pds-03-streamlit-data-review-v1.py
pds-03A-WorkOnStreamlitHelperFunctions-v1.ipynb
resources/cards.csv
draft-scores/scores_XXX.txt (where XXX is the relevant set reference)

## Files prefixed with pds-

These are the working files. 

**pds-01**
The only difference between the two version of pds-01 is that the cleared version has had all the cell outputs cleared to allow it to load faster.

**pds-02**
This contains logic to go and get the draft scores

**pds-03**
pds-03-streamlit-data-review-v1.py captures all code to recreate the project apart from sourcing cards.csv (though contains a link to where this file can be sourced). 

**pds-03A**
This is a jupyter notebook where I experimented with functions and code that were eventually utilised in the pds-03 file. 


# Other FIles

**cards.csv**
This is the primary source of cards data for the project. As the file is 55mb I am not sure if github will accept this when I push it up. I have zipped it so it can be uploaded to GitHub since it exceeds the 50mb recommended file size.

It can be downloaded from the following link if needed:

https://mtgjson.com/api/v5/AllPrintingsCSVFiles.zip


**draft-scores**
These are sourced using an api at https://www.draftaholicsanonymous.com/

The code to access this API and generate the files is included in the streamlit workbook.

Each set scores are downloaded for is identified by 3 character alphanumeric code.


**param_sweep_outputs2101173902.csv**
These are the outputs of a parameter sweep. The code used to generate this is included in pds-03. It has been produced ahead of time and loaded to reduce load times as it takes around half an hour to run on my computer.
