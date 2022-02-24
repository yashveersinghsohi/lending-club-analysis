# Lending Club Analysis
This repository contains an Exploratory Data Analysis of the Lending Club Dataset.

# Code
Open the `EDA.ipynb` notebook to see the analysis, and open the `ZEST.py` module to see the code used to perform the analysis.

# Data
The datasets are downloaded and unzipped from this link given in the question prompt - [click](https://www.kaggle.com/wordsforthewise/lending-club)

Store the datasets in a `Data` directory (at the same level as the notebook and the module), to load them in the notebook for EDA.

# Conclusions

Applicants have a **better chance of loan approval** if they - 
- Request moderate loan amounts (not too low, not too massive).
- Have lower DTI ratio.
- Reside in the states and zip codes listed in question 3.
- Are more experienced, or have more number of years in employment.
- Avoid taking out loans for medical, home improvement, or for clearing past debts.

In addition to this, out of the accepted candidate, the ones that are at a **greater risk of default** are the ones who - 
- Are offered high interest and long term loans.
- Are offered loans of grade E, F, and G.
- Have greater number of credit history inquiries against their name.
- Have lower FICO and Credit Limits
- Take out loans for consolidating debts, clearing credit card bills and for home improvements.
