# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from wordcloud import WordCloud


class EDA:
    def __init__(self, accepted: pd.DataFrame, rejected: pd.DataFrame) -> None:
        """
        This function initializes dataframes for the accepted and rejected datasets.

        ARGS:
            :accepted (pd.DataFrame): Dataset for approved loans.
            :rejected (pd.DataFrame): Dataset for rejected loans.
        
        RETURNS:
            :(None):
        """
        # Initializing approved candidates
        self.accepted = accepted
        # Adding binary default variable (0-no default, 1-default) for easy visualization
        self.accepted['default'] = np.where(
            self.accepted['loan_status'].isin(
                ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
            ), 
            1, 0
        )
        # Initializing rejected candidates
        self.rejected = rejected
    
    def loanAmountRequested(self) -> None:
        """
        This function is used to compare the loan amount that approved and 
        rejected applicants request for.

        First the loan amounts of both approved and rejected applicants are collected,
        and then boxplots for both these distributions are plotted (after outlier removal).

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Loan amounts of approved applicants
        accepted_loan_amount = self.accepted['funded_amnt'].dropna()
        # Clipping outliers (points outside the 3rd standard deviation are removed)
        accepted_loan_amount = pd.Series(
            np.where(
                np.abs(stats.zscore(accepted_loan_amount)) < 3, 
                accepted_loan_amount, np.nan
            )
        ).dropna()

        # Loan amounts of rejected applicants
        rejected_loan_amount = self.rejected['Amount Requested'].dropna()
        # Clipping outliers (points outside the 3rd standard deviation are removed)
        rejected_loan_amount = pd.Series(
            np.where(
                np.abs(stats.zscore(rejected_loan_amount)) < 3, 
                rejected_loan_amount, np.nan
            )
        ).dropna()

        # Plotting boxplots to compare the 2 distributions
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # BoxPlot for approved applicants
        sns.boxplot(x=accepted_loan_amount, ax=ax[0], color='lightblue')
        ax[0].set_title("Loan Amount for Approved Applications", size=15)
        ax[0].set_xlabel("")
        
        # BoxPlot for rejected applicants
        sns.boxplot(x=rejected_loan_amount, ax=ax[1], color='orange')
        ax[1].set_title("Loan Amount for Rejected Applications", size=15)
        ax[1].set_xlabel("")

        plt.show()

    def dtiApplicants(self) -> None:
        """
        This function is used to compare the Debt-to-Income ratio that approved and 
        rejected applicants.

        First the dti of both approved and rejected applicants are collected,
        and then boxplots for both these distributions are plotted (after outlier removal).

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # DTI of approved applicants
        accepted_dti = self.accepted['dti'].dropna()
        # Clipping outliers (points outside the 3rd standard deviation are removed)
        accepted_dti = pd.Series(
            np.where(
                np.abs(stats.zscore(accepted_dti)) < 3, 
                accepted_dti, np.nan
            )
        ).dropna()

        # DTI of rejected applicants
        rejected_dti = self.rejected['Debt-To-Income Ratio'].dropna()
        # Cleaning text column and converting to float data type
        rejected_dti = rejected_dti.apply(lambda dtiStr: dtiStr.replace("%", "")).astype(np.float32)
        # Clipping outliers (points within the middle 98 percentile are retained)
        rejected_dti = pd.Series(
            np.where(
                (rejected_dti < rejected_dti.quantile(0.99)) & 
                (rejected_dti > rejected_dti.quantile(0.01)), 
                rejected_dti, np.nan
            )
        ).dropna()

        # Plotting boxplots to compare the 2 distributions
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        # BoxPlot for approved applicants
        sns.boxplot(x=accepted_dti, ax=ax[0], color='lightblue')
        ax[0].set_title("DTI for Approved Applications", size=15)
        ax[0].set_xlabel("")
        
        # BoxPlot for rejected applicants
        sns.boxplot(x=rejected_dti, ax=ax[1], color='orange')
        ax[1].set_title("DTI for Rejected Applications", size=15)
        ax[1].set_xlabel("")

        plt.show()
    
    def locationApplicants(self) -> None:
        """
        This function is used to compare the approvals vs the rejections based on the 
        applicant's state and zip code.

        For each state and zipcode, we calculate the ratio of number of applicants 
        approved vs rejected. If this number is high, the state/zipcode is favorable.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Ratio of number of approvals vs rejections for each state
        accept_to_reject_state_ratio = (
            self.accepted['addr_state'].value_counts()/self.rejected['State'].value_counts()
        ).sort_values()

        print("STATES")
        print("Highest Accept-to-Reject ratio")
        print(list(accept_to_reject_state_ratio.index[-10:]))
        print("-"*80)
        print("Lowest Accept-to-Reject ratio")
        print(list(accept_to_reject_state_ratio.index[:10]))
        print("-"*80)
        print("-"*80)

        # Ratio of number of approvals vs rejections for each zipcode
        accept_to_reject_zip_ratio = (
            self.accepted['zip_code'].value_counts()/self.rejected['Zip Code'].value_counts()
        ).sort_values()

        print("ZIP CODES")
        print("Highest Accept-to-Reject ratio")
        print(list(accept_to_reject_zip_ratio.index[-10:]))
        print("-"*80)
        print("Lowest Accept-to-Reject ratio")
        print(list(accept_to_reject_zip_ratio.index[:10]))

    def experienceOfApplicants(self) -> None:
        """
        This function is used to compare the number of years of experience of 
        approved vs rejected applicants.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Preprocessing the number of years of employment for approved applicants
        accepted_emp_length = self.accepted['emp_length'].dropna().apply(
            lambda length: "".join([l for l in length if l.isnumeric()])
        ).astype(np.int32)

        # Preprocessing the number of years of employment for rejected applicants
        rejected_emp_length = self.rejected['Employment Length'].dropna().apply(
            lambda length: "".join([l for l in length if l.isnumeric()])
        ).astype(np.int32)

        # Creating a temporary dataframe for visualization
        emp_length_df = pd.DataFrame(
            np.c_[
                ((accepted_emp_length.value_counts().sort_index()/accepted_emp_length.shape[0])*100).round(2).to_numpy(),
                ((rejected_emp_length.value_counts().sort_index()/rejected_emp_length.shape[0])*100).round(2).to_numpy()
            ],
            index=np.arange(1, 11, 1),
            columns=['Accepted', 'Rejected']
        )
        # Plotting a bar graph to compare the approvals vs rejections for 
        # different years of employment
        fig, ax = plt.subplots(figsize=(8, 5))
        emp_length_df.plot.bar(color=['blue', 'orange'], ax=ax)
        ax.set_title('Percent of Loan Approvals/Rejections based on Number of Years of Employment', size=15)
        ax.set_xlabel('Number of Years of Employment', size=12)
        ax.set_ylabel('Percent of Loan Approvals/Rejections', size=12)
        plt.show()

    def loanTitleWordClouds(self) -> None:
        """
        This function is used to create word clouds for the words used to describe
        the title of the loan by applicants.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Cleaning loan titles, lower casing them, splitting them into words,
        # and counting their frequency.
        accepted_words_counter = Counter(self.accepted['title'].dropna().str.lower().str.split(' ').explode().to_list())
        rejected_words_counter = Counter(self.rejected['Loan Title'].dropna().str.lower().str.split(' ').explode().to_list())

        # Initializaing and creating the 2 word clouds 
        wc_accepted, wc_rejected = WordCloud(), WordCloud()
        wc_accepted.fit_words(accepted_words_counter-rejected_words_counter)
        wc_rejected.fit_words(rejected_words_counter-accepted_words_counter)

        # Visualzing the common (relatively distinct) words used in loan titles 
        # for approved and rejected candidates. 
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))
        ax[0].axis("off")
        ax[1].axis("off")
        
        ax[0].imshow(wc_accepted)
        ax[0].set_title("Common Words in Approved Loan Applications", size=20)

        ax[1].imshow(wc_rejected)
        ax[1].set_title("Common Words in Rejected Loan Applications", size=20)
        plt.show()

    def defaultsInterestRate(self) -> None:
        """
        This function is used to compare the interest rate of loans that are paid of 
        and the ones that are defaulted on.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Collecting the interest rates on defaulted and non-defaulted loans 
        accepted_int_rate = self.accepted[['int_rate', 'default']].dropna(subset=['int_rate'])
        accepted_not_default = np.where(
            accepted_int_rate['default']==1, 
            np.nan, accepted_int_rate['int_rate']
        )
        accepted_default = np.where(
            accepted_int_rate['default']==1, 
            accepted_int_rate['int_rate'], np.nan
        )

        # Plotting BoxPlots to compare the distributions
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        sns.boxplot(x=accepted_not_default, ax=ax[0], color='lightblue')
        ax[0].set_title("Interest rate for Non-Defaulters", size=15)
        ax[0].set_xlabel("")

        sns.boxplot(x=accepted_default, ax=ax[1], color='orange')
        ax[1].set_title("Interest rate for Defaulters", size=15)
        ax[1].set_xlabel("")

        plt.show()

    def defaultsTerm(self) -> None:
        """
        This function is used to compare the term of loans that are paid of 
        and the ones that are defaulted on.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Plotting bar grahs to compare the effects of term size on defaults.
        fig, ax = plt.subplots(figsize=(8, 5))
        self.accepted.groupby(by=['default'])['term'].value_counts().unstack().plot.bar(ax=ax)
        ax.set_xlabel('Deafult', size=12)
        ax.set_ylabel('Number of Applications (in millions)', size=12) 
        ax.set_title('Number of Defaults for 36 and 60 month loan terms', size=15)
        plt.show()

    def defaultsLoanGrade(self) -> None:
        """
        This function is used to uncover the correlation between the grade of the loan
        and the default rates 

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Plotting bar grahs to compare the effects of loan grades on defaults.
        fig, ax = plt.subplots(figsize=(8, 5))
        self.accepted.groupby(by=['default'])['grade'].value_counts().unstack().T.plot.bar(ax=ax)
        ax.set_xlabel('Loan Grade', size=12)
        ax.set_ylabel('Number of Applications (in millions)', size=12) 
        ax.set_title('Number of Defaults w.r.t loan grades', size=15)
        plt.show()

    
    def defaultsCreditInq(self) -> None:
        """
        This function is used to compare the number of credit inquiries made on 
        loan applicants and how it correlates with the defaulted loans.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Collecting the number of credit inquiries of applicants
        accepted_cr_inq = self.accepted[['inq_last_12m', 'default']].dropna(subset=['inq_last_12m'])
        # Clipping top 1 percentile of outliers
        accepted_cr_inq = accepted_cr_inq[
            accepted_cr_inq['inq_last_12m']<accepted_cr_inq['inq_last_12m'].quantile(0.99)
        ]
        accepted_not_default = np.where(
            accepted_cr_inq['default']==1, 
            np.nan, accepted_cr_inq['inq_last_12m']
        )
        accepted_default = np.where(
            accepted_cr_inq['default']==1, 
            accepted_cr_inq['inq_last_12m'], np.nan
        )

        # Visualizing the 2 distributions (number of inquiries for defaults and non-defaults)
        # using boxplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        sns.boxplot(x=accepted_not_default, ax=ax[0], color='lightblue')
        ax[0].set_title("Number of Credit Inquiries for Non-Defaulters", size=15)
        ax[0].set_xlabel("")

        sns.boxplot(x=accepted_default, ax=ax[1], color='orange')
        ax[1].set_title("Number of Credit Inquiries for Defaulters", size=15)
        ax[1].set_xlabel("")

        plt.show()
 
    def defaultsFicoRange(self) -> None:
        """
        This function is used to compare the upper and lower FICO bounds for the applicants,
        and how it correlates with defaults.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        fig, ax = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
        
        # Visualizing Lower FICO scores for defaulters and non-defaulters
        accepted_fico_low = self.accepted[['fico_range_low', 'default']].dropna(subset=['fico_range_low'])
        accepted_not_default = np.where(accepted_fico_low['default']==1, np.nan, accepted_fico_low['fico_range_low'])
        accepted_default = np.where(accepted_fico_low['default']==1, accepted_fico_low['fico_range_low'], np.nan)

        sns.boxplot(x=accepted_not_default, ax=ax[0, 0], color='lightblue')
        ax[0, 0].set_title("Lower FICO range for Non-Defaulters", size=15)
        ax[0, 0].set_xlabel("")

        sns.boxplot(x=accepted_default, ax=ax[1, 0], color='orange')
        ax[1, 0].set_title("Lower FICO range for Defaulters", size=15)
        ax[1, 0].set_xlabel("")

        # Visualizing Upper FICO scores for defaulters and non-defaulters
        accepted_fico_high = self.accepted[['fico_range_high', 'default']].dropna(subset=['fico_range_high'])
        accepted_not_default = np.where(accepted_fico_high['default']==1, np.nan, accepted_fico_high['fico_range_high'])
        accepted_default = np.where(accepted_fico_high['default']==1, accepted_fico_high['fico_range_high'], np.nan)

        sns.boxplot(x=accepted_not_default, ax=ax[0, 1], color='lightblue')
        ax[0, 1].set_title("Upper FICO range for Non-Defaulters", size=15)
        ax[0, 1].set_xlabel("")

        sns.boxplot(x=accepted_default, ax=ax[1, 1], color='orange')
        ax[1, 1].set_title("Upper FICO range for Defaulters", size=15)
        ax[1, 1].set_xlabel("")

        plt.show()

    def defaultsCreditLimit(self) -> None:
        """
        This function is used to compare the credit limit of applicants and how it
        correlated with the default rate

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Collecting the credit limit of applicants
        accepted_credit_limit = self.accepted[['tot_hi_cred_lim', 'default']].dropna(subset=['tot_hi_cred_lim'])
        # Clipping top 1 percentile of outliers
        accepted_credit_limit = accepted_credit_limit[
            accepted_credit_limit['tot_hi_cred_lim']<accepted_credit_limit['tot_hi_cred_lim'].quantile(0.99)
        ]
        accepted_not_default = np.where(
            accepted_credit_limit['default']==1, 
            np.nan, accepted_credit_limit['tot_hi_cred_lim']
        )
        accepted_default = np.where(
            accepted_credit_limit['default']==1, 
            accepted_credit_limit['tot_hi_cred_lim'], np.nan
        )
        
        # Visualizing the 2 distributions (credit limit for defaults and non-defaults)
        # using boxplots
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        sns.boxplot(x=accepted_not_default, ax=ax[0], color='lightblue')
        ax[0].set_title("Credit Limit for Non-Defaulters", size=15)
        ax[0].set_xlabel("")

        sns.boxplot(x=accepted_default, ax=ax[1], color='orange')
        ax[1].set_title("Credit Limit for Defaulters", size=15)
        ax[1].set_xlabel("")

        plt.show()

    def defaultsLoanPurpose(self) -> None:
        """
        This function is used to uncover the correlation between the number of defaults
        and the purpose of the loan.

        ARGS:
            :(None):
        
        RETURNS:
            :(None):
        """
        # Plotting bar grahs to compare the effects of loan purpose on defaults.
        fig, ax = plt.subplots(figsize=(8, 5))
        self.accepted.groupby(by=['default'])['purpose'].value_counts().unstack().T.plot.bar(ax=ax)
        ax.set_xlabel('Loan Purpose', size=12)
        ax.set_ylabel('Number of Applications (in millions)', size=12) 
        ax.set_title('Purpose of Loan for Defaulters and Non-Defaulters', size=15)
        plt.show()