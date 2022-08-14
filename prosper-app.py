import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing

st.write("""
## Team - A
""")
html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Status Prediction App</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.write("""

This app predicts the **Loan Status** of the borrower.
""")

st.sidebar.header('User Input Features')

@st.cache
def calculation(input_df):
    data = pd.read_csv('updated_data.csv')
    dn = data[['AvailableBankcardCredit', 'BankcardUtilization', 'BorrowerAPR',
               'BorrowerRate', 'DebtToIncomeRatio', 'DelinquenciesLast7Years',
               'EmploymentStatus', 'EmploymentStatusDuration',
               'EstimatedEffectiveYield', 'EstimatedLoss', 'EstimatedReturn',
               'IncomeRange', 'Investors', 'LenderYield', 'LoanOriginalAmount',
               'LoanOriginationQuarter', 'LoanStatus', 'Occupation',
               'OpenRevolvingMonthlyPayment', 'ProsperRating (numeric)',
               'ProsperScore', 'RevolvingCreditBalance', 'StatedMonthlyIncome', 'Term',
               'TotalCreditLinespast7years', 'TotalTrades']]
    df_new = dn.drop(columns=['LoanStatus'])
    df = pd.concat([input_df, df_new], axis=0)

    # Lebel Encoding the variables
    LE = LabelEncoder()
    df['IncomeRange'] = LE.fit_transform(df['IncomeRange'])

    # one hot encoding
    # Listing the columns with object datatype
    col = df.dtypes[df.dtypes == 'object'].index
    df_num = pd.get_dummies(data=df, columns=col, drop_first=True)
    X_scaled = preprocessing.StandardScaler().fit(df_num).transform(df_num)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Convert to dataframe
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)

    df = X_pca[:1]  # Selects only the first row (the user input data)
    return df

def turn_off():
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], on_change=turn_off)
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        EmploymentStatus = st.sidebar.selectbox('Employment Status', ('Employed', 'Full-time', 'Not employed',
                                                                      'Part-time', 'Retired', 'Self-employed',
                                                                      'Other'), on_change=turn_off)
        IncomeRange = st.sidebar.selectbox('Income Range', ('$0', '$1-24,999', '$25,000-49,999', '$50,000-74,999',
                                                            '$75,000-99,999', '$100,000+', 'Not employed'), on_change=turn_off)
        LoanOriginationQuarter = st.sidebar.selectbox('Loan Origination Quarter', ('Q3 2009', 'Q4 2009', 'Q1 2010',
                                                                                   'Q2 2010', 'Q3 2010', 'Q4 2010',
                                                                                   'Q1 2011', 'Q2 2011', 'Q3 2011',
                                                                                   'Q4 2011', 'Q1 2012', 'Q2 2012',
                                                                                   'Q3 2012', 'Q4 2012', 'Q1 2013',
                                                                                   'Q2 2013', 'Q3 2013', 'Q4 2013',
                                                                                   'Q1 2014'), on_change=turn_off)
        Occupation = st.sidebar.selectbox('Occupation', ('Accountant/CPA', 'Administrative Assistant', 'Analyst',
                                                         'Architect', 'Attorney', 'Biologist', 'Bus Driver',
                                                         'Car Dealer', 'Chemist', 'Civil Service', 'Clergy',
                                                         'Clerical',
                                                         'Computer Programmer', 'Construction', 'Dentist', 'Doctor',
                                                         'Engineer - Chemical', 'Engineer - Electrical',
                                                         'Engineer - Mechanical' 'Executive', 'Fireman',
                                                         'Flight Attendant', 'Food Service',
                                                         'Food Service Management',
                                                         'Homemaker', 'Investor', 'Judge', 'Laborer', 'Landscaping',
                                                         'Medical Technician', 'Military Enlisted',
                                                         'Military Officer',
                                                         'Nurse (LPN)', 'Nurse (RN)', "Nurse's Aide", 'Other',
                                                         'Pharmacist', 'Pilot - Private/Commercial',
                                                         'Police Officer/Correction Officer', 'Postal Service',
                                                         'Principal', 'Professional', 'Professor', 'Psychologist',
                                                         'Realtor', 'Religious', 'Retail Management',
                                                         'Sales - Commission', 'Sales - Retail', 'Scientist',
                                                         'Skilled Labor', 'Social Worker',
                                                         'Student - College Freshman',
                                                         'Student - College Graduate Student',
                                                         'Student - College Junior',
                                                         'Student - College Senior', 'Student - College Sophomore',
                                                         'Student - Community College',
                                                         'Student - Technical School',
                                                         'Teacher', "Teacher's Aide", 'Tradesman - Carpenter',
                                                         'Tradesman - Electrician', 'Tradesman - Mechanic',
                                                         'Tradesman - Plumber', 'Truck Driver', 'Unknown',
                                                         'Waiter/Waitress'), on_change=turn_off)
        Term = st.sidebar.selectbox('Term in Months', (12, 36, 60), on_change=turn_off)
        AvailableBankcardCredit = st.sidebar.slider('Available Bank card Credit', 0, 33000, 11400, on_change=turn_off)
        BankcardUtilization = st.sidebar.slider('Bank card Utilization', 0.0, 2.5, 0.56, on_change=turn_off)
        BorrowerAPR = st.sidebar.slider('Borrower APR', 0.04, 0.423, 0.22, on_change=turn_off)
        BorrowerRate = st.sidebar.slider('Borrower Rate', 0.04, 0.36, 0.19, on_change=turn_off)
        DebtToIncomeRatio = st.sidebar.slider('Debt To Income Ratio', 0.0, 10.0, 0.26, on_change=turn_off)
        DelinquenciesLast7Years = st.sidebar.slider('Delinquencies Last 7 Years', 0, 99, 0, on_change=turn_off)
        EmploymentStatusDuration = st.sidebar.slider('Employment Status Duration', 0, 755, 100, on_change=turn_off)
        EstimatedEffectiveYield = st.sidebar.slider('Estimated Effective Yield', -0.18, 0.32, 0.17, on_change=turn_off)
        EstimatedLoss = st.sidebar.slider('Estimated Loss', 0.0, 0.37, 0.08, on_change=turn_off)
        EstimatedReturn = EstimatedEffectiveYield - EstimatedLoss
        Investors = st.sidebar.slider('Investors', 1, 1189, 1, on_change=turn_off)
        LenderYield = st.sidebar.slider('Lender Yield', 0.0, 0.36, 0.18, on_change=turn_off)
        LoanOriginalAmount = st.sidebar.slider('Loan Original Amount', 1000, 35000, 9000, on_change=turn_off)
        OpenRevolvingMonthlyPayment = st.sidebar.slider('Open Revolving Monthly Payment', 0, 1180, 430, on_change=turn_off)
        ProsperRatingNumeric = st.sidebar.slider('Prosper Rating (Numeric)', 1, 7, 4, on_change=turn_off)
        ProsperScore = st.sidebar.slider('Prosper Score', 1, 11, 6, on_change=turn_off)
        RevolvingCreditBalance = st.sidebar.slider('Revolving Credit Balance', 0, 45000, 17940, on_change=turn_off)
        StatedMonthlyIncome = st.sidebar.slider('Stated Monthly Income', 0, 15000, 6000, on_change=turn_off)
        TotalCreditLinespast7years = st.sidebar.slider('Total Credit Lines past 7 years', 2, 125, 27, on_change=turn_off)
        TotalTrades = st.sidebar.slider('Total Trades', 1, 122, 24, on_change=turn_off)

        data = {'AvailableBankcardCredit': AvailableBankcardCredit,
                'BankcardUtilization': BankcardUtilization,
                'BorrowerAPR': BorrowerAPR,
                'BorrowerRate': BorrowerRate,
                'DebtToIncomeRatio': DebtToIncomeRatio,
                'DelinquenciesLast7Years': DelinquenciesLast7Years,
                'EmploymentStatus': EmploymentStatus,
                'EmploymentStatusDuration': EmploymentStatusDuration,
                'EstimatedEffectiveYield': EstimatedEffectiveYield,
                'EstimatedLoss': EstimatedLoss,
                'EstimatedReturn': EstimatedReturn,
                'IncomeRange': IncomeRange,
                'Investors': Investors,
                'LenderYield': LenderYield,
                'LoanOriginalAmount': LoanOriginalAmount,
                'LoanOriginationQuarter': LoanOriginationQuarter,
                'Occupation': Occupation,
                'OpenRevolvingMonthlyPayment': OpenRevolvingMonthlyPayment,
                'ProsperRating (numeric)': ProsperRatingNumeric,
                'ProsperScore': ProsperScore,
                'RevolvingCreditBalance': RevolvingCreditBalance,
                'StatedMonthlyIncome': StatedMonthlyIncome,
                'Term': Term,
                'TotalCreditLinespast7years': TotalCreditLinespast7years,
                'TotalTrades': TotalTrades}
        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()



# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)



# Reads in saved classification model
LR = pickle.load(open('LR_pickle.pkl', 'rb'))


safe_html="""  
      <div style="background-color:#8CE57A;padding:10px >
       <h2 style="color:white;text-align:center;">Borrower is Accepted</h2>
       </div>
    """
danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;">Borrower is High Risk type</h2>
       </div>
    """
if "button1" not in st.session_state:
    st.session_state.button1 = False

def callback():
    st.session_state.button1 = True

button1 = st.button("Predict Borrower type",on_click=callback)
threshold = 0.5
if button1 or st.session_state.button1:

    df = calculation(input_df)

    # Apply model to make predictions
    prediction_proba = LR.predict_proba(df)
    st.subheader('Prediction')
    if prediction_proba[0][1] > threshold:
        st.markdown(safe_html, unsafe_allow_html=True)
    else:
        st.markdown(danger_html, unsafe_allow_html=True)
    st.write(' ')

    if st.checkbox("Show Probability") :
        st.subheader('Prediction Probability')
        pred = '{0:.{1}f}'.format(prediction_proba[0][0] * 100, 2)
        pred1 = '{0:.{1}f}'.format(prediction_proba[0][1] * 100, 2)
        st.write('Probability of High Risk is: {}%'.format(pred))
        st.write('Probability of Acceptance is: {}%'.format(pred1))

    if prediction_proba[0][1] > threshold:
        if st.checkbox("Show ROI") :
            st.subheader('Return on Investment (ROI)')
            NetIncome = input_df['LoanOriginalAmount'] * input_df['BorrowerRate'] * input_df['Term'] / 12
            ans = np.array(NetIncome * 100 / input_df['LoanOriginalAmount'])

            NetIncome2 = input_df['LoanOriginalAmount'] * input_df['LenderYield'] * input_df['Term'] / 12
            ans2 = np.array(NetIncome2 * 100 / input_df['LoanOriginalAmount'])

            st.success('ROI When fees are not taken into account : {}%'.format(ans[0]))
            st.success('ROI When fees are taken into account : {}%'.format(ans2[0]))











