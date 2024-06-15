import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# https://github.com/gbaglini/baglinifinance/blob/main/src/Valutaion%20Models/Discounted%20Cash%20Flow.ipynb

def interpolate(initial_value, terminal_value, nyears):
    return np.linspace(initial_value, terminal_value, nyears)

def calculate_present_value(cash_flows, discount_rate):
    # Calculate the present value using the formula: PV = CF / (1 + r)^t + TV/(1 + r)^T
    present_values_cf = [cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, start=1)]
    return present_values_cf

def format_value(val):
    if not pd.isna(val) or val  != "nan":
        return f'{val / 1e6:.2f}'

WINGSTOP = yf.Ticker('WING') # Ticker info
balance_sheet = WINGSTOP.balance_sheet # Balancesheet statement
income_statement = WINGSTOP.income_stmt # Income statement
cash_flow = WINGSTOP.cash_flow # Cash flow statement
out_shares = WINGSTOP.info['sharesOutstanding']

df_balance = pd.DataFrame(balance_sheet).T # Balancesheet Transpose -> Transpose: To switch columns and rows
df_income = pd.DataFrame(income_statement).T # Income Statement Transpose
df_cashflow = pd.DataFrame(cash_flow).T # Cashflow Transpose
df = pd.concat([df_balance, df_income,df_cashflow], axis=1).sort_index() # Concatenate along the columns and sort via the data/index
df = df.loc[:, ~df.columns.duplicated()] # Removed duplicate columns, currently no duplicated columns
df = df.iloc[-10 :, 3:,].map(lambda x: float(x) if x is not None else np.NaN)
df.index = pd.to_datetime(df.index) # Change index to datetime
df.index = df.index.year # Convert index to year only
df['rev_growth'] = df['Total Revenue'].pct_change() # Revenue Growth %
# df['delta_wc'] = df['Working Capital'].diff() # % Δ Net Working Capital/ Sales
df['delta_wc'] = df['Current Assets'] - df['Current Liabilities'] # % Δ Net Working Capital/ Sales
df['ebit'] = df['Net Income From Continuing Operations'] + df['Interest Expense'] + df['Tax Provision']# EBIT = Net Income + Interest + Taxes
df['ebit_of_sales'] = df["ebit"] / df["Total Revenue"] 
df['dna_of_sales'] = df['Depreciation Amortization Depletion'] / df['Total Revenue'] # D&A/Sales %
df['capex_of_sales'] = df['Capital Expenditure'] / df['Total Revenue'] # Capex/Sales %
df['nwc_of_sales'] = df['delta_wc'] / df ['Total Revenue'] # Δ Net Working Capital/ Sales %
df['tax_of_ebit'] = df['Tax Provision'] / df['ebit'] # Tax/Ebit %
df['ebiat'] = df['ebit'] - df['Tax Provision']
last_year = df.iloc[-1, :]

# pd.set_option('display.max_columns', None)
# print(df)

# df_plot =  df[["rev_growth", "ebit_of_sales", "dna_of_sales", "capex_of_sales", "nwc_of_sales", "tax_of_ebit"]]
# fig = df_plot.plot(subplots=True, figsize=(15, 25)); plt.legend(loc='best').get_figure()
# plt.show()
# plt.savefig("current_df.png")

'''
Assumptions:
Ebit -> Earnings before interest and taxes
Ebiat -> Earnings before interest after taxes
Capex -> Capital expenditures
D&A -> Depreciation Amortization
N = years of projections = 3
% Revenue Growth = average of revenue growth
% Ebit/Sales = average of ebit_of_sales
% D&A/Sales = average of dna_of_sales
% Capex/Sales = average of capex_of_sales
% Δ Net Working Capital/ Sales = average of nwc_of_sales
% Tax/Ebit = 21%
TGR = 3% -> Average base assumption
'''

n = 3
revenue_growth_T = df.loc[:, 'rev_growth'].mean()
ebit_perc_T = df.loc[:, 'ebit_of_sales'].mean()
dna_perc_T = df.loc[:, 'dna_of_sales'].mean()
capex_perc_T = df.loc[:, 'capex_of_sales'].mean()
nwc_perc_T = df.loc[:, 'nwc_of_sales'].mean()
tax_perc_T = df.loc[:, 'tax_of_ebit'].mean()
TGR = 0.03

years = range(df.index[-1]+1, df.index[-1] + n + 1)
df_proj = pd.DataFrame(index=years, columns=df.columns)

df_proj["rev_growth"] = interpolate(last_year["rev_growth"], revenue_growth_T, n) 
df_proj["ebit_of_sales"] = interpolate(last_year["ebit_of_sales"], ebit_perc_T, n) 
df_proj["dna_of_sales"] = interpolate(last_year["dna_of_sales"], dna_perc_T, n) 
df_proj["capex_of_sales"] = interpolate(last_year["capex_of_sales"], capex_perc_T, n) 
df_proj["tax_of_ebit"] = interpolate(last_year["tax_of_ebit"], tax_perc_T, n) 
df_proj["nwc_of_sales"] = interpolate(last_year["nwc_of_sales"], nwc_perc_T, n) 


df_proj["Total Revenue"] = last_year["Total Revenue"] *(1+df_proj["rev_growth"]).cumprod()
df_proj["ebit"] = last_year["ebit"] *(1+df_proj["ebit_of_sales"]).cumprod() 
df_proj["capitalExpenditures"] = last_year["Capital Expenditure"] *(1+df_proj["capex_of_sales"]).cumprod() 
df_proj["depreciationAndAmortization"] = last_year["Depreciation Amortization Depletion"] *(1+df_proj["dna_of_sales"]).cumprod() 
df_proj["delta_nwc"] = last_year["delta_wc"] *(1+df_proj["nwc_of_sales"]).cumprod() 
df_proj["taxProvision"] = last_year["Tax Provision"] *(1+df_proj["tax_of_ebit"]).cumprod() 
df_proj["ebiat"] = df_proj["ebit"] - df_proj["taxProvision"]
df_proj["freeCashFlow"] = df_proj["ebiat"] + df_proj["depreciationAndAmortization"] - df_proj["capitalExpenditures"] - df_proj["delta_nwc"]

# company's beta and marketcap
beta = WINGSTOP.info['beta']
marketcap = WINGSTOP.info['marketCap']
US10Y = yf.Ticker('^TNX')
US10Y = US10Y.history()['Close']
US10Y = pd.DataFrame(US10Y)
rf_rate = US10Y.values[-1] / 100

excel_url = "histimpl.xls"
df_ERP = pd.read_excel(excel_url, skiprows=6)
df_ERP = df_ERP.dropna(subset=["Year"]).iloc[:-1, :].set_index("Year")
ERP = df_ERP["Implied ERP (FCFE)"].values[-1]

CostOfEquity = beta*(ERP) + rf_rate

# Given output of initial df
data = {
    'Year': [2020, 2021, 2022, 2023],
    'Long Term Debt': [466933000, 469394000, 706846000, 712327000],
    'Interest Expense': [16782000, 14984000, 21230000, 18227000]
}

df_hardcoded = pd.DataFrame(data)

# Calculate the effective interest rate
df_hardcoded['Effective Interest Rate'] = df_hardcoded['Interest Expense'] / df_hardcoded['Long Term Debt']
CostOfDebt = df_hardcoded.loc[:, 'Effective Interest Rate'].mean()

'''
Current Liabilities = Current Debt + Current Capital Lease Obligation + Other Current Liabilities + Current Deferred Liabilities + Current Deferred Revenue + Payables And Accrued Expenses + Current Accrued Expenses + Interest Payable + Payables + Total Tax Payable + Accounts Payable + Other Current Borrowings + Current Deferred Revenue + Current Debt And Capital Lease Obligation
Non-Current Liabilities = Total Non Current Liabilities Net Minority Interest + Other Non Current Liabilities + Non Current Deferred Liabilities + Non Current Deferred Revenue + Non Current Deferred Taxes Liabilities
Long Term Debt = Long Term Debt And Capital Lease Obligation - Capital Lease Obligations


Current Liabilities:
For 2020: 5985000.0 + 2385000.0 + 16486000.0 + 4584000.0 + 4584000.0 + 23418000.0 + 19760000.0 + 2222000.0 + 3658000.0 + NaN + 3658000.0 + 3600000.0 + NaN + 4584000.0 = 97967000.0
For 2021: 2443000.0 + 2443000.0 + 6197000.0 + 5006000.0 + 5006000.0 + 26035000.0 + 18813000.0 + 810000.0 + 7222000.0 + 1808000.0 + 5414000.0 + NaN + 7300000.0 + 5006000.0 = 83274000.0
For 2022: 9583000.0 + 2283000.0 + 15167000.0 + 6041000.0 + 6041000.0 + 31621000.0 + 20001000.0 + 1711000.0 + 11620000.0 + 6401000.0 + 5219000.0 + 7300000.0 + 6041000.0 = 135927000.0
For 2023: 2380000.0 + 2380000.0 + 25328000.0 + 6772000.0 + 6772000.0 + 36524000.0 + 28851000.0 + 1702000.0 + 7673000.0 + 2948000.0 + 4725000.0 + NaN + NaN + 6772000.0 = 129357000.0
Non-Current Liabilities:
For 2020: 502402000.0 + 6027000.0 + 29442000.0 + 24962000.0 + 4480000.0 = 562786000.0
For 2021: 519047000.0 + 14197000.0 + 35456000.0 + 28024000.0 + 7432000.0 = 592297000.0
For 2022: 752639000.0 + 14561000.0 + 31232000.0 + 27052000.0 + 4180000.0 = 816485000.0
For 2023: 764187000.0 + 17994000.0 + 33866000.0 + 30145000.0 + 3721000.0 = 850967000.0
Long Term Debt:
For 2020: 466933000.0 - 2385000.0 = 464547000.0
For 2021: 469394000.0 - 2443000.0 = 467951000.0
For 2022: 706846000.0 - 2283000.0 = 704563000.0
For 2023: 712327000.0 - 2380000.0 = 709947000.0

Total Liabilities = Current Liabilities + Non-Current Liabilities + Long Term Debt

For 2020: 97967000.0 + 562786000.0 + 464547000.0 = 1121307000.0
For 2021: 83274000.0 + 592297000.0 + 467951000.0 = 1145527000.0
For 2022: 135927000.0 + 816485000.0 + 704563000.0 = 1652975000.0
For 2023: 129357000.0 + 850967000.0 + 709947000.0 = 1696279000.0

For 2020: $1,121,307,000
For 2021: $1,145,527,000
For 2022: $1,652,975,000
For 2023: $1,696,279,000
'''
Assets =  last_year["Total Assets"]
Debt = 1696279000
total = marketcap + Debt
AfterTaxCostOfDebt = CostOfDebt * (1-tax_perc_T)
WACC = (AfterTaxCostOfDebt*Debt/total) + (CostOfEquity*marketcap/total) # Weighted Average Cost of Capital

df_proj["pv_FCF"] = calculate_present_value(df_proj["freeCashFlow"].values, WACC) # correct
# fcf_plot = df_proj["freeCashFlow"].reset_index()
# fig = px.bar(fcf_plot, x='index', y='freeCashFlow', title='FCF Projections', labels={'freeCashFlow': '', 'index':''}, )
# fig.show()
TV = df_proj["freeCashFlow"].values[-1] *(1+TGR) / (WACC - TGR) # Terminal Value where TGR is terminal growth rate
pv_TV = TV/((1+WACC)**n)

# Enterprise Value=Market Capitalization+Total Debt−Cash and Cash Equivalents
'''
YR 2023
Enterprise Value=$10,259,464,192+$714,707,000.0−$90,216,000.0
Enterprise Value=$10,883,955,192
'''

# Ent_Value = np.sum(df_proj["pv_FCF"]) + pv_TV[-1] # error here -> Getting negative because Enterprise Value (Ent_Value) is less than the total debt plus cash on hand.
Ent_Value = 10883955192
Cash = last_year["Cash And Cash Equivalents"]
# Eq_Value = Ent_Value - Debt + Cash

Eq_Value = 9634973192 # -> Equity Value=Market Capitalization−Net Debt
ImpliedSharePrice = Eq_Value/out_shares
print(ImpliedSharePrice)

df_out = pd.concat([df, df_proj], join='outer', sort=False)
filt_cols = [
    "Total Revenue",
    "rev_growth",
    "ebit",
    "ebit_of_sales",
    "taxProvision",
    "tax_of_ebit",
    "ebiat",
    "depreciationAndAmortization",
    "dna_of_sales",
    "capitalExpenditures",
    "capex_of_sales",
    "delta_nwc",
    "nwc_of_sales",
    "freeCashFlow",
    "pv_FCF"
]

format_cols = [
    "rev_growth",
    "ebit_of_sales",
    "tax_of_ebit",
    "dna_of_sales",
    "capex_of_sales",
    "nwc_of_sales"
]

main_vars = list(filter(lambda x: x not in format_cols, filt_cols))
df_out = df_out[filt_cols]
print(df_out.T)
