# [LIBRARIES]
import pandas as pd
import numpy as np
import cvxpy as cp
from tabulate import tabulate
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

np.random.seed(42)
random.seed(42)


# [LOAD DATA]

#TECHNOLOGY
AAPL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AAPL-history-daily-ten-yrs.csv"
MSFT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\MSFT-history-daily-ten-yrs.csv"
NVDA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\NVDA-history-daily-ten-yrs.csv"
TSM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\TSM-history-daily-ten-yrs.csv"
SAP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\SAP-history-daily-ten-yrs.csv"
AVGO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AVGO-history-daily-ten-yrs.csv"
STM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\STM-history-daily-ten-yrs.csv"
ORCL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\ORCL-history-daily-ten-yrs.csv"
INTC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\INTC-history-daily-ten-yrs.csv"
AMD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AMD-history-daily-ten-yrs.csv"

#FINANCE
BAC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BAC-history-daily-ten-yrs.csv"
DB = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\DB-history-daily-ten-yrs.csv"
HSBC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\HSBC-history-daily-ten-yrs.csv"
JPM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\JPM-history-daily-ten-yrs.csv"
MS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\MS-history-daily-ten-yrs.csv"
SAN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\SAN-history-daily-ten-yrs.csv"
AXP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\AXP-history-daily-ten-yrs.csv"
BLK = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BLK-history-daily-ten-yrs.csv"
BNPQF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BNPQF-history-daily-ten-yrs.csv"
GS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\GS-history-daily-ten-yrs.csv"
ING = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\ING-history-daily-ten-yrs.csv"
LYG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\LYG-history-daily-ten-yrs.csv"
UBS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\UBS-history-daily-ten-yrs.csv"
WFC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\WFC-history-daily-ten-yrs.csv"

#CONSUMER STAPLES
AMZN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\AMZN-history-daily-ten-yrs.csv"
BUD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\BUD-history-daily-ten-yrs.csv"
DEO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\DEO-history-daily-ten-yrs.csv"
TM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\TM-history-daily-ten-yrs.csv"
KO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\KO-history-daily-ten-yrs.csv"
KR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\KR-history-daily-ten-yrs.csv"
MDLZ = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\MDLZ-history-daily-ten-yrs.csv"
NKE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\NKE-history-daily-ten-yrs.csv"
NSRGF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\NSRGF-history-daily-ten-yrs.csv"
PEP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\PEP-history-daily-ten-yrs.csv"
PG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\PG-history-daily-ten-yrs.csv"
TSN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\TSN-history-daily-ten-yrs.csv"
UL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\UL-history-daily-ten-yrs.csv"

#ENERGY
BP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\BP-history-daily-ten-yrs.csv"
COP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\COP-history-daily-ten-yrs.csv"
CVX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\CVX-history-daily-ten-yrs.csv"
SHEL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\SHEL-history-daily-ten-yrs.csv"
SLB = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\SLB-history-daily-ten-yrs.csv"
TTE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\TTE-history-daily-ten-yrs.csv"
XOM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\XOM-history-daily-ten-yrs.csv"
DVN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\DVN-history-daily-ten-yrs.csv"
EOG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\EOG-history-daily-ten-yrs.csv"
EQNR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\EQNR-history-daily-ten-yrs.csv"
HAL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\HAL-history-daily-ten-yrs.csv"
MPC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\MPC-history-daily-ten-yrs.csv"
OXY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\OXY-history-daily-ten-yrs.csv"
VLO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\VLO-history-daily-ten-yrs.csv"

#INDUSTRIAL
ABBNY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\ABBNY-history-daily-ten-yrs.csv"
EADSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\EADSF-history-daily-ten-yrs.csv"
GE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\GE-history-daily-ten-yrs.csv"
SBGSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\SBGSF-history-daily-ten-yrs.csv"
TSLA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\TSLA-history-daily-ten-yrs.csv"
CAT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\CAT-history-daily-ten-yrs.csv"
ETN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\ETN-history-daily-ten-yrs.csv"
MMM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\MMM-history-daily-ten-yrs.csv"
SDVKY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\SDVKY-history-daily-ten-yrs.csv"

#COMMUNICATION SERVICES
META = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\META-history-daily-ten-yrs.csv"
CMCSA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\CMCSA-history-daily-ten-yrs.csv"
DTEGY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\DTEGY-history-daily-ten-yrs.csv"
GOOGL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\GOOGL-history-daily-ten-yrs.csv"
NFLX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\NFLX-history-daily-ten-yrs.csv"
VOD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\VOD-history-daily-ten-yrs.csv"
DIS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\DIS-history-daily-ten-yrs.csv"
ORANY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\ORANY-history-daily-ten-yrs.csv"
PSO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\PSO-history-daily-ten-yrs.csv"
ITVPY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\ITVPY-history-daily-ten-yrs.csv"
VZ = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\VZ-history-daily-ten-yrs.csv"
T = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\T-history-daily-ten-yrs.csv"

#AEROSPACE & DEFENCE
BAESF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\BAESF-history-daily-ten-yrs.csv"
RTX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\RTX-history-daily-ten-yrs.csv"
AXON = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\AXON-history-daily-ten-yrs.csv"
BA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\BA-history-daily-ten-yrs.csv"
LHX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\LHX-history-daily-ten-yrs.csv"
NOC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\NOC-history-daily-ten-yrs.csv"
TXT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\TXT-history-daily-ten-yrs.csv"

#ESG
BEP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\BEP-history-daily-ten-yrs.csv"
ENPH = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\ENPH-history-daily-ten-yrs.csv"
FSLR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\FSLR-history-daily-ten-yrs.csv"
IBDSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\IBDSF-history-daily-ten-yrs.csv"
PLUG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\PLUG-history-daily-ten-yrs.csv"
VWSYF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\VWSYF-history-daily-ten-yrs.csv"
NEE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\NEE-history-daily-ten-yrs.csv"
SEDG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\SEDG-history-daily-ten-yrs.csv"
RUN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\RUN-history-daily-ten-yrs.csv"
BLDP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\VWSYF-history-daily-ten-yrs.csv"

#SHIPPING
DSDVF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\DSDVF-history-daily-ten-yrs.csv"
FRO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\FRO-history-daily-ten-yrs.csv"
MATX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"
DAC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\DAC-history-daily-ten-yrs.csv"
GOGL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"
KEX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\KEX-history-daily-ten-yrs.csv"
SBLK = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"


pd.set_option('display.max_rows', None)
file = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\stock_analysis.csv"
stock_analysis = pd.read_csv(file, sep=';', header=1)
stock_analysis.columns = stock_analysis.columns.str.strip()

market_caps = {

    "AAPL": 2959,"MSFT": 2734,"TSM": 678,"AVGO": 804,"STM": 18,"ORCL": 361,"INTC": 83,"AMD": 141,
    "BAC": 283,"HSBC": 184,"JPM": 645,"MS": 176,"SAN": 101,"BLK": 136,"BNPQF": 91,"GS": 165,"ING": 57,
    "UBS": 89,"WFC": 211,"AMZN": 1832,"BUD": 130,"TM": 233,"KO": 314,"KR": 47,"MDLZ": 87,"NKE": 82,
    "PEP": 196,"PG": 400,"TSN": 22,"UL": 159,"COP": 112,"SHEL": 195,"SLB": 48,"DVN": 20,"EOG": 61,
    "OXY": 39,"ABBNY": 94,"EADSF": 122,"SBGSF": 131,"TSLA": 776,"MMM": 70,"META": 1271,"GOOGL": 1843,
    "NFLX": 416,"VOD": 23,"DIS": 153,"ORANY": 38,"PSO": 10,"VZ": 185,"RTX": 172,"BA": 122,"LHX": 41,
    "NOC": 78,"TXT": 12,"ENPH": 7,"VWSYF": 13,"NEE": 136,"SEDG": 1,"BLDP": 0.3,"DSDVF": 44,"MATX": 3,
    "DAC": 1,"GOGL": 1,"SBLK": 2
}

stock_analysis["Market cap (billion)"] = stock_analysis["Stock"].map(market_caps)
total_market_cap = stock_analysis["Market cap (billion)"].sum()
stock_analysis["Market Weight"] = stock_analysis["Market cap (billion)"] / total_market_cap


# [FILTER FUNCTION]
def select_candidate_stocks(df, min_auc=0.55, p_thresh=None, max_log_loss=None, require_market_cap=True, include_verdicts=('BUY', 'HOLD')
):

    df_filtered = df[
        df["Investor verdict"].isin(include_verdicts) &
        (df["ROC AUC"] >= min_auc)
    ]

    if p_thresh is not None:
        df_filtered = df_filtered[df_filtered["Predicted P(>2%)"] >= p_thresh]

    if max_log_loss is not None:
        df_filtered = df_filtered[df_filtered["Log Loss"] <= max_log_loss]

    if require_market_cap:
        df_filtered = df_filtered[df_filtered["Market cap (billion)"].notna()]

    return df_filtered.reset_index(drop=True)

selected = select_candidate_stocks(stock_analysis, min_auc=0.55, p_thresh=0.3, require_market_cap=True, include_verdicts=['BUY', 'HOLD']
)
selected["Sector"] = selected["Sector"].replace({
    "Aerosapce & Defence": "Aerospace & Defense",
    "Aerospace & Defence": "Aerospace & Defense"
})

pd.set_option('display.max_rows', None)   
pd.set_option('display.max_columns', None)     
pd.set_option('display.width', 0)              
pd.set_option('display.max_colwidth', None)    
print(selected.to_string(index=False))


sector_colors = {
    "Technology": "#1f77b4",
    "Finance": "#003f5c",
    "Consumer staples": "#ff7f0e",
    "Energy": "#d62728",
    "Industrial": "#7f7f7f",
    "Communication services": "#9467bd",
    "Aerospace & Defense": "#2ca02c",
    "ESG": "#66c2a5", 
    "Shipping": "#17becf"
}

sector_counts = selected['Sector'].value_counts().sort_values(ascending=False)
colors = [sector_colors.get(sector, '#cccccc') for sector in sector_counts.index]

fig_sector = go.Figure(
    data=[
        go.Bar(
            x=sector_counts.index,
            y=sector_counts.values,
            marker=dict(color=colors),
            text=sector_counts.values,
            textposition='outside'
        )
    ]
)

fig_sector.update_layout(
    title="Distribution of Selected Stocks by Sector",
    xaxis_title="Sector",
    yaxis_title="Number of Stocks",
    height=500,
    width=900,
    margin=dict(t=60, b=100),
)

fig_sector.update_xaxes(tickangle=45)
fig_sector.show()



# [VaR FUNCTION]
def calculate_var95(expected_return, volatility):
    
    confidence_level = 0.95
    z_score = 1.645
    var = -(expected_return - z_score * volatility)
    return var


def print_recommended_portfolios(title, portfolios, selected_df):
    print(f"\n=== {title} ===")
    for name, p in portfolios.items():
        print(f"\n{name}")
        print(f"  Expected Return: {p['Expected Return']:.4f}")
        print(f"  Volatility:      {p['Volatility']:.4f}")
        print(f"  Sharpe Ratio:    {p['Sharpe Ratio']:.4f}")
        print(f"  VaR 95%:         {p['VaR 95%']:.4f}")
        print(f"  Assets (k):      {len(p['Stocks'])}")
        
        rows = []
        for stock, weight in zip(p['Stocks'], p['Weights']):
            stock_info = selected_df[selected_df['Stock'] == stock]
            sector = stock_info['Sector'].values[0] if not stock_info.empty else "N/A"
            region = stock_info['Region'].values[0] if not stock_info.empty else "N/A"
            rows.append({
                "Stock": stock,
                "Weight": round(weight, 4),
                "Sector": sector,
                "Region": region
            })
        df_rows = pd.DataFrame(rows)
        print(tabulate(df_rows, headers="keys", tablefmt="fancy_grid", showindex=False))



# [COMPUTE MU]
def calculate_expected_return(predicted_p, historical_mean_return, weight_pred=0.6):
    
    predicted_return_20d = predicted_p * 0.02
    combined_20d = weight_pred * predicted_return_20d + (1 - weight_pred) * historical_mean_return
    annualized_return = (1 + combined_20d) ** (252 / 20) - 1

    return annualized_return

def load_stock_returns(file_path, window=20):

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df['Close'].pct_change(periods=window)

    return df['Return'].dropna().mean()

mu_values = []
for _, row in selected.iterrows():
    ticker = row['Stock']
    predicted_p = row['Predicted P(>2%)']
    
    try:
        path = eval(ticker)  
        hist_return = load_stock_returns(path)
        mu = calculate_expected_return(predicted_p, hist_return)
        mu_values.append(mu)
    except Exception as e:
        print(f"Could not compute mu for {ticker}: {e}")
        mu_values.append(np.nan)

selected['mu'] = mu_values



# [COMPUTE COVARIANCE]
def get_daily_returns(file_path):

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df['Close'].pct_change()

    return df[['Date', 'Return']].dropna()

returns_dict = {}

for _, row in selected.iterrows():

    ticker = row['Stock']
    try:
        path = eval(ticker)
        df_returns = get_daily_returns(path)
        returns_dict[ticker] = df_returns.set_index('Date')['Return']
    except Exception as e:
        print(f"Could not load returns for {ticker}: {e}")

returns_data = pd.DataFrame(returns_dict).dropna()
cov_matrix = returns_data.cov() * 252


def nearest_positive_semidefinite(matrix):

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 0] = 0

    return eigvecs @ np.diag(eigvals) @ eigvecs.T

cov_matrix_psd = nearest_positive_semidefinite(cov_matrix.values)
Sigma = cp.Constant(cov_matrix_psd)



# [COMBINED MONTE CARLO + TRACK BEST PER k]
def simulate_portfolios_and_extract_best(selected_df, mu_vector, cov_matrix, k_values, num_simulations = 100000, lambda_risk_aversion = 0.5):

    tickers = selected_df['Stock'].tolist()
    mu_vector = np.array(mu_vector)
    cov_matrix = np.array(cov_matrix)
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}

    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': [], 'Stocks': [], 'k': []}
    best_per_k = {}

    for _ in range(num_simulations):
        k = random.choice(k_values)
        if k > len(tickers):
            continue

        selected_assets = random.sample(tickers, k)
        indices = [ticker_to_index[t] for t in selected_assets]
        weights = np.random.random(k)
        weights /= np.sum(weights)

        mu_sel = mu_vector[indices]
        sigma_sel = cov_matrix[np.ix_(indices, indices)]

        expected_return = np.dot(weights, mu_sel)
        variance = np.dot(weights.T, np.dot(sigma_sel, weights))
        std_dev = np.sqrt(variance)
        sharpe_ratio = expected_return / std_dev if std_dev > 0 else 0


        results['Returns'].append(expected_return)
        results['Volatility'].append(std_dev)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)
        results['Stocks'].append(selected_assets)
        results['k'].append(k)

        if k not in best_per_k or sharpe_ratio > best_per_k[k]['Sharpe Ratio']:
            best_per_k[k] = {
                'Stocks': selected_assets,
                'Weights': weights,
                'Expected Return': expected_return,
                'Volatility': std_dev,
                'Sharpe Ratio': sharpe_ratio
            }

    return results, best_per_k


top_k_values = [6, 7, 8, 9, 10]
sim_results, optimal_markowitz = simulate_portfolios_and_extract_best(
    selected_df=selected,
    mu_vector=selected['mu'].values,
    cov_matrix=cov_matrix_psd,
    k_values=top_k_values,
    num_simulations = 100000,
    lambda_risk_aversion=0.5
)

for k, portfolio in optimal_markowitz.items():
    expected_return = portfolio['Expected Return']
    volatility = portfolio['Volatility']
    var95 = calculate_var95(expected_return, volatility)
    portfolio['VaR 95%'] = var95


# [EFFICIENT FRONTIER PLOT]
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sim_results['Volatility'],
    y=sim_results['Returns'],
    mode='markers',
    marker=dict(
        color=sim_results['Sharpe'],
        colorscale='ice',
        colorbar=dict(title='Sharpe Ratio'),
        size=5,
        opacity=0.7
    ),
    name='Simulated Portfolios'
))

best_idx = np.argmax(sim_results['Sharpe'])
fig.add_trace(go.Scatter(
    x=[sim_results['Volatility'][best_idx]],
    y=[sim_results['Returns'][best_idx]],
    mode='markers+text',
    marker=dict(color='red', size=15, symbol='star'),
    textposition='bottom center',
    name='Best Sharpe'
))

min_vol_idx = np.argmin(sim_results['Volatility'])
fig.add_trace(go.Scatter(
    x=[sim_results['Volatility'][min_vol_idx]],
    y=[sim_results['Returns'][min_vol_idx]],
    mode='markers+text',
    marker=dict(color='blue', size=15, symbol='star'),
    textposition='bottom center',
    name='Min Volatility'
))

max_ret_idx = np.argmax(sim_results['Returns'])
fig.add_trace(go.Scatter(
    x=[sim_results['Volatility'][max_ret_idx]],
    y=[sim_results['Returns'][max_ret_idx]],
    mode='markers+text',
    marker=dict(color='green', size=15, symbol='star'),
    textposition='bottom center',
    name='Max Return'
))

fig.update_layout(
    title='Efficient Frontier (Standard Markowitz Portfolios)',
    title_font=dict(size=22),
    height=700,
    width=950,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255,255,255,0.6)',
        bordercolor='gray',
        borderwidth=1
    ),
    xaxis_title="Volatility (Annualized)",
    yaxis_title="Expected Annual Return"
)

fig.show()



# [BLACK-LITTERMAN]
def simulate_portfolios_bl(selected_df, mu_vector, cov_matrix, k_values, num_simulations=100000):

    tickers = selected_df['Stock'].tolist()
    mu_vector = np.array(mu_vector)
    cov_matrix = np.array(cov_matrix)
    ticker_to_index = {ticker: i for i, ticker in enumerate(tickers)}

    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': [], 'Stocks': [], 'k': []}
    best_per_k = {}

    for _ in range(num_simulations):
        k = random.choice(k_values)
        if k > len(tickers):
            continue

        selected_assets = random.sample(tickers, k)
        indices = [ticker_to_index[t] for t in selected_assets]
        weights = np.random.random(k)
        weights /= np.sum(weights)

        mu_sel = mu_vector[indices]
        sigma_sel = cov_matrix[np.ix_(indices, indices)]

        expected_return = np.dot(weights, mu_sel)
        variance = np.dot(weights.T, np.dot(sigma_sel, weights))
        std_dev = np.sqrt(variance)
        sharpe_ratio = expected_return / std_dev if std_dev > 0 else 0

        results['Returns'].append(expected_return)
        results['Volatility'].append(std_dev)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)
        results['Stocks'].append(selected_assets)
        results['k'].append(k)

        if k not in best_per_k or sharpe_ratio > best_per_k[k]['Sharpe Ratio']:
            best_per_k[k] = {
                'Stocks': selected_assets,
                'Weights': weights,
                'Expected Return': expected_return,
                'Volatility': std_dev,
                'Sharpe Ratio': sharpe_ratio
            }

    return results, best_per_k


def optimize_black_litterman_sharpe(selected_df, cov_matrix_psd, lambda_risk_aversion=0.5,
                                     tau_values=[0.005, 0.01, 0.025, 0.05],
                                     omega_strengths=[0.05, 0.1, 0.2],
                                     k_values=range(6, 11), num_simulations=100000):
    best_sharpe = -np.inf
    best_combo = None
    best_results = None
    best_optimal = None
    best_mu_bl = None

    Sigma = np.array(cov_matrix_psd)
    market_weights_vector = selected_df['Market Weight'].values.reshape(-1, 1)
    Pi = lambda_risk_aversion * Sigma @ market_weights_vector
    selected_df['pi (implied return)'] = Pi.flatten()
    P = np.eye(len(selected_df))
    Q = selected_df['mu'].values.reshape(-1, 1)

    log_loss_scaled = selected_df['Log Loss'].values
    log_loss_scaled = log_loss_scaled / np.max(log_loss_scaled)

    for tau in tau_values:
        for omega_strength in omega_strengths:
            Omega_diag = omega_strength * log_loss_scaled * np.diag(tau * P @ Sigma @ P.T)
            Omega = np.diag(Omega_diag)

            inv_term = np.linalg.inv(tau * Sigma)
            middle_term = P.T @ np.linalg.inv(Omega) @ P
            right_term = P.T @ np.linalg.inv(Omega) @ Q

            mu_bl = np.linalg.inv(inv_term + middle_term) @ (inv_term @ Pi + right_term)
            mu_bl = mu_bl.flatten()

            sim_results, optimal = simulate_portfolios_bl(
                selected_df=selected_df,
                mu_vector=mu_bl,
                cov_matrix=Sigma,
                k_values=k_values,
                num_simulations=num_simulations
            )

            best_idx = np.argmax(sim_results['Sharpe'])
            max_sharpe = sim_results['Sharpe'][best_idx]

            if max_sharpe > best_sharpe:
                best_sharpe = max_sharpe
                best_combo = (tau, omega_strength)
                best_results = sim_results
                best_optimal = optimal
                best_mu_bl = mu_bl

    selected_df['mu_BL'] = best_mu_bl
    return best_results, best_optimal


sim_results_bl, optimal_bl = optimize_black_litterman_sharpe(

    selected_df=selected,
    cov_matrix_psd=cov_matrix_psd,
    lambda_risk_aversion=0.5,
    tau_values=[0.005, 0.01, 0.025, 0.05],
    omega_strengths=[0.05, 0.1, 0.2],
    k_values=list(range(6, 11)),
    num_simulations=100000
)


for k, portfolio in optimal_bl.items():
    expected_return = portfolio['Expected Return']
    volatility = portfolio['Volatility']
    var95 = calculate_var95(expected_return, volatility)
    portfolio['VaR 95%'] = var95


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sim_results_bl['Volatility'],
    y=sim_results_bl['Returns'],
    mode='markers',
    marker=dict(
        color=sim_results_bl['Sharpe'],
        colorscale='ice',
        colorbar=dict(title='Sharpe Ratio'),
        size=5,
        opacity=0.7
    ),
    name='Simulated Portfolios (BL)'
))

best_idx = np.argmax(sim_results_bl['Sharpe'])
fig.add_trace(go.Scatter(
    x=[sim_results_bl['Volatility'][best_idx]],
    y=[sim_results_bl['Returns'][best_idx]],
    mode='markers+text',
    marker=dict(color='red', size=15, symbol='star'),
    textposition='bottom center',
    name='Best Sharpe (BL)'
))

min_vol_idx = np.argmin(sim_results_bl['Volatility'])
fig.add_trace(go.Scatter(
    x=[sim_results_bl['Volatility'][min_vol_idx]],
    y=[sim_results_bl['Returns'][min_vol_idx]],
    mode='markers+text',
    marker=dict(color='blue', size=15, symbol='star'),
    textposition='bottom center',
    name='Min Volatility (BL)'
))

max_ret_idx = np.argmax(sim_results_bl['Returns'])
fig.add_trace(go.Scatter(
    x=[sim_results_bl['Volatility'][max_ret_idx]],
    y=[sim_results_bl['Returns'][max_ret_idx]],
    mode='markers+text',
    marker=dict(color='green', size=15, symbol='star'),
    textposition='bottom center',
    name='Max Return (BL)'
))

fig.update_layout(
    title='Efficient Frontier (Black-Litterman Portfolios)',
    title_font=dict(size=22),
    height=700,
    width=950,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255,255,255,0.6)',
        bordercolor='gray',
        borderwidth=1
    ),
    xaxis_title="Volatility (Annualized)",
    yaxis_title="Expected Annual Return"
)

fig.show()



# [BACKTESTING PORTFOLIOS]
best_markowitz = max(optimal_markowitz.values(), key=lambda x: x['Sharpe Ratio'])
best_bl = max(optimal_bl.values(), key=lambda x: x['Sharpe Ratio'])
returns_data = pd.DataFrame(returns_dict).dropna()
train_returns = returns_data[returns_data.index < '2021-01-01']
test_returns = returns_data[returns_data.index >= '2021-01-01']


def backtest_portfolio(tickers, weights, return_data):
    portfolio_returns = return_data[tickers].dot(weights)
    portfolio_value = (1 + portfolio_returns).cumprod()
    return portfolio_value

markowitz_backtest = backtest_portfolio(best_markowitz['Stocks'], best_markowitz['Weights'], test_returns)
bl_backtest = backtest_portfolio(best_bl['Stocks'], best_bl['Weights'], test_returns)


def evaluate_backtest(portfolio_value):

    returns = portfolio_value.pct_change().dropna()
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252/len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()
    return cagr, vol, sharpe, max_drawdown

def print_backtest_results(name, portfolio_value):

    cagr, vol, sharpe, dd = evaluate_backtest(portfolio_value)
    print(f"\n{name} Backtest Performance (2021–2025):")
    print(f"  CAGR:           {cagr:.2%}")
    print(f"  Volatility:     {vol:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Max Drawdown:   {dd:.2%}")


def get_backtest_metrics(name, portfolio_value):
    returns = portfolio_value.pct_change().dropna()
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_dd = ((portfolio_value / portfolio_value.cummax()) - 1).min()
    return {
        "Portfolio": name,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }


SPY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Benchmarks\SPY-history.csv"
spy_data = pd.read_csv(SPY, index_col="Date", parse_dates=True)
spy_series = spy_data["Close"]
spy_returns = spy_series.pct_change().dropna()
spy_backtest = (1 + spy_returns).reindex(test_returns.index).fillna(0).cumprod()


min_vol_idx_m = np.argmin(sim_results['Volatility'])
max_ret_idx_m = np.argmax(sim_results['Returns'])
min_vol_idx_bl = np.argmin(sim_results_bl['Volatility'])
max_ret_idx_bl = np.argmax(sim_results_bl['Returns'])

min_vol_portfolio_m = {'Stocks': sim_results['Stocks'][min_vol_idx_m], 'Weights': sim_results['Weights'][min_vol_idx_m]}
max_ret_portfolio_m = {'Stocks': sim_results['Stocks'][max_ret_idx_m], 'Weights': sim_results['Weights'][max_ret_idx_m]}
min_vol_portfolio_bl = {'Stocks': sim_results_bl['Stocks'][min_vol_idx_bl], 'Weights': sim_results_bl['Weights'][min_vol_idx_bl]}
max_ret_portfolio_bl = {'Stocks': sim_results_bl['Stocks'][max_ret_idx_bl], 'Weights': sim_results_bl['Weights'][max_ret_idx_bl]}


named_portfolios = [
    ("Markowitz (Best Sharpe)", best_markowitz),
    ("Black-Litterman (Best Sharpe)", best_bl),
    ("Markowitz (Min Volatility)", min_vol_portfolio_m),
    ("Black-Litterman (Min Volatility)", min_vol_portfolio_bl),
    ("Markowitz (Max Return)", max_ret_portfolio_m),
    ("Black-Litterman (Max Return)", max_ret_portfolio_bl)
]


for name, portfolio in named_portfolios:
    weights = portfolio["Weights"]
    stocks = portfolio["Stocks"]

    expected_return = np.dot(weights, selected.set_index("Stock").loc[stocks, "mu"])
    vol_vector = returns_data[stocks].cov().values
    volatility = np.sqrt(np.dot(weights, np.dot(vol_vector, weights))) * np.sqrt(252)
    sharpe = expected_return / volatility
    var95 = -(expected_return - 1.645 * volatility)

    print(f"\n{name.upper()}")
    print(f"Expected Return: {expected_return:.4f}")
    print(f"Volatility:      {volatility:.4f}")
    print(f"Sharpe Ratio:    {sharpe:.4f}")
    print(f"VaR 95%:         {var95:.4f}")

    rows = []
    for stock, weight in zip(stocks, weights):
        info = selected[selected["Stock"] == stock]
        sector = info["Sector"].values[0] if not info.empty else "N/A"
        region = info["Region"].values[0] if not info.empty else "N/A"
        rows.append({
            "Stock": stock,
            "Weight": round(weight, 4),
            "Sector": sector,
            "Region": region
        })
    df = pd.DataFrame(rows)
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))



min_vol_m_backtest = backtest_portfolio(min_vol_portfolio_m['Stocks'], min_vol_portfolio_m['Weights'], test_returns)
max_ret_m_backtest = backtest_portfolio(max_ret_portfolio_m['Stocks'], max_ret_portfolio_m['Weights'], test_returns)
min_vol_bl_backtest = backtest_portfolio(min_vol_portfolio_bl['Stocks'], min_vol_portfolio_bl['Weights'], test_returns)
max_ret_bl_backtest = backtest_portfolio(max_ret_portfolio_bl['Stocks'], max_ret_portfolio_bl['Weights'], test_returns)
results = [
    get_backtest_metrics("Markowitz (Best Sharpe)", markowitz_backtest),
    get_backtest_metrics("Black-Litterman (Best Sharpe)", bl_backtest),
    get_backtest_metrics("Markowitz (Min Vol)", min_vol_m_backtest),
    get_backtest_metrics("Markowitz (Max Ret)", max_ret_m_backtest),
    get_backtest_metrics("Black-Litterman (Min Vol)", min_vol_bl_backtest),
    get_backtest_metrics("Black-Litterman (Max Ret)", max_ret_bl_backtest),
    get_backtest_metrics("SPY Benchmark", spy_backtest),
]

performance_df = pd.DataFrame(results)
performance_df["CAGR"] = performance_df["CAGR"].apply(lambda x: f"{x:.2%}")
performance_df["Volatility"] = performance_df["Volatility"].apply(lambda x: f"{x:.2%}")
performance_df["Sharpe Ratio"] = performance_df["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
performance_df["Max Drawdown"] = performance_df["Max Drawdown"].apply(lambda x: f"{x:.2%}")
print(performance_df)


fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=markowitz_backtest, x=markowitz_backtest.index, name="Markowitz Best Sharpe",
                          line=dict(color="green", width=2)))
fig1.add_trace(go.Scatter(y=bl_backtest, x=bl_backtest.index, name="BL Best Sharpe",
                          line=dict(color="red", width=2)))
fig1.add_trace(go.Scatter(y=spy_backtest, x=spy_backtest.index, name="SPY Benchmark",
                          line=dict(color="black", width=2)))

fig1.update_layout(
    title="Backtest: Best Sharpe Portfolios vs. Benchmarks",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
)
fig1.show()


fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=max_ret_m_backtest, x=max_ret_m_backtest.index, name="Markowitz Max Return",
                          line=dict(color="green", width=2)))
fig2.add_trace(go.Scatter(y=max_ret_bl_backtest, x=max_ret_bl_backtest.index, name="BL Max Return",
                          line=dict(color="red", width=2)))
fig2.add_trace(go.Scatter(y=spy_backtest, x=spy_backtest.index, name="SPY Benchmark",
                          line=dict(color="black", width=2)))

fig2.update_layout(
    title="Backtest: Max Return Portfolios vs. Benchmarks",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
)
fig2.show()


fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=min_vol_m_backtest, x=min_vol_m_backtest.index, name="Markowitz Min Vol",
                          line=dict(color="green", width=2)))
fig3.add_trace(go.Scatter(y=min_vol_bl_backtest, x=min_vol_bl_backtest.index, name="BL Min Vol",
                          line=dict(color="red", width=2)))
fig3.add_trace(go.Scatter(y=spy_backtest, x=spy_backtest.index, name="SPY Benchmark",
                          line=dict(color="black", width=2)))

fig3.update_layout(
    title="Backtest: Min Volatility Portfolios vs. Benchmarks",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
)
fig3.show()


