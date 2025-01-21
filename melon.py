import streamlit as st
import pandas as pd
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import yfinance as yf
import io

###############################################################################
# HELPER 
###############################################################################

def validate_transaction(transaction_type, lot, price, current_cash=None, total_value=None):
    """
    Validate transaction inputs:
      - lot > 0
      - price > 0
      - sufficient cash for BUY orders
    If invalid, raise ValueError.
    """
    if lot < 1:
        raise ValueError("Lot must be at least 1.")
    if price <= 0:
        raise ValueError("Price per share must be > 0.")
    if not transaction_type in ["BUY", "SELL"]:
        raise ValueError("Transaction type must be BUY or SELL.")
    if transaction_type == "BUY" and current_cash is not None and total_value is not None:
        if total_value > current_cash:
            raise ValueError(f"Insufficient funds. Required: Rp {total_value:,.2f}, Available: Rp {current_cash:,.2f}")

def last_day_of_month(any_date: datetime.date) -> datetime.date:
    """
    Returns the last day of the given date's month (28-31).
    """
    next_month = any_date.replace(day=28) + relativedelta(days=4)
    return next_month - relativedelta(days=next_month.day)

def get_latest_prices(df_price):
    """
    Returns dict {stock: latest_close_price} for the maximum date per stock in df_price.
    """
    if df_price.empty:
        return {}
    df_price_sorted = df_price.sort_values(by="Date", ascending=True)
    latest_prices = {}
    for stock_code, grp in df_price_sorted.groupby("Stock"):
        row = grp.iloc[-1]
        latest_prices[stock_code] = row["Close Price"]
    return latest_prices

def get_price_on_date(stock, date_val, df_price):
    """
    Find the latest available close price for stock on or before date_val.
    Returns 0 if no price found.
    """
    mask = (df_price["Stock"] == stock) & (df_price["Date"] <= pd.to_datetime(date_val))
    if mask.any():
        return df_price[mask]["Close Price"].iloc[-1]
    return 0.0

def build_monthly_statements(df_trans, df_price):
    """
    Builds monthly statements showing stock values only, both aggregate and per stock.
    Returns two DataFrames: one for aggregate, one for per stock breakdown.
    """
    if df_trans.empty or df_price.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Convert dates and sort
    df_t = df_trans.copy()
    df_t["Date"] = pd.to_datetime(df_t["Date"])
    df_t = df_t.sort_values(by="Date")

    df_p = df_price.copy()
    df_p["Date"] = pd.to_datetime(df_p["Date"])
    df_p = df_p.sort_values(by="Date")

    # Get date range from actual data
    start_date = df_t["Date"].min()
    end_date = max(df_t["Date"].max(), df_p["Date"].max())

    # Generate EOM dates
    eom_dates = []
    current = datetime.date(start_date.year, start_date.month, 1)
    while current <= end_date.date():
        eom = last_day_of_month(current)
        eom_dates.append(eom)
        current = (eom + datetime.timedelta(days=1)).replace(day=1)

    # For aggregate values
    records_agg = []
    # For per stock breakdown
    records_detailed = []
    
    # Get unique stock ticker
    unique_stocks = set(df_t["Stock"].unique())
    
    # For each EOM date
    for eom_date in eom_dates:
        # Get transactions up to this date
        transactions_to_date = df_t[df_t["Date"].dt.date <= eom_date].copy()
        
        # Track positions per stock
        positions = {}
        
        # Process all transactions chronologically
        for _, trans in transactions_to_date.iterrows():
            stock = trans["Stock"]
            if stock not in positions:
                positions[stock] = {"shares": 0}
            
            shares = trans["Shares"]
            
            if trans["Type"] == "BUY":
                positions[stock]["shares"] += shares
            else:  # SELL
                positions[stock]["shares"] -= shares

        # Calculate stock values
        total_stock_value = 0
        stock_values = {}
        
        for stock in unique_stocks:
            if stock not in positions:
                continue
                
            shares = positions[stock]["shares"]
            if shares > 0:
                # Find the latest price up to eom_date
                mask = (df_p["Stock"] == stock) & (df_p["Date"].dt.date <= eom_date)
                if mask.any():
                    latest_price = df_p[mask]["Close Price"].iloc[-1]
                    stock_value = shares * latest_price
                    total_stock_value += stock_value
                    stock_values[stock] = stock_value
                    
                    # Add to detailed records
                    records_detailed.append({
                        "Month": eom_date,
                        "Stock": stock,
                        "Shares": shares,
                        "Price": latest_price,
                        "Stock Value": stock_value
                    })

        # Add to aggregate records
        records_agg.append({
            "Month": eom_date,
            "Stock Value": total_stock_value
        })

    df_agg = pd.DataFrame(records_agg)
    df_detailed = pd.DataFrame(records_detailed)
    
    return df_agg, df_detailed

def calc_weighted_avg_and_rgain(df_trans, initial_cash=0):
    """
    Loops through transactions in ascending date order,
    applies Weighted Average for BUY, calculates Realized Gain for SELL.
    Now includes proper cash tracking and fixed array length issue.
    """
    df = df_trans.copy()
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    portfolio = {
        "cash": {"amount": initial_cash},  # Start with initial cash balance
    }
    realized_gains = []  # Will store gains for ALL transactions

    for i in range(len(df)):
        row = df.loc[i]
        stock = row["Stock"]
        t_type = row["Type"]
        shares = row["Shares"]
        px = row["Price/Share"]
        total_value = shares * px

        if stock not in portfolio:
            portfolio[stock] = {"shares": 0, "avg_price": 0.0}

        realized_gain = 0.0  # Default for BUY transactions

        if t_type == "BUY":
            # Validate cash available
            if total_value > portfolio["cash"]["amount"]:
                st.warning(f"Warning: Insufficient cash for transaction on {row['Date']}. Proceeding anyway.")
            
            old_sh = portfolio[stock]["shares"]
            old_avg = portfolio[stock]["avg_price"]
            new_sh = old_sh + shares
            if new_sh > 0:
                new_avg = ((old_avg * old_sh) + (px * shares)) / new_sh
            else:
                new_avg = 0.0
            portfolio[stock]["shares"] = new_sh
            portfolio[stock]["avg_price"] = new_avg
            portfolio["cash"]["amount"] -= total_value

        elif t_type == "SELL":
            old_sh = portfolio[stock]["shares"]
            old_avg = portfolio[stock]["avg_price"]
            if shares > old_sh:
                shares_sold = old_sh
                total_value = shares_sold * px
            else:
                shares_sold = shares

            realized_gain = (px - old_avg) * shares_sold
            portfolio[stock]["shares"] = old_sh - shares_sold
            if portfolio[stock]["shares"] == 0:
                portfolio[stock]["avg_price"] = 0.0
            portfolio["cash"]["amount"] += total_value

        realized_gains.append(realized_gain)  # Append for EVERY transaction

    df["Realized Gain"] = realized_gains  
    total_rgain = sum(realized_gains)
    return df, portfolio, total_rgain

def compute_performance_metrics(df_daily):
    """
    Computes simple performance metrics if we have daily portfolio values:
      - Return over the entire period
      - Annualized return (approx)
      - Sharpe ratio (approx) with a fixed risk-free rate assumption
    df_daily must have columns: ['Date', 'PortfolioValue']
    We'll do daily returns = (val[t] - val[t-1]) / val[t-1].
    """
    if df_daily.empty or "PortfolioValue" not in df_daily.columns:
        return {}
    df_daily = df_daily.sort_values(by="Date")
    df_daily["Daily Return"] = df_daily["PortfolioValue"].pct_change()
    df_daily.dropna(inplace=True)

    if df_daily.empty:
        return {}

    # Total return (simple)
    start_val = df_daily.iloc[0]["PortfolioValue"]
    end_val = df_daily.iloc[-1]["PortfolioValue"]
    total_return = (end_val - start_val) / start_val if start_val != 0 else 0

    # Annualized Return (assuming ~252 trading days or ~365 calendar days, we pick a convention)
    days_elapsed = (df_daily.iloc[-1]["Date"] - df_daily.iloc[0]["Date"]).days
    if days_elapsed > 0:
        # using simple approach: (1 + total_return)^(365/days_elapsed) - 1
        annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
    else:
        annualized_return = 0

    # Sharpe Ratio
    # risk_free ~ 2-6%, let's assume 5% / 365 daily
    daily_rf = 0.05 / 365
    df_daily["Excess Return"] = df_daily["Daily Return"] - daily_rf
    avg_excess = df_daily["Excess Return"].mean()
    std_excess = df_daily["Excess Return"].std()
    if std_excess != 0:
        sharpe_ratio = (avg_excess * np.sqrt(365)) / std_excess
    else:
        sharpe_ratio = 0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio
    }

###############################################################################
# STREAMLIT 
###############################################################################

def main():
    st.set_page_config(page_title="Portfolio Tracker 88", layout="wide")

    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        show_auth_page()
        return

    # If user is authenticated, show the normal menu
    menu = [
        "Dashboard",
        "Transactions",
        "Stock Prices",
        "Fetch Real Time Price (Yahoo .JK)",
        "Reports & Visualization",
        "Advanced Analysis",
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Dashboard":
        show_dashboard()
    elif choice == "Transactions":
        show_transactions()
    elif choice == "Stock Prices":
        show_prices()
    elif choice == "Fetch Real Time Price (Yahoo .JK)":
        show_fetch_jk()
    elif choice == "Reports & Visualization":
        show_reports_and_viz()
    elif choice == "Advanced Analysis":
        show_advanced_analysis()

def show_auth_page():
    st.title("Authentication")

    # Make sure we have a place to store users and user data
    if "users" not in st.session_state:
        # Dictionary for credentials: {username: password}
        st.session_state["users"] = {}
    if "user_data" not in st.session_state:
        # Dictionary for per-user data: {username: {...}}
        st.session_state["user_data"] = {}
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None

    # If already logged in, show logout button
    if st.session_state["authenticated"]:
        st.info(f"You are logged in as {st.session_state['current_user']}.")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["current_user"] = None
            st.rerun()
        return  # Stop here

    # Otherwise, let user pick "Login" or "Register"
    mode = st.radio("Select mode", ["Login", "Register"], horizontal=True)

    if mode == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in st.session_state["users"]:
                if st.session_state["users"][username] == password:
                    # Valid login
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = username
                    # Ensure user data subdict
                    if username not in st.session_state["user_data"]:
                        st.session_state["user_data"][username] = {
                            "transactions": [],
                            "prices": [],
                            "daily_values": []
                        }
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid password.")
            else:
                st.error("User not found. Please register first.")

    else:
        st.subheader("Register a New Account")
        new_username = st.text_input("Choose a Username")
        new_password = st.text_input("Choose a Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            # Basic validation
            if not new_username:
                st.error("Username cannot be empty.")
                return
            if not new_password:
                st.error("Password cannot be empty.")
                return
            if new_password != confirm_password:
                st.error("Passwords do not match!")
                return

            # Check if username is available
            if new_username in st.session_state["users"]:
                st.error("Username already taken. Please choose another.")
            else:
                # Register user
                st.session_state["users"][new_username] = new_password

                # Create user_data subdict
                st.session_state["user_data"][new_username] = {
                    "transactions": [],
                    "prices": [],
                    "daily_values": []
                }

                # Auto-login or just success message?
                # Let's do auto-login for convenience:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = new_username

                st.success(f"User '{new_username}' registered and logged in!")
                st.rerun()


def show_dashboard():
    st.title("CuanPro Stock Portfolio Monitor,   Lite version")
    st.markdown("""
    **Features**:
    1. Weighted Average Cost & Realized Gains
    2. Error Handling & Validation (no negative inputs, SELL limited by holdings)
    3. Visualization: line charts, bar charts, pie charts
    4. Advanced Analysis: Annualized Return, Sharpe Ratio, etc.
    5. Export Data to CSV/Excel
    """)
    st.markdown("""
    **Instructions**:
    1. You need to input initial cash balance (this is not meant for trading but to solve "infinite cash" problem when adding new transactions)
    2. Add new transactions in the "Transaction" page, make sure the ticker input is correct
    3. Price data can be added automatically on "Fetch Real Time Price" page if available in Yahoo Finance, if not manually on "Stock Prices" page
    """)

def show_transactions():
    st.title("Transactions (BUY/SELL)")
    
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]

    # Get initial cash balance first
    initial_cash = st.number_input(
        "Initial Cash Balance (Rp)", 
        min_value=0, 
        value=100000000,  # Rp 100M  default
        step=1000000
    )

    st.write("**Note**: 1 Lot = 100 shares.")
    
    # Calculate current available cash based on transaction history
    current_cash = initial_cash
    if user_data["transactions"]:
        df_trans = pd.DataFrame(user_data["transactions"])
        df_trans = df_trans.sort_values(by="Date")
        for _, row in df_trans.iterrows():
            if row["Type"] == "BUY":
                current_cash -= (row["Shares"] * row["Price/Share"])
            else:  # SELL
                current_cash += (row["Shares"] * row["Price/Share"])
    
    st.write(f"Available Cash: Rp {current_cash:,.2f}")

    with st.form("transaction_form"):
        t_type = st.selectbox("Transaction Type", ["BUY", "SELL"])
        stock_code = st.text_input("Stock Code", placeholder="e.g. BBCA, TLKM, or BBCA.JK, etc.")
        date_val = st.date_input("Transaction Date", datetime.date.today())
        lot = st.number_input("Quantity (Lot)", min_value=1, step=1, value=1)
        price = st.number_input("Price per Share (Rp)", min_value=1, step=1, value=1000)
        
        shares = lot * 100
        total_value = shares * price

        submit = st.form_submit_button("Submit Transaction")
        if submit:
            try:
                validate_transaction(t_type, lot, price)
                
                # Additional cash validation for BUY orders
                if t_type == "BUY" and total_value > current_cash:
                    st.error(f"Insufficient funds. Transaction requires Rp {total_value:,.2f} but only Rp {current_cash:,.2f} available.")
                    return
                
                # Additional validation for SELL orders
                if t_type == "SELL":
                    # Calculate current holdings of this stock
                    current_holdings = 0
                    if user_data["transactions"]:
                        df_trans = pd.DataFrame(user_data["transactions"])
                        df_stock = df_trans[df_trans["Stock"] == stock_code.upper().strip()]
                        for _, row in df_stock.iterrows():
                            if row["Type"] == "BUY":
                                current_holdings += row["Shares"]
                            else:
                                current_holdings -= row["Shares"]
                    
                    if shares > current_holdings:
                        st.error(f"Insufficient shares. Attempting to sell {shares} shares but only {current_holdings} available.")
                        return

                # Add transaction
                user_data["transactions"].append({
                    "Date": date_val,
                    "Type": t_type,
                    "Stock": stock_code.upper().strip(),
                    "Lot": lot,
                    "Shares": shares,
                    "Price/Share": float(price),
                    "Total (Rp)": float(total_value),
                })
                
                # Update current cash
                if t_type == "BUY":
                    current_cash -= total_value
                else:
                    current_cash += total_value
                    
                st.success("Transaction added successfully.")
                st.rerun()  # Refresh to show updated cash balance
                
            except ValueError as e:
                st.error(f"Error: {e}")

    st.write("---")
    st.subheader("Transaction History")
    if user_data["transactions"]:
        df_trans = pd.DataFrame(user_data["transactions"])
        df_trans = df_trans.sort_values(by="Date", ascending=False)
        st.dataframe(df_trans)
        csv_data = df_trans.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_data, file_name="transactions.csv")
    else:
        st.info("No transactions recorded.")

def show_prices():
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]

    st.title("Manual Stock Prices (ID Market)")
    st.write("Use this if the ticker isn't available on Yahoo or if data is incomplete.")

    with st.form("price_form"):
        stock_code = st.text_input("Stock Code", placeholder="e.g. BBCA or BBCA.JK")
        date_val = st.date_input("Price Date", datetime.date.today())
        close_price = st.number_input("Close Price (Rp)", min_value=1, step=1, value=1000)
        submit = st.form_submit_button("Submit Price")
        if submit:
            if stock_code.strip() == "":
                st.error("Stock Code cannot be empty.")
            else:
                user_data["prices"].append({
                    "Date": date_val,
                    "Stock": stock_code.upper().strip(),
                    "Close Price": float(close_price),
                })
                st.success("Price record added.")

    st.write("---")
    if user_data["prices"]:
        df_price = pd.DataFrame(user_data["prices"])
        df_price = df_price.sort_values(by="Date", ascending=False)
        st.subheader("All Price Records")
        st.dataframe(df_price)
        # Export
        csv_data = df_price.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_data, file_name="prices.csv")
    else:
        st.info("No price data recorded.")

def show_fetch_jk():
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]
    """
    Fetching Indonesian stock data from Yahoo Finance
    using the .JK suffix. E.g. BBCA.JK for Bank Central Asia.
    """
    st.title("Fetch Indonesian Stocks (.JK) from Yahoo Finance")

    with st.form("fetch_form"):
        st.subheader("Enter Ticker Symbol with .JK suffix (e.g., BBCA.JK, TLKM.JK)")
        col1, col2 = st.columns(2)
        with col1:
            ticker_jk = st.text_input("Ticker Symbol (Yahoo .JK)", "BBCA.JK")
            start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=30))
            end_date = st.date_input("End Date", datetime.date.today())
        with col2:
            pass

        submitted = st.form_submit_button("Fetch Data")
        if submitted:
            if start_date > end_date:
                st.error("Start Date cannot be after End Date.")
            else:
                fetch_id_stock_prices(ticker_jk, start_date, end_date)
                st.success(f"Fetched data for {ticker_jk} from {start_date} to {end_date}.")

    st.write("---")
    st.subheader("Updated Price History")
    if user_data["prices"]:
        df_price = pd.DataFrame(user_data["prices"])
        df_price = df_price.sort_values(by="Date", ascending=False).reset_index(drop=True)
        st.dataframe(df_price)
    else:
        st.info("No stock prices recorded yet.")


def fetch_id_stock_prices(ticker_jk, start_date, end_date):
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]
    """
    Fetch data from Yahoo Finance for Indonesian stocks with .JK suffix.
    Store results in user_data["prices"] with columns:
      Date, Stock, Close Price
    """
    ticker = yf.Ticker(ticker_jk)
    try:
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data returned for {ticker_jk} in the given date range.")
            return
        df = df.reset_index()
        for idx, row in df.iterrows():
            date_val = row["Date"].date()
            close_price = row["Close"]
            user_data["prices"].append({
                "Date": date_val,
                "Stock": ticker_jk.upper(),
                "Close Price": float(close_price),
            })
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker_jk}: {e}")

def show_reports_and_viz():
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]

    st.title("Holdings Report")

    # Get initial cash balance from user
    initial_cash = st.number_input("Initial Cash Balance (Rp)", 
                                 min_value=0, 
                                 value=100000000,  # Rp 100M default
                                 step=1000000)

    df_trans = pd.DataFrame(st.session_state["user_data"][user]["transactions"])
    df_price = pd.DataFrame(user_data["prices"])

    if df_trans.empty:
        st.warning("No transactions found. Please add transactions first.")
        return
    if df_price.empty:
        st.warning("No price data found. Please add price data first.")
        return

    # Add Debugging Code Here
    st.write("Transactions DataFrame (df_trans):", df_trans)
    st.write("Stock Prices DataFrame (df_price):", df_price)

    # Calculate with initial cash
    df_report, portfolio, total_rgain = calc_weighted_avg_and_rgain(df_trans, initial_cash)

    # Current portfolio summary
    latest_dict = get_latest_prices(df_price)
    cash_value = portfolio["cash"]["amount"]
    stock_value = 0
    
    breakdown = []
    for s, data_stk in portfolio.items():
        if s != "cash" and data_stk["shares"] > 0:
            px = latest_dict.get(s, 0)
            val = data_stk["shares"] * px
            stock_value += val
            breakdown.append({
                "Asset": s,
                "Shares": data_stk["shares"],
                "Price": px,
                "Value(Rp)": val
            })

    total_value = cash_value + stock_value

    # Display summary
    st.write("### Portfolio Summary")
    summary_data = {
        "Component": ["Cash", "Stocks", "Total Portfolio"],
        "Value(Rp)": [cash_value, stock_value, total_value],
        "Percentage": [
            cash_value/total_value*100 if total_value else 0,
            stock_value/total_value*100 if total_value else 0,
            100.0
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary)

    # Pie chart for diversification (only cash and stocks)
    if len(breakdown) > 0:
        # Create a new DataFrame only for the pie chart (excluding Total Portfolio)
        df_pie = pd.DataFrame({
            "Component": ["Cash", "Stocks"],
            "Value(Rp)": [cash_value, stock_value]
        })
        fig_pie = df_pie.plot.pie(
            y="Value(Rp)", 
            labels=df_pie["Component"],
            autopct="%.1f%%", 
            figsize=(5,5)
        ).get_figure()
        st.pyplot(fig_pie)

    # EOM-based monthly statements
    st.write("---")
    st.write("### Monthly EOM Portfolio Value")
    df_agg, df_detailed = build_monthly_statements(df_trans, df_price)
    
    # Show aggregate view
    if not df_agg.empty:
        st.write("#### Aggregate Stock Value")
        st.dataframe(df_agg.sort_values(by="Month"))
        st.line_chart(df_agg.set_index("Month")["Stock Value"])
    
    # Show detailed breakdown
    if not df_detailed.empty:
        st.write("#### Stock Value Breakdown by Ticker")
        df_pivot = df_detailed.pivot_table(
            index="Month",
            columns="Stock",
            values="Stock Value",
            aggfunc="sum"
        ).fillna(0)
        st.dataframe(df_pivot)
        
        # Create stacked area chart
        st.write("#### Stock Value Composition Over Time")
        st.area_chart(df_pivot)

    # Realized Gains analysis
    st.write("---")
    st.write("### Realized Gains by Date Range")
    df_report["Date"] = pd.to_datetime(df_report["Date"])
    start_date = st.date_input("Start Date", df_report["Date"].min().date())
    end_date = st.date_input("End Date", df_report["Date"].max().date())
    if start_date > end_date:
        st.error("Start Date must not exceed End Date.")
    else:
        mask = (df_report["Date"].dt.date >= start_date) & (df_report["Date"].dt.date <= end_date)
        df_filtered = df_report.loc[mask]
        realized_sum = df_filtered["Realized Gain"].sum()
        st.write(f"Realized Gain from {start_date} to {end_date}: **Rp {realized_sum:,.2f}**")

    # Top Gainers/Losers
    st.write("---")
    st.write("### Top Gainers / Losers (Realized)")
    df_gain_by_stock = df_report.groupby("Stock")["Realized Gain"].sum().reset_index()
    df_gain_by_stock = df_gain_by_stock.sort_values(by="Realized Gain", ascending=False)
    st.write("#### By Stock (Descending Gains)")
    st.dataframe(df_gain_by_stock)

    # Export options
    st.write("#### Export Options")
    csv_report = df_report.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Transactions+Gains CSV", data=csv_report, file_name="report.csv")

def show_advanced_analysis():
    user = st.session_state["current_user"]
    user_data = st.session_state["user_data"][user]

    st.title("Performance Report")
    
    # Get initial cash balance
    initial_cash = st.number_input(
        "Initial Cash Balance (Rp)", 
        min_value=0, 
        value=100000000,  # Rp 100M default
        step=1000000
    )

    df_trans = pd.DataFrame(st.session_state["user_data"][user]["transactions"])
    df_price = pd.DataFrame(st.session_state["user_data"][user]["prices"])

    if df_trans.empty or df_price.empty:
        st.warning("Please add transactions and price data first.")
        return
    
    # Create df_t from df_trans
    df_t = df_trans.sort_values(by="Date").reset_index(drop=True)
    df_t["Date"] = pd.to_datetime(df_t["Date"])
    df_price["Date"] = pd.to_datetime(df_price["Date"])

    # Build daily timeseries
    min_date = df_t["Date"].min()
    max_date = df_price["Date"].max()
    date_range = pd.date_range(min_date, max_date, freq="D")

    portfolio = {
        "cash": {"amount": initial_cash}  # Initialize with specified cash
    }
    df_daily_vals = []
    t_index = 0

    for single_date in date_range:  
        while t_index < len(df_t) and pd.Timestamp(df_t.loc[t_index, "Date"]) <= pd.Timestamp(single_date):
            row = df_t.loc[t_index]
            stock = row["Stock"]
            t_type = row["Type"]
            px = row["Price/Share"]
            sh = row["Shares"]
            total_value = sh * px

            if stock not in portfolio:
                portfolio[stock] = {"shares": 0, "avg_price": 0}

            if t_type == "BUY":
                old_sh = portfolio[stock]["shares"]
                old_avg = portfolio[stock]["avg_price"]
                new_sh = old_sh + sh
                if new_sh > 0:
                    new_avg = ((old_avg * old_sh) + (px * sh)) / new_sh
                else:
                    new_avg = 0
                portfolio[stock]["shares"] = new_sh
                portfolio[stock]["avg_price"] = new_avg
                portfolio["cash"]["amount"] -= total_value
            elif t_type == "SELL":
                old_sh = portfolio[stock]["shares"]
                sh_sold = min(sh, old_sh)
                total_value = sh_sold * px
                portfolio[stock]["shares"] = old_sh - sh_sold
                if portfolio[stock]["shares"] == 0:
                    portfolio[stock]["avg_price"] = 0
                portfolio["cash"]["amount"] += total_value
            t_index += 1

        # Calculate portfolio values
        cash_val = portfolio["cash"]["amount"]
        stock_val = 0
        for stk, data_stk in portfolio.items():
            if stk != "cash" and data_stk["shares"] > 0:
                px_current = get_price_on_date(stk, single_date, df_price)
                stock_val += data_stk["shares"] * px_current
        
        total_val = cash_val + stock_val

        df_daily_vals.append({
            "Date": single_date,
            "Cash": cash_val,
            "Stock Value": stock_val,
            "Total Value": total_val,
            "Cash %": (cash_val / total_val * 100) if total_val != 0 else 0
        })

    df_daily = pd.DataFrame(df_daily_vals)
    user_data["daily_values"] = df_daily

    # Display portfolio composition
    st.write("### Portfolio Composition")
    latest_row = df_daily.iloc[-1]
    
    summary_data = {
        "Component": ["Cash", "Stocks", "Total Portfolio"],
        "Value(Rp)": [
            latest_row["Cash"],
            latest_row["Stock Value"],
            latest_row["Total Value"]
        ],
        "Percentage": [
            latest_row["Cash %"],
            100 - latest_row["Cash %"],
            100.0
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary)

    # Show daily values
    st.write("### Daily Portfolio Values (Head & Tail)")
    st.dataframe(df_daily[["Date", "Cash", "Stock Value", "Total Value", "Cash %"]].head(5))
    st.dataframe(df_daily[["Date", "Cash", "Stock Value", "Total Value", "Cash %"]].tail(5))

    # Visualization
    st.write("### Portfolio Components Over Time")
    chart_data = df_daily.melt(
        id_vars=["Date"], 
        value_vars=["Cash", "Stock Value", "Total Value"],
        var_name="Component",
        value_name="Value"
    )
    st.line_chart(df_daily.set_index("Date")[["Cash", "Stock Value", "Total Value"]])

    # Compute performance metrics using Total Value
    df_daily_for_metrics = df_daily.rename(columns={"Total Value": "PortfolioValue"})
    metrics = compute_performance_metrics(df_daily_for_metrics)
    if metrics:
        st.write("### Performance Metrics")
        st.write(f"**Total Return**: {metrics['total_return']*100:.2f}%")
        st.write(f"**Annualized Return**: {metrics['annualized_return']*100:.2f}%")
        st.write(f"**Sharpe Ratio**: {metrics['sharpe_ratio']:.3f}")
    else:
        st.info("Not enough daily data to compute performance metrics.")

if __name__ == "__main__":
    main()
