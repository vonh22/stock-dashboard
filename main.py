import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import json


def get_stock_data(ticker_symbol):
    """
    Fetch stock data using yfinance
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return None
    

def get_current_price(stock):
    """
    Get current price and price change
    """
    try:
        # Get today's data with 1-minute interval
        today_data = stock.history(period='1d', interval='1m')
        if not today_data.empty:
            current_price = today_data['Close'].iloc[-1]
            prev_close = today_data['Open'].iloc[0]
            price_change = ((current_price - prev_close) / prev_close) * 100
            return current_price, price_change
        return None, None
    except Exception as e:
        st.error(f"Error fetching current price: {str(e)}")
        return None, None

def plot_stock_price(stock, ticker_symbol, timeframe='1mo'):
    """
    Create stock price chart using plotly
    """
    try:
        # Map timeframes to intervals
        interval_mapping = {
            '1d': '15m',    # 15 minute intervals for 1 day
            '1wk': '1h',    # 1 hour intervals for 1 week
            '1mo': '1d',    # Daily intervals for 1 month
            '1y': '1d',     # Daily intervals for 1 year
            '5y': '1wk'     # Weekly intervals for 5 years
        }
        
        interval = interval_mapping.get(timeframe, '1d')
        hist = stock.history(period=timeframe, interval=interval)
        
        if hist.empty:
            st.warning("No historical data available for the selected timeframe.")
            return None
            
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'])])
        
        # Customize the chart appearance
        fig.update_layout(
            title=f'{ticker_symbol} Stock Price ({timeframe})',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_white',
            height=500,
            xaxis_rangeslider_visible=False,  # Hide rangeslider for cleaner look
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def display_company_overview(stock):
    """
    Display company overview information
    """
    info = stock.info
    st.write("### Company Overview")
    st.write(info.get('longBusinessSummary', 'No company description available.'))
    
    # Key company metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Sector", info.get('sector', 'N/A'))
    with cols[1]:
        st.metric("Industry", info.get('industry', 'N/A'))
    with cols[2]:
        market_cap = info.get('marketCap', 0)
        market_cap_b = market_cap / 1e9  # Convert to billions
        st.metric("Market Cap", f"${round(market_cap_b, 2)}B")
    with cols[3]:
        st.metric("Country", info.get('country', 'N/A'))

def format_number(num):
    """
    Format large numbers for display
    """
    if num >= 1e9:
        return f"${round(num/1e9, 2)}B"
    elif num >= 1e6:
        return f"${round(num/1e6, 2)}M"
    else:
        return f"${round(num, 2)}"

def display_key_financials(stock):
    """
    Display key financial information
    """
    info = stock.info
    
    # Valuation Metrics
    st.write("### Valuation Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("P/E Ratio", round(info.get('trailingPE', 0), 2))
    with cols[1]:
        st.metric("P/B Ratio", round(info.get('priceToBook', 0), 2))
    with cols[2]:
        st.metric("Forward P/E", round(info.get('forwardPE', 0), 2))
    with cols[3]:
        st.metric("PEG Ratio", round(info.get('pegRatio', 0), 2))

    # Trading Information
    st.write("### Trading Information")
    cols = st.columns(4)
    with cols[0]:
        st.metric("52 Week High", f"${round(info.get('fiftyTwoWeekHigh', 0), 2)}")
    with cols[1]:
        st.metric("52 Week Low", f"${round(info.get('fiftyTwoWeekLow', 0), 2)}")
    with cols[2]:
        st.metric("50 Day Avg", f"${round(info.get('fiftyDayAverage', 0), 2)}")
    with cols[3]:
        st.metric("200 Day Avg", f"${round(info.get('twoHundredDayAverage', 0), 2)}")

    # Dividend Information
    st.write("### Dividend Information")
    cols = st.columns(3)
    with cols[0]:
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        st.metric("Dividend Yield", f"{round(dividend_yield, 2)}%")
    with cols[1]:
        payout_ratio = info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0
        st.metric("Payout Ratio", f"{round(payout_ratio, 2)}%")
    with cols[2]:
        ex_dividend_date = info.get('exDividendDate', 'N/A')
        if ex_dividend_date != 'N/A':
            ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
        st.metric("Ex-Dividend Date", ex_dividend_date)
    
    # Latest Earnings from Income Statement
    st.write("### Recent Earnings")
    income_stmt = stock.income_stmt
    if not income_stmt.empty:
        net_income = income_stmt.loc['Net Income']
        earnings_data = pd.DataFrame({
            'Period': net_income.index.strftime('%Y-%m-%d'),
            'Net Income': net_income.values
        })
        earnings_data['Net Income'] = earnings_data['Net Income'].apply(format_number)
        st.dataframe(earnings_data, hide_index=True)
    else:
        st.write("No earnings data available")

def display_company_financials(stock):
    """
    Display company financial statements
    """
    # Function to format financial statements
    def format_statement(df):
        if df.empty:
            return df
        # Convert values to millions and round to 2 decimal places
        formatted_df = df.apply(lambda x: x.apply(lambda y: f"${round(y/1e6, 2)}M" if pd.notnull(y) else "N/A"))
        return formatted_df

    # Income Statement
    st.write("### Income Statement (in millions)")
    income_stmt = format_statement(stock.income_stmt)
    if not income_stmt.empty:
        st.dataframe(income_stmt)
    else:
        st.write("No income statement data available")

    # Balance Sheet
    st.write("### Balance Sheet (in millions)")
    balance_sheet = format_statement(stock.balance_sheet)
    if not balance_sheet.empty:
        st.dataframe(balance_sheet)
    else:
        st.write("No balance sheet data available")

    # Cash Flow
    st.write("### Cash Flow Statement (in millions)")
    cash_flow = format_statement(stock.cashflow)
    if not cash_flow.empty:
        st.dataframe(cash_flow)
    else:
        st.write("No cash flow data available")




def get_ai_analysis(company_data):
    """
    Get AI analysis from local Llama model via Ollama with streaming response
    """
    url = "http://localhost:11434/api/generate"
    
    # Prepare the prompt for company analysis
    prompt = f"""You are a financial expert analyst. Analyze this company data and provide insights:

Company Overview:
{company_data['overview']}

Key Financial Metrics:
- Market Cap: {company_data['marketCap']}
- P/E Ratio: {company_data['peRatio']}
- Revenue Growth: {company_data['revenueGrowth']}
- Profit Margins: {company_data['profitMargins']}

Provide a detailed analysis covering:
1. Company's market position and competitive advantages
2. Financial health assessment
3. Growth prospects and potential risks
4. Investment recommendation with supporting rationale

Format the analysis in clear sections with bullet points for key takeaways.
"""

    data = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True
    }
    
    try:
        with st.spinner('Analyzing company data... This may take a few minutes as the AI model processes the information.'):
            # Increased timeout significantly: 30s for connection, 5 minutes for reading
            response = requests.post(
                url, 
                json=data, 
                timeout=(30, 300),  # (connection timeout, read timeout)
                stream=True,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                full_response = ""
                response_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Add an initial message
                response_placeholder.markdown("Generating analysis... You'll see the response appear here in real-time.")
                
                for i, line in enumerate(response.iter_lines()):
                    if line:
                        json_response = json.loads(line)
                        chunk = json_response.get('response', '')
                        full_response += chunk
                        
                        # Update progress and response in place
                        progress = min(1.0, (i + 1) / 50)  # Adjust denominator based on typical response length
                        progress_bar.progress(progress)
                        response_placeholder.markdown(full_response)
                
                # Clear progress bar after completion
                progress_bar.empty()
                return full_response
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return None
                
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. Please ensure Ollama is running and try again.")
        st.info("Tip: Check if the Ollama service is running on your Mac")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to Ollama server")
        st.info("""
        Please check:
        1. Is Ollama running on your Mac?
        2. Is it running on port 11434?
        3. Try running 'ollama serve' in your terminal
        """)
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error: {str(e)}")
        st.info("Please check your network connection and try again")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
        st.info("If this persists, try restarting Ollama")
        return None

def display_ai_analysis(stock):
    """
    Display AI analysis of the company with improved UI feedback
    """
    st.write("### AI-Powered Company Analysis")
    
    # Initialize session state for analysis history if not exists
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Prepare company data for analysis
    info = stock.info
    company_data = {
        'overview': info.get('longBusinessSummary', ''),
        'marketCap': format_number(info.get('marketCap', 0)),
        'peRatio': round(info.get('trailingPE', 0), 2),
        'revenueGrowth': f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else 'N/A',
        'profitMargins': f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else 'N/A'
    }
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.button("Generate New Analysis"):
            analysis = get_ai_analysis(company_data)
            
            if analysis:
                # Add timestamp to analysis
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.analysis_history.append({
                    'timestamp': timestamp,
                    'content': analysis
                })
    
    with col2:
        if st.button("Clear History", help="Clear all previous analyses"):
            st.session_state.analysis_history = []
            st.rerun()
    
    # Display analysis history
    if st.session_state.analysis_history:
        for idx, analysis in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(f"Analysis {len(st.session_state.analysis_history) - idx} - {analysis['timestamp']}", expanded=(idx == 0)):
                st.markdown(analysis['content'])
                
                # Download button for each analysis
                st.download_button(
                    label="Download This Analysis",
                    data=analysis['content'],
                    file_name=f"company_analysis_{analysis['timestamp']}.txt",
                    mime="text/plain",
                    key=f"download_{idx}"
                )
    else:
        st.info("Click 'Generate New Analysis' to get AI-powered insights about this company.")

def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    
    # Simplified sidebar
    with st.sidebar:
        ticker_symbol = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
        analyze_button = st.button("Get Data")

    if analyze_button:
        # Store the stock data in session state so it persists between timeframe changes
        st.session_state.stock = get_stock_data(ticker_symbol)
        st.session_state.ticker = ticker_symbol
    
    # Check if we have stock data in session state
    if hasattr(st.session_state, 'stock') and st.session_state.stock:
        stock = st.session_state.stock
        ticker_symbol = st.session_state.ticker
        
        # Display company name and current price
        info = stock.info
        st.title(f"{info.get('longName', ticker_symbol)} ({ticker_symbol})")
        
        # Get current price and price change
        current_price, price_change = get_current_price(stock)
        
        if current_price is not None and price_change is not None:
            # Color the price change based on whether it's positive or negative
            price_color = "green" if price_change >= 0 else "red"
            st.markdown(
                f"""
                <h3 style='margin-bottom: 0;'>
                    Current Price: ${current_price:.2f} 
                    <span style='color: {price_color};'>
                        ({price_change:+.2f}%)
                    </span>
                </h3>
                """,
                unsafe_allow_html=True
            )
        else:
            # Fallback to basic price display if real-time data isn't available
            hist = stock.history(period='1d')
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
                st.subheader(f"Last Price: ${last_price:.2f}")
            else:
                st.subheader("Price data unavailable")
        
        # Stock price and timeframe selection
        timeframe = st.radio(
            "Select Timeframe:",
            ["1D", "1W", "1M", "1Y", "5Y"],
            horizontal=True,
            key="timeframe"
        )
        
        # Map radio buttons to yfinance periods
        timeframe_mapping = {
            "1D": "1d",
            "1W": "1wk",
            "1M": "1mo",
            "1Y": "1y",
            "5Y": "5y"
        }
        
        # Display stock price chart
        fig = plot_stock_price(stock, ticker_symbol, timeframe_mapping[timeframe])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        
        # Company Overview
        display_company_overview(stock)
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Key Financial Information", "Company Financials", "AI Analysis"])
        
        with tab1:
            display_key_financials(stock)
        
        with tab2:
            display_company_financials(stock)
            
        with tab3:
            display_ai_analysis(stock)

if __name__ == "__main__":
    main()