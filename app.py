import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta
from datetime import date
import numpy as np
from streamlit import session_state as ss

# Dummy user database
users = {
    "acap": "100",
    "user2": "password2"
}

# Logout button callback
def logout_callback():
    ss.logged_in = False

# Check if the user is logged in
if 'logged_in' not in ss or not ss.logged_in:
    st.title("Login to Apeiron Option Book")

    # User input
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check if the login button is pressed and the credentials are correct
    if st.button("Login"):
        if username in users and users[username] == password:
            ss.logged_in = True
            ss.username = username
        else:
            st.error("Incorrect username or password")
else:
    # User is logged in, display the main app (or a logout button)
    # st.title(f"Welcome {ss.username}!")           

    # Load the data
    def load_data():
        csv_file = 'Options_Positions.csv'
        df = pd.read_csv(csv_file)

        df = df[df['Status'] == 'Active']

        # Remove dollar signs and convert 'Notl ($)' to numeric
        df['Notl ($)'] = df['Notl ($)'].replace('[\$,]', '', regex=True).astype(float)

        # Convert 'Expiry Date' to datetime
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])

        # Create 'Option Type' column before filtering
        df['Option Type'] = df['Direction'] + ' ' + df['Call/Put']

        # Adjust notional sizes and apply negative sign for 'sell call' and 'buy put'
        df['Notl ($)'] = df.apply(
            lambda x: -1 * round(x['Notl ($)'], 2) if (x['Call/Put'] == 'Call' and x['Direction'] == 'Sell') 
                                                or (x['Call/Put'] == 'Put' and x['Direction'] == 'Buy') 
                                                else round(x['Notl ($)'], 2),
            axis=1
        )

        return df

    def page1(df):

        # Filter the data to only include active positions
        active_df = df

        # Add a button to choose from available expiries, sorted in ascending order
        unique_expiries = sorted(active_df['Expiry Date'].dt.strftime('%Y-%m-%d').unique())
        selected_expiry = st.selectbox('Select Expiry Date', unique_expiries)

        # Filter the DataFrame based on the selected expiry
        expiry_df = active_df[active_df['Expiry Date'].dt.strftime('%Y-%m-%d') == selected_expiry]

        # Define color mapping for the different option types
        option_types_color = {
            'Sell Call': '#E74C3C',  # Lighter red
            'Buy Put': '#C0392B',    # Darker red
            'Sell Put': '#2ECC71',   # Lighter green
            'Buy Call': '#27AE60'    # Darker green
        }

        # No need to aggregate data since we want to plot individual trades
        # Round the notional sizes to 2 decimal places

        # Plotly bar chart
        fig = px.bar(
            expiry_df,
            x='Ticker',
            y='Notl ($)',
            color='Option Type',  # Use 'Option Type' to determine the color
            color_discrete_map=option_types_color,  # Map 'Option Type' to the colors
            labels={'Notl ($)': 'Notional Size'},
            hover_data={'Ticker': True, 'Notl ($)': ':.2f', 'Option Type': True}
        )

        # Customize hover data to show option type instead of color
        fig.update_traces(
            hovertemplate="<br>".join([
                "Ticker: %{x}",
                "Notional Size: %{y:.2f}",
            ]),
            # text=expiry_df['Option Type']  # This will display the option type on hover
        )

        fig.update_layout(
            width=800,
            height=800
        )

        # Center the chart on the page using container
        with st.container():
            st.write("## Options Notional Size by Ticker")  # Title for the chart
            st.plotly_chart(fig, use_container_width=True)  # Set to True to use the full width of the container

    # Function to plot the graph for a specific ticker and expiry
    def plot_for_ticker(df, ticker):
        df_ticker = df[df['Ticker'] == ticker]
        unique_expiries = sorted(df_ticker['Expiry Date'].unique())
        
        for expiry in unique_expiries:
            df_expiry = df_ticker[df_ticker['Expiry Date'] == expiry]
            fig = px.bar(
                df_expiry,
                x='Strike',
                y='Notl ($)',
                color='Option Type',
                title=f'Notional for {ticker} - Expiry: {expiry.strftime("%Y-%m-%d")}',
                labels={'Notl ($)': 'Notional ($)'},
                text='Notl ($)'
            )
            st.plotly_chart(fig)


    # Page 2 function
    def page2(sub_dfs):
        st.title('Options Notional by Ticker and Expiry')
        
        # Create a dropdown for selecting a ticker
        ticker = st.selectbox('Select Ticker', list(sub_dfs.keys()))
        
        # Once a ticker is selected, plot the graphs for that ticker
        plot_for_ticker(sub_dfs[ticker], ticker)


    def page3():
        
        # Black-Scholes Model
        def black_scholes(S, K, T, r, sigma, option_type="call"):
            """
            Calculates the Black-Scholes option price for a call or put option.

            Parameters:
            S : float - Current stock price
            K : float - Option strike price
            T : float - Time to maturity in years
            r : float - Risk-free interest rate
            sigma : float - Volatility of the underlying asset
            option_type : str - Either "call" or "put"

            Returns:
            float - Black-Scholes option price
            """
            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == "call":
                option_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
            elif option_type == "put":
                option_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
            return option_price


        st.title("Option Payoff Matrix")

        col1, col2, col3, col4 = st.columns(4)

        # User input for BSM parameters
        with col1:
            S = st.number_input("Underlying Spot Price ($)", min_value=0.0, value=3800.0)
        with col2:
            K = st.number_input("Strike Price ($)", min_value=0.0, value=4000.0)
        with col3:
            r = st.number_input("Risk-Free Interest Rate (%)", min_value=0.0, value=5.0) / 100
        with col4:
            sigma_input = st.number_input("Sigma", min_value=0.0, value=50.0) / 100

        # User selects the expiry date using Streamlit's date_input
        expiry_date = st.date_input("Expiry Date")

        # Get the current date
        current_date = date.today()

        # Calculate the difference between expiry_date and current_date
        time_to_expiry = expiry_date - current_date

        # Get the number of days from the resulting timedelta object
        T = time_to_expiry.days / 365

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox("Option Type", ["call", "put"])
        with col2:
            trade_direction = st.selectbox("Trade Direction", ["buy", "sell"])
        orginal_option_price = black_scholes(S, K, T,r,sigma_input, option_type)

        col1, col2 = st.columns(2)
        with col1:
            # Slider for implied volatility (IV)
            iv_slider = st.slider("Implied Volatility (IV) Change Range (%)", -50, 50, (-10, 10))
            iv_step = st.slider("IV Step Size (%)", min_value=1, max_value=5, value=2)

        with col2:
            # Slider for underlying price movement
            price_movement = st.slider("Underlying Price Movement Range (%)", -50, 50, (-10, 10))
            price_step = st.slider("Underlying Price Step Size (%)", min_value=1, max_value=5, value=2)

        st.divider() 

        col1, col2 = st.columns(2)
        col1.metric("Option Price in $", '$ '+ str(round(orginal_option_price,4)))
        col2.metric("Option Price in %", str(round(orginal_option_price/S,6)*100)+' %')

        st.divider() 
        # Calculate matrices
        iv_range = np.arange(iv_slider[0], iv_slider[1], iv_step) / 100
        price_range = np.arange(price_movement[0], price_movement[1], price_step) / 100

        price_matrix = []
        payoff_matrix = []



        iv_list = [round(sigma_input + iv,2) for iv in iv_range]
        price_list = [round(S * (1+ price ),2) for price in price_range]

        #change days left
        def_day = time_to_expiry.days
        days_to_expiry = st.slider("Days Left to Expiry", min_value=1, max_value=100, value=def_day)
        T = days_to_expiry / 365  # Convert days to years

        if st.button('Calculate'):
            
            # st.balloons()


            

            for iv in iv_list:

                price_row =[]
                payoff_row = []
                
                for price in price_list:

                    option_price = black_scholes(price,K,T,r,iv,option_type)

                    price_row.append(option_price / S * 100)

                    if trade_direction == "buy":
                            payoff = option_price / orginal_option_price

                    elif trade_direction == "sell":
                            payoff = 1 - option_price / orginal_option_price if option_price < orginal_option_price else - option_price / orginal_option_price 

                    payoff_row.append(payoff)

                price_matrix.append(price_row)
                payoff_matrix.append(payoff_row)

            z = price_matrix
            z_text = price_matrix

            k = payoff_matrix
            k_text = payoff_matrix

            x=[str('$')+str(o)+"." for o in price_list]
            y=[str(o * 100)+str(' %.') for o in iv_list]

            fig_price = px.imshow(z, 
                            x=x, 
                            y=y, 
                            color_continuous_scale='Aggrnyl', 
                            aspect="auto",
                            text_auto='.2f',
                            labels=dict(x="Underlying Price", y="IV (%)", color="Option Price (%)")
            )
            fig_price.update_xaxes(side="top")

            fig_payoff = px.imshow(k,
                                x = x,
                                y = y,
                                color_continuous_scale='Aggrnyl',
                                aspect="auto",
                                text_auto='.2f',
                                labels=dict(x="Underlying Price", y="IV (%)", color="Payoff Multiple (x)"))
            fig_payoff.update_xaxes(side="top")

            st.plotly_chart(fig_price)
            st.plotly_chart(fig_payoff)



    # Main function to set up the Streamlit app
    def main():
        # Load the CSV file into a DataFrame
        df = pd.read_csv('Options_Positions.csv')
        df = load_data()
        
        # Preprocess the DataFrame
        grouped_df = df
        
        # Group the DataFrame by the 'Ticker' column
        sub_dfs = {ticker: group for ticker, group in grouped_df.groupby('Ticker')}

        # Set up sidebar navigation
        st.sidebar.title("Apeiron Options Book")
        page = st.sidebar.selectbox("", ("Options by Expiry", "Options by Tickers","Options Calculator"))
        st.sidebar.success("Select a page above.")


        if page == "Options by Expiry":
            # call your existing function or page1 function here
            page1(df)
        elif page == "Options by Tickers":
            page2(sub_dfs)
        elif page == "Options Calculator":
            page3()

    if __name__ == "__main__":
        main()# Load the data




