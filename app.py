import streamlit as st
import pandas as pd
import plotly.express as px


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
    pass

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
        page = st.sidebar.selectbox("", ("Options by Expiry", "Options by Tickers"))
        st.sidebar.success("Select a page above.")


        if page == "Options by Expiry":
            # call your existing function or page1 function here
            page1(df)
        elif page == "Options by Tickers":
            page2(sub_dfs)

    if __name__ == "__main__":
        main()# Load the data
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
        page = st.sidebar.selectbox("", ("Options by Expiry", "Options by Tickers"))
        st.sidebar.success("Select a page above.")


        if page == "Options by Expiry":
            # call your existing function or page1 function here
            page1(df)
        elif page == "Options by Tickers":
            page2(sub_dfs)

    if __name__ == "__main__":
        main()

    # Logout button
    if st.button("Logout", on_click=logout_callback):
        pass



