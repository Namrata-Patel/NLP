import streamlit as st
import pandas as pd

st.title('Open Tickets')

# Retrieve ticket data from session state
ticket_dict = st.session_state.get('tickets', {})

# Combine all tickets from different departments into one list
all_tickets = []
for department, tickets in ticket_dict.items():
    for ticket in tickets:
        all_tickets.append((department, ticket[0], ticket[1]))  # department, ticket generation time, ticket query

# Create a DataFrame from the combined tickets
ticket_df = pd.DataFrame(all_tickets, columns=['Department', 'Ticket Generation Time', 'Ticket Query'])

# Sort the DataFrame by ticket generation time in descending order
ticket_df = ticket_df.sort_values(by='Ticket Generation Time', ascending=False)

# Reset the index of ticket_df starting from 1
ticket_df.reset_index(drop=True, inplace=True)
ticket_df.index += 1

# Display the DataFrame as a table
st.write("All Tickets:")
st.table(ticket_df)
