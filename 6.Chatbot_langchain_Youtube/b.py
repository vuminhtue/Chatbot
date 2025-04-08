import streamlit as st

# Initialize session state variables
if "form1_data" not in st.session_state:
    st.session_state.form1_data = ""
if "form2_data" not in st.session_state:
    st.session_state.form2_data = ""

# Callback functions to update the state
def update_form1():
    st.session_state.form1_data = st.session_state.input1

def update_form2():
    st.session_state.form2_data = st.session_state.input2

st.title("Two Independent Forms with Persistent State")

# Form 1
with st.form("form1"):
    st.write("Form 1")
    st.text_input("Input in Form 1", key="input1", value=st.session_state.form1_data)
    submit_button1 = st.form_submit_button(label="Submit Form 1", on_click=update_form1)

# Form 2
with st.form("form2"):
    st.write("Form 2")
    st.text_input("Input in Form 2", key="input2", value=st.session_state.form2_data)
    submit_button2 = st.form_submit_button(label="Submit Form 2", on_click=update_form2)

# Display the results from both forms persistently
st.write("### Submitted Data:")
st.write(f"Form 1 Data: {st.session_state.form1_data}")
st.write(f"Form 2 Data: {st.session_state.form2_data}")
