import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Inject custom CSS for larger input boxes
st.markdown(
    """
    <style>
    .stNumberInput > div > div > input {
        font-size: 18px !important;  /* Increase font size */
        padding: 12px !important;    /* Increase padding */
        height: auto !important;     /* Ensure height adjusts accordingly */
    }
    /* CSS for the bottom line and copyright text */
    .bottom-line {
        border-top: 2px solid #e6e6e6;
        margin-top: 50px;
        padding-top: 20px;
        text-align: center;
        color: #888888;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load your model and image
def load_resources():
    model = joblib.load('modelXGBoost.pkl')
    image = Image.open('HTC_interface_.png')
    return model, image

# Function to predict and display results
def predict(model, inputs):
    # Convert to numpy array and reshape for the model
    input_array = np.array(inputs).reshape(1, -1)
    
    # Get predictions
    predictions = model.predict(input_array).round(1)
    
    return predictions

# Main function to run the Streamlit app
def main():
    # Display the title first with increased size
    st.markdown(
        "<h1 style='text-align: center; font-size: 55px;'>Hydrochar Prediction APP</h1>", 
        unsafe_allow_html=True
    )
    
    # Load and display the image
    model, image = load_resources()
    st.image(image, use_column_width=True)
    
    # Title and description
    st.markdown("<p style='color: grey;'>Enter the biomass and operational values below to predict the hydrochar properties:</p>", unsafe_allow_html=True)
    
    # Increase the width of the columns by adjusting the proportions
    col1, col2, col3, col4 = st.columns([3, 3, 2, 3], gap="large")
    
    # Define session state variables for input fields
    if "inputs" not in st.session_state:
        st.session_state.inputs = [None] * 9  # Initialize with None for the inputs
    
    # Biomass values input
    with col1:
        st.subheader("Biomass Values")
        C = st.number_input("C(%):", min_value=0.0, max_value=100.0, value=st.session_state.inputs[0], step=0.1, key="C")
        H = st.number_input("H(%):", min_value=0.0, max_value=100.0, value=st.session_state.inputs[1], step=0.1, key="H")
        N = st.number_input("N(%):", min_value=0.0, max_value=100.0, value=st.session_state.inputs[2], step=0.1, key="N")
        VM = st.number_input("VM(%):", min_value=0.0, max_value=100.0, value=st.session_state.inputs[3], step=0.1, key="VM")
        Ash = st.number_input("Ash(%):", min_value=0.0, max_value=100.0, value=st.session_state.inputs[4], step=0.1, key="Ash")
    
    # Operational values input
    with col2:
        st.subheader("Operational Values")
        RV = st.number_input("RV(ml):", min_value=0.0, max_value=5000.0, value=st.session_state.inputs[5], step=1.0, key="RV")
        RT = st.number_input("RT(min):", min_value=0.0, max_value=1000.0, value=st.session_state.inputs[6], step=1.0, key="RT")
        T = st.number_input("T(ºC):", min_value=0.0, max_value=1000.0, value=st.session_state.inputs[7], step=1.0, key="T")
        BWR = st.number_input("B/W ratio:", min_value=0.0, max_value=1.0, value=st.session_state.inputs[8], step=0.01, key="BWR")
    
    # Predict and Refresh buttons in the third column, centered
    with col3:
        # Calculate the space needed to center the buttons
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)  # Add space above buttons
        
        predict_button = st.button("Predict")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # Add space between buttons
        refresh_button = st.button("Refresh")
        
        if predict_button:
            # Store current inputs in session state
            st.session_state.inputs = [C, H, N, VM, Ash, RV, RT, T, BWR]
            
            # Check if any inputs are None (unfilled)
            if None in st.session_state.inputs:
                st.error("Please fill out all fields before predicting")
            else:
                # Make predictions
                predictions = predict(model, st.session_state.inputs)
                
                # Store predictions in session state to display them later
                st.session_state.predictions = predictions
        
        if refresh_button:
            # Reset all inputs to None (unfilled)
            st.session_state.inputs = [None] * 9
            st.session_state.predictions = None  # Clear predictions
    
    # Prediction values in the fourth column with labels
    with col4:
        st.subheader("Hydrochar Predicted Values")
        output_labels_texts = ['C_h:', 'H_h:', 'N_h:', 'O_h:', 'VM_h:', 'FC_h:', 'Ash_h:', 'Yield_h:']
        
        if "predictions" in st.session_state and st.session_state.predictions is not None:
            for label, prediction in zip(output_labels_texts, st.session_state.predictions[0]):
                st.write(f"**{label}** {prediction:.1f}")
        else:
            # Display labels with empty values if no predictions are made yet
            for label in output_labels_texts:
                st.write(f"**{label}** -")
    
    # Add a note below the input fields with grey text
    st.markdown("<p style='color: grey;'>Note: Make sure data meets following equations: C + H + N + O + Ash = 100% and VM + FC + Ash = 100%</p>", unsafe_allow_html=True)

    # Add the bottom line and copyright notice
    st.markdown(
        """
        <div class="bottom-line">
            Copyright © 2024 Darwin Ortiz
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the main function
if __name__ == "__main__":
    main()
    
    
#to know the path and run in anaconda
#cd C:\Users\Darwin\OneDrive - VŠCHT\IHE thesis\Streamlit
#streamlit run filename.py