import streamlit as st
import os
from src.app import main

if __name__ == "__main__":
    # Set environment variables if needed
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    # Run the Streamlit app
    main()