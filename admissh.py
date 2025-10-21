import streamlit as st
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from mapie.metrics import regression_coverage_score

st.title("Admission Prediction")

st.write("This is a simple web app that predicts the chance of admission to a university based on the GRE score, TOEFL score, university rating, SOP, LOR, CGPA, and research experience.")

st.write("Please enter the following information to predict the chance of admission:")

