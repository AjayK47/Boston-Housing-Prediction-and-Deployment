from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the Boston Housing model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred', methods=['POST'])
def predict1():
    # Collect input features from the form
    crim = float(request.form["Crime_rate"])
    zn = float(request.form["Proportion_residential_land"])
    indus = float(request.form["Non_retail_business_acres"])
    chas = float(request.form["Bound_by_Charles_River"])  # Assuming a continuous value for Charles River
    nox = float(request.form["Nitric_oxides_concentration"])
    rm = float(request.form["Average_rooms_per_dwelling"])
    age = float(request.form["Proportion_occupied_units_prior_1940"])
    dis = float(request.form["Weighted_distances_employment_centers"])
    rad = float(request.form["Index_accessibility_radial_highways"])
    tax = float(request.form["Property_tax_rate"])
    ptratio = float(request.form["Pupil_teacher_ratio"])
    b = float(request.form["Proportion_black_residents"])
    lstat = float(request.form["Percentage_lower_status"])

    # Scale the input features using the loaded scaler
    input_data = [[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]
    scaled_data = scaler.transform(input_data)

    # Make predictions using the Boston Housing model
    output = model.predict(scaled_data) * 1000

    return render_template("home.html", result=f"Housing Median cost price in that area is: ${output[0]:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
