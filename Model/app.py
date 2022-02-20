from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

def model_predict(i):
    if i==1:
        return "Normal"
    else:
        return "Abnormal"
@app.route('/', methods=['GET', 'POST'])
# Main function here
def main():
    
    # If a form is submitted
    if request.method == "POST": # Displaying Result based on values retrieved from Get (Front End)
        
        # Unpickle classifier
        gbc = joblib.load("gbc.pkl")
        
        # Get values through input bars
        pelvic_incidence = request.form.get("Pelvic_incidence")
        pelvic_tilt  = request.form.get("Pelvic_tilt")
        lumbar_lordosis_angle = request.form.get("Lumbar_Lordosis_Angle")
        sacral_slope  = request.form.get("Sacral_slope")
        pelvic_radius = request.form.get("Pelvic_radius")
        degree_spondylolisthesis  = request.form.get("Degree_spondylolisthesis")
        pelvic_slope = request.form.get("Pelvic_slope")
        Direct_tilt = request.form.get("Direct_tilt")
        thoracic_slope = request.form.get("Thoracic_slope")
        cervical_tilt = request.form.get("Cervical_tilt")
        sacrum_angle = request.form.get("Sacrum_angle")
        scoliosis_slope = request.form.get("Scoliosis_slope")
        # Put inputs to dataframe
        X = pd.DataFrame([[pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,degree_spondylolisthesis,pelvic_slope,Direct_tilt,thoracic_slope,cervical_tilt,sacrum_angle,scoliosis_slope]], columns = ["Col1", "Col2","Col3","Col4","Col5","Col6","Col7","Col8","Col9","Col10","Col11","Col12"])
        
        # Get prediction
        prediction = model_predict(gbc.predict(X)[0])      
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
