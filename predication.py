import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

class prediction_model:
    def __init__(self):
        print("load the trained model and the encoders...")
        # load the trained model
        model = load_model('churn_model.h5')    
        with open('onehot_encoder_geography.pkl', 'rb') as file:
            onehot_encoder_geography = pickle.load(file)
        with open('label_encoder_gender.pkl', 'rb') as file:
            lebel_encoder_gender = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)  

        input_data = {
            'CreditScore': 750,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 35,
            'Tenure': 3,
            'Balance': 60000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 70000
        }

        geo_ecoded = onehot_encoder_geography.transform([[input_data['Geography']]]).toarray()  
        geo_ecoded_df = pd.DataFrame(geo_ecoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography'])) 
        print(geo_ecoded_df)  


        input_df = pd.DataFrame([input_data])
        print(input_df)

        ##Encode my categorical variables
        input_df['Gender'] = lebel_encoder_gender.transform(input_df['Gender'])
        print(input_df)
        ##concatenate the encoded geography columns with the original input dataframe
        input_df = pd.concat([input_df, geo_ecoded_df], axis=1)
        input_df = input_df.drop('Geography', axis=1)
        print(input_df)

        ##scale the  the input data
        input_scaled = scaler.transform(input_df)
        print(input_scaled)

        ##make a prediction
        prediction = model.predict(input_scaled)
        print("Prediction (probability of churn):", prediction[0][0])
        print("Predicted class (0 = not churn, 1 = churn):", int(prediction[0][0] > 0.5))

        if prediction[0][0] > 0.5:
            print("The customer is likely to churn.")
        else:
            print("The customer is not likely to churn.")
       









        
        
      


        




    



if __name__ == "__main__":
    tv = prediction_model()
