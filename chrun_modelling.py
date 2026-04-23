import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

class churn_model:
    def __init__(self):
        print("Loading the dataset and preprocessing the data...")
        
        # load the dataset
        data = pd.read_csv('Churn_Modelling.csv')
       

        # preprocess the data
        data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        print(data.head(5))

        # encode categorical variables
        lebel_encoder_gender = LabelEncoder()
        data['Gender'] = lebel_encoder_gender.fit_transform(data['Gender'])
        print(data.head(5))

        #encode the 'Geography' column using one-hot encoding
        from sklearn.preprocessing import OneHotEncoder
        onehot_encoder_geography = OneHotEncoder()
        geography_encoded = onehot_encoder_geography.fit_transform(data[['Geography']])
        geography_encoded
        print(geography_encoded.toarray())
        onehot_encoder_geography.get_feature_names_out(['Geography'])

        geography_dataframe = pd.DataFrame(
        geography_encoded.toarray(),
        columns=onehot_encoder_geography.get_feature_names_out(['Geography']),
        index=data.index)
        print(geography_dataframe.head(5))

        data = pd.concat([data, geography_dataframe], axis=1)
        data = data.drop('Geography', axis=1)
        print(data.head(5))

       #save the preprocessed data to a pickle file
        with open('label_encoder_gender.pkl', 'wb') as file:
            pickle.dump(lebel_encoder_gender, file)

        with open('onehot_encoder_geography.pkl', 'wb') as file:
            pickle.dump(onehot_encoder_geography, file)

        X = data.drop('Exited', axis=1)
        y = data['Exited']

        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

        #Scale the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #print the shapes of the training and testing sets
        print("Training set shape:", X_train_scaled.shape)
        print("Testing set shape:", X_test_scaled.shape)

        #pickle the scaler
        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        #sequential model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
        import datetime

        #buid ANN model
        model = Sequential([
            Dense(64,activation='relu', input_shape=(X_train_scaled.shape[1],)), #first hidden layer connected with input layer
            Dense(32, activation='relu'), #second hidden layer
            Dense(1, activation='sigmoid') #output layer
        ])

        print(model.summary())

        import tensorflow as tf
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.losses.BinaryCrossentropy()

        #compile the model

        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

        #setup the TensorBoard callback
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)   

        #early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_callback, tensorboard_callback]
        )
        
        model.save('churn_model.h5')


        
        
      


        




    



if __name__ == "__main__":
    tv = churn_model()