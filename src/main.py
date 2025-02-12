import data_handler as dh
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Conv1D
from sklearn.preprocessing import MinMaxScaler
import train as Train

if __name__ == '__main__':

    # Load and normalize data
    stock = 'AAPL'
    data, labels = dh.process(stock)  # brings the features and labels

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)  # 'data[0]' contains stock features

    X, y = Train.create_sequences(data_scaled, labels)

    # Split into train/test sets
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Train model
    model = Train.build_model(X.shape[1], X.shape[2])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # save the model
    model.save('model.h5')
    print('Done')
