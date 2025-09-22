import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


ticker = 'TSLA'
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(months=3) 

df = yf.download(ticker, start=start_date, end=end_date, interval='1d')

# Features
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


look_back = 25  
forecast_out = 14  #how much do I want predicted
if len(scaled_data) < look_back + forecast_out:
    raise ValueError(f"Not enough data to create sequences. Dataset has {len(scaled_data)} rows, "
                     f"but need at least {look_back + forecast_out}.")

# Create sequences
def create_sequences(data, look_back, forecast_out):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_out + 1):
        X.append(data[i:i + look_back, :])  
        y.append(data[i + look_back:i + look_back + forecast_out, 0])  
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back, forecast_out)
print("Shape of X:", X.shape) 
print("Shape of y:", y.shape)  


X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

input_size = X_train.shape[2] 
hidden_layer_size = 50
output_size = forecast_out

model = LSTM(input_size, hidden_layer_size, output_size)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 20
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 5 == 0:
        print(f'Epoch {i}, Loss: {single_loss.item()}')

with torch.no_grad():
    last_sequence = torch.tensor(scaled_data[-look_back:], dtype=torch.float32)
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))
    forecast_set = model(last_sequence).numpy()

forecast_set = scaler.inverse_transform(
    np.hstack((forecast_set.reshape(-1, 1), np.zeros((forecast_set.shape[0], scaled_data.shape[1] - 1))))
)[:, 0]


df['Forecast'] = np.nan
last_date = df.index[-1]
next_business_day = last_date + pd.offsets.BDay(1)

for i in forecast_set:
    df.loc[next_business_day] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    next_business_day += pd.offsets.BDay(1)


plt.figure(figsize=(12, 6))
df['Close'].plot(label='Actual Price')
df['Forecast'].plot(label='Forecasted Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(ticker)
plt.show()
