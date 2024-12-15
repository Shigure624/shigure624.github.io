import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv') # 导入CSV文件
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

le = LabelEncoder()
df['Course'] = le.fit_transform(df['Course'])
df['YearOfStudy'] = le.fit_transform(df['YearOfStudy'])

df_numeric = df.select_dtypes(include=['number'])

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
Data = df_normalized.values # 转成numpy数组Data，大小是N行*M列

# 构建自编码器模型：输入、编码器和解码器
input_layer = Input(shape=(Data.shape[1],))                    # 定义输入层的形状，Data.shape[1] 表示列数，输入数据的特征数为 M
encoded = Dense(10, activation='relu')(input_layer)            # 编码器层，压缩数据到一个较小的潜在空间，比如设置10个神经元，制定relu为激活函数。
decoded = Dense(Data.shape[1], activation='sigmoid')(encoded)  # 解码器层，重建数据到原始特征数，输出层，特征数为 M

# 定义自编码器模型
autoencoder = Model(inputs=input_layer, outputs=decoded)

# 编译模型，使用均方误差作为损失函数
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# 训练模型
autoencoder.fit(Data, Data, epochs=50, batch_size=32, shuffle=True, validation_split=0.1)

# 使用模型进行重建
reconstructed_data = autoencoder.predict(Data)

mse_per_sample = np.mean((Data - reconstructed_data)**2, axis=1)
threshold = np.percentile(mse_per_sample, 95)
is_anomaly = mse_per_sample > threshold
