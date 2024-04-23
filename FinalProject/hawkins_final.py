from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import ResNet50

# Load pre-trained CNN
base_model = ResNet50(weights='imagenet', include_top=False)

model = Sequential([
    TimeDistributed(base_model, input_shape=(None, frame_height, frame_width, channels)),
    TimeDistributed(Dense(512, activation='relu')),
    LSTM(256),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
