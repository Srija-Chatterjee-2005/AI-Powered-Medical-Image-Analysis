from model import build_model
from preprocess import load_data

train, val = load_data()

model = build_model()

model.fit(train, validation_data=val, epochs=5)

model.save("model/model.keras")

print("MODEL TRAINED")