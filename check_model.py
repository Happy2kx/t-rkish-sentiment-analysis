import pickle

# Sınıflarını kontrol etmek için model yükle
with open('models_balanced/Logistic_Regression.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model type:", type(model))

# Sınıfları almayı dene
if hasattr(model, 'classes_'):
    print("Model classes:", model.classes_)
elif hasattr(model, 'estimator') and hasattr(model.estimator, 'classes_'):
    print("Calibrated model - Base estimator classes:", model.estimator.classes_)
elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'classes_'):
    print("Calibrated model - Base estimator classes:", model.base_estimator.classes_)
else:
    print("Cannot find classes attribute")

# Ayrıca modelin basit bir test için ne tahmin ettiğini kontrol et
print("\n--- Testing Model Prediction ---")
with open('models_balanced/Logistic_Regression_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Pozitif ve negatif örnekleri test et
test_positive = "çok güzel mükemmel harika"
test_negative = "çok kötü berbat rezalet"

vec_pos = vectorizer.transform([test_positive])
vec_neg = vectorizer.transform([test_negative])

pred_pos = model.predict(vec_pos)[0]
pred_neg = model.predict(vec_neg)[0]

print(f"\nPositive test: '{test_positive}' -> Prediction: {pred_pos}")
print(f"Negative test: '{test_negative}' -> Prediction: {pred_neg}")
