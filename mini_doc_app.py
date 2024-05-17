import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import joblib

# Load pre-trained model and vectorizer
nlp = English()
X_train = pd.read_csv("data/v4_SVM/X_train.csv")
vectorizer = TfidfVectorizer(lowercase=False, max_features=10000, ngram_range=(1, 2))
model = joblib.load('random_forest_model.joblib')

def tokenize_and_lemmatize(text):
  """
  Tokenizes and lemmatizes the input text.

  Args:
      text: A string representing the input text.

  Returns:
      a list of lemmas
  """
  lemmas = []
  
  doc = nlp(text)

  for token in doc:
      if token.is_alpha and not token.is_stop:
          lemmas.append(token.lemma_)
          new_lemma_tokens = " ".join(lemmas)

  return new_lemma_tokens

def predict_condition(user_input):
  """
  Preprocesses user input, predicts skin condition using the SVM model,
  and returns the prediction result.

  Args:
      user_input: A string representing the user's description.

  Returns:
      A string with the predicted skin condition.
  """
  try:
      # Lowercase and remove punctuation
      #user_input = user_input.lower()
      #user_input = "".join([char for char in user_input if char.isalnum() or char.isspace()])
      
      # Tokenize and lemmatize
      # lemmatized_text = tokenize_and_lemmatize(user_input)
      
      # Vectorize the text
      vectorized_input = vectorizer.fit_transform([user_input])
      vectorized_input = pd.DataFrame(vectorized_input.toarray(), columns=vectorizer.get_feature_names_out())
      #vectorized_input = pd.DataFrame(vectorized_input)

      # Load missing columns from training data (if any)
      missing_columns = set(X_train.columns) - set(vectorized_input.columns)
      for col in missing_columns:
          vectorized_input[col] = 0
      

      vectorized_input = vectorized_input[X_train.columns]
      st.write(vectorized_input)
      # Predict using the model
      prediction = model.predict(vectorized_input)[0]

      # Return the prediction result
      return prediction

  except Exception as e:
      # Handle potential errors during preprocessing or prediction
      error_message = f"An error occurred: {str(e)}"
      st.error(error_message)
      return None

# Streamlit app layout and functionality
st.title(":blue[MiniDoc] AI Disease Prediction App")
user_input_text = st.text_area("Describe your condition (as detailed as possible):", height=100)

if st.button("Predict"):
    prediction = predict_condition(user_input_text)
    if prediction:
        st.success(f"Predicted condition: {prediction}. \nPlease consult a healthcare professional. This is not a replacement for medical advice")
    else:
        st.warning("Prediction failed. Please try again and remember to consult a healthcare professional. \nThis app is not a replacement for medical advice")

# Display model information (optional)
st.header("Model Information")
st.header("This app is powered by a trained Random Forest model.", divider = 'rainbow')

st.write("- Symptoms described in natural language to be transformed to Diagnosis")
st.write("- Askhat Aubakirov")
st.write("- askhat.aub.work@gmail.com")
st.write("- https://www.linkedin.com/in/askhattio/")
st.write("- May 17th 2024")

# Add more information about the model or data here if desired.

# Host the app using Streamlit sharing or a cloud platform
# Refer to Streamlit documentation for deployment instructions:
# https://docs.streamlit.io/en/stable/deploy.html
