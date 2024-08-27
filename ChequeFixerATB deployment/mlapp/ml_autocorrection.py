import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from paddleocr import PaddleOCR
import sqlite3
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Layer
import language_tool_python
from num2words import num2words
import re
def segmentation(image_path):

  image = cv2.imread(image_path)
  #remove bank logo
  (b, g, r) = image[100, 1000]
    # trying to change the pixels values to that colore
  for i in range(0, 200):
        for j in range(0, 1820):
            image[i, j] = (b, g, r)
    # Remove OR BEARER
  (b2, g2, r2) = image[500, 750]
  for i in range(240, 350):
        for j in range(1880, 2365):
            image[i, j] = (b2, g2, r2)
  for i in range(0, 83):
        for j in range(1750, 2340):
            image[i, j] = (b, g, r)
    # Attemping to remove the line under "RUPEES line" without losing data details
  # Get the color of the pixel at coordinates (500, 750) to remove the line
  (b3, g3, r3) = image[500, 750]
  for i in range(405, 510):
          for j in range(0, 1780):
              image[i, j] = (b3, g3, r3)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image=cv2.medianBlur(image,5)
  image=cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)
 
  id_image = image[920:1200, 580:900]
  image = image[40:530, 200:2285]
  # Extract the cheque words amount from the image
  words_image = image[255:480, 0:1490]
  words_image=cv2.erode(words_image,np.ones((2,2),np.uint8),iterations=1)
  amount_image = image[330:450, 1450:2085]
  client_image = image[150:290, 0:1490]
  date_image = image[30:109, 1520:2090]
  return amount_image, date_image, words_image, client_image, id_image
def preprocess_date_image(image_path):
    # Load the image in grayscale
    image = image_path

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= 9 and w >= 8:  # Filter small contours
            digit_image = binary_image[y:y + h, x:x + w]
            # Add margins around the digit
            margin = 9
            digit_image = cv2.copyMakeBorder(digit_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # Resize to 28x28 pixels
            digit_image = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
            digit_images.append(digit_image)

    # If the number of extracted digits is not 8, display a warning message
    if len(digit_images) != 8:
        print(f"Warning: {len(digit_images)} digits detected, 8 digits expected.")

    return digit_images
import numpy as np
import cv2
from matplotlib import pyplot as plt



def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated_image

# Preprocessing function for amount image
def preprocess_amount_image(image_path):
    image = image_path

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary_image = rotate_image(binary_image, -2)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= 16 and w >= 5 and h <= 39 and w <= 35:  # Adjusted width criteria to include commas
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)
            digit_image = binary_image[y:y + h, x:x + w]
            margin = 6
            digit_image = cv2.copyMakeBorder(digit_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            digit_image = rotate_image(digit_image, 5)
            digit_image = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
            digit_images.append(digit_image)

    return digit_images
def preprocess_id_image(image_path):
    # Load the image in grayscale
    image = image_path


    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to enhance digit clarity
   

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digit_images = []
    for i, contour in enumerate(contours):
        if i == 0 or i == len(contours) - 1:
            continue  # Ignore the first and last contours

        x, y, w, h = cv2.boundingRect(contour)

        # Ensure the contour has reasonable height and width
        if h >= 10 and w >= 5:
            # Extract the digit image region
            digit_image = binary_image[y:y + h, x:x + w]

            # Add margins around the digit
            margin = 13
            digit_image = cv2.copyMakeBorder(digit_image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Rotate the digit image slightly
            digit_image = rotate_image(digit_image, -4)

            # Resize to 28x28 pixels
            digit_image = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)

            digit_images.append(digit_image)

    return digit_images, binary_image
def predicted_date(image_path):
    digit_images = preprocess_date_image(image_path)


    digit_str = ""

    # Load the trained kNN model data
    with np.load('mlapp/knn_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']

    # Initialize kNN
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    for image in digit_images:
        # Resize image to 20x20 pixels to match training data
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
        image = image.reshape(-1, 400).astype(np.float32)

        # Predict the class
        ret, result, neighbours, dist = knn.findNearest(image, k=5)
        digit = int(result[0][0])
        digit_str += str(digit)

    digit_str = list(digit_str)
    digit_str.insert(2, '/')
    digit_str.insert(5, '/')
    final_string = ''.join(digit_str)
    return final_string
def insert_commas(digit_str):
    # Initialiser la chaîne résultante
    result = ''

    # Initialiser le compteur pour les positions
    count = 0

    # Parcourir la chaîne de chiffres
    i = 0
    while i < len(digit_str):
        if count == 3:
            # Insérer une virgule si nécessaire
            if i < len(digit_str) and digit_str[i] == '1':
                result += ''
                count = 1
                i += 1
            else:
                result += digit_str[i]
                count = 1
                i += 1
        elif count == 6:
            # Insérer une virgule si nécessaire
            if i < len(digit_str) and digit_str[i] == '1':
                result += ''
                count = 1
                i += 1
            else:
                result += digit_str[i]
                count = 1
                i += 1
        else:
            # Ajouter le chiffre et incrémenter le compteur
            result += digit_str[i]
            count += 1
            i += 1

    return result

def predicted_amount(image_path):
    To_predict_images = preprocess_amount_image(image_path)
    digit_str = ""
    hindi_numerals = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
    }

    with np.load('mlapp/knn_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    for image in To_predict_images:
        image = cv2.resize(image, (20, 20), interpolation=cv2.INTER_AREA)
        image = image.reshape(-1, 400).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(image, k=4)
        digit = int(result[0][0])
        digit_str += hindi_numerals.get(digit, '')

    # Remove possible misplaced commas from digit_str
    digit_str = digit_str.replace(',', '')

    # Insert commas based on custom logic
    formatted_amount = insert_commas(digit_str)

    return formatted_amount
def paddle_ocr_id(image_path):
  
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Perform OCR on the image
    result = ocr.ocr(image_path, cls=True)

    # Extract text from the result
    extracted_text = []
    for line in result:
        for word_info in line:
            extracted_text.append(word_info[-1][0])

    # Combine the extracted text into a single string
    digit_str = ''.join(extracted_text)

    # Remove any non-numeric characters
    cleaned_digit_str = ''.join(filter(str.isdigit, digit_str))

    return cleaned_digit_str
def preprocess_image(image_path):
    # Charger l'image en niveaux de gris
    image = image_path


    binary_image = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)

    # Appliquer la dilatation
    kernel = np.ones((1, 1), np.uint8)
    dilated_image = cv2.erode(binary_image, kernel, iterations=1)

    return dilated_image
def extract_text_from_image(image_path):
    # Prétraiter l'image
    preprocessed_image = preprocess_image(image_path)

    # Extraire le texte avec Tesseract
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 ')

    return text
def preprocess_and_extract_ocr(image_path):
    # Make sure segmentation returns the correct image data
    amount_image, date_image, words_image, client_image, id_image = segmentation(image_path)

    amount_text = predicted_amount(amount_image)
    date_text = predicted_date(date_image)
    words_text = extract_text_from_image(words_image)
    client_text = extract_text_from_image(client_image)
    id_text = paddle_ocr_id(id_image)

    return {
        'amount': amount_text,
        'date': date_text,
        'words': words_text,
        'client': client_text,
        'id': id_text
    }
def extract_text_from_data(data):
    text = []
    for i, item in enumerate(data.splitlines()):
        if i == 0:
            continue
        item = item.split()
        if len(item) == 12:
            text.append(item[11])
    return ' '.join(text).strip()

#function to clean up the text
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text)).strip()
#function to convert numerical amounts to words in Hindis
def number_to_hindi_words(amount):
    words = num2words(amount, lang='en_IN')
    return words

class AttentionLayer(Layer):
    def call(self, decoder_outputs, encoder_outputs):
        # Compute attention scores
        scores = tf.matmul(decoder_outputs, encoder_outputs, transpose_b=True)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Compute context vector
        context_vector = tf.matmul(attention_weights, encoder_outputs)
        
        # Concatenate context vector with decoder outputs
        combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)
        return combined_context

def load_seq2seq_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

def seq2seq_correction(seq2seq_model, text, tokenizer, max_sequence_length):
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")
    
    if max_sequence_length is None:
        raise ValueError("max_sequence_length cannot be None")
    
    # Tokenize and pad the input text
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
    
    # Predict using the seq2seq model
    predicted_seq = seq2seq_model.predict([padded_text, padded_text])
    
    # Convert the predicted sequence back to text
    corrected_tokens = np.argmax(predicted_seq, axis=-1)
    corrected_text = tokenizer.sequences_to_texts(corrected_tokens)
    
    return clean_text(corrected_text[0])

def language_tool_correction(extracted_text):
    tool = language_tool_python.LanguageTool('en')
    matches = tool.check(extracted_text)
    corrected_text = language_tool_python.utils.correct(extracted_text, matches)
    return clean_text(corrected_text)

def correct_text_using_best_method(image_path, seq2seq_model_path='/Users/mac/ChequeFixerATB/mlapp/seq2seq_model.keras', tokenizer=None, max_sequence_length=None):
    ocr_results = preprocess_and_extract_ocr(image_path)
    extracted_text = ocr_results['words']
    amount_digit = ocr_results['amount']

    # Load the Seq2Seq model
    seq2seq_model = load_seq2seq_model(seq2seq_model_path)
    
    # Convert the amount to Hindi words
    converted_text = number_to_hindi_words(amount_digit)
    
    # Correct text using Seq2Seq model if tokenizer and max_sequence_length are provided
    is_correct_seq2seq = False
    corrected_text_seq2seq = ""
    if tokenizer is not None and max_sequence_length is not None:
        corrected_text_seq2seq = seq2seq_correction(seq2seq_model, extracted_text, tokenizer, max_sequence_length)
        is_correct_seq2seq = (corrected_text_seq2seq == extracted_text)
    
    # Correct text using LanguageTool
    corrected_text_lt = language_tool_correction(extracted_text)
    is_correct_lt = (corrected_text_lt == extracted_text)
    
    # Choose the method with better performance
    if is_correct_seq2seq:
        final_corrected_text = corrected_text_seq2seq
    elif is_correct_lt:
        final_corrected_text = corrected_text_lt
    else:
        # Default to converted text if neither method is correct
        final_corrected_text = converted_text
    
    return final_corrected_text, is_correct_seq2seq or is_correct_lt
def initialize_database(db_path):
    """Initialize the SQLite database and create the required tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table for storing corrected cheque data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cheque_data (
        id TEXT PRIMARY KEY,
        client TEXT,
        amount TEXT,
        date TEXT,
        corrected_text TEXT
    )
    ''')

    conn.commit()
    conn.close()
def insert_corrected_data(db_path, cheque_id, client, amount, date, corrected_text):
    """Insert the corrected cheque data into the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    INSERT OR REPLACE INTO cheque_data (id, client, amount, date, corrected_text)
    VALUES (?, ?, ?, ?, ?)
    ''', (cheque_id, client, amount, date, corrected_text))

    conn.commit()
    conn.close()
def correct_and_save_data(image_path, db_path, seq2seq_model_path='/Users/mac/ChequeFixerATB/mlapp/seq2seq_model.keras', tokenizer=None, max_sequence_length=100):
    # Process and extract text from the cheque image
    ocr_results = preprocess_and_extract_ocr(image_path)
    extracted_text = ocr_results['words']
    amount_digit = ocr_results['amount']
    client_text = ocr_results['client']
    date_text = ocr_results['date']
    cheque_id = ocr_results['id']

    # Load the Seq2Seq model
    seq2seq_model = load_seq2seq_model(seq2seq_model_path)
    
    # Convert the amount to Hindi words
    converted_text = number_to_hindi_words(amount_digit)
    
    # Correct text using Seq2Seq model
    corrected_text_seq2seq = seq2seq_correction(seq2seq_model, extracted_text, tokenizer, max_sequence_length)
    is_correct_seq2seq = (corrected_text_seq2seq == extracted_text)
    
    # Correct text using LanguageTool
    corrected_text_lt = language_tool_correction(extracted_text)
    is_correct_lt = (corrected_text_lt == extracted_text)
    
    # Choose the method with better performance
    if is_correct_seq2seq:
        final_corrected_text = corrected_text_seq2seq
    elif is_correct_lt:
        final_corrected_text = corrected_text_lt
    else:
        # Default to converted text if neither method is correct
        final_corrected_text = converted_text
    
    # Insert the corrected data into the database
    insert_corrected_data(db_path, cheque_id, client_text, amount_digit, date_text, final_corrected_text)
    
    return final_corrected_text

def save_to_database(image_path):
    # Enregistrement des informations dans la base de données (Django ORM)
    from .models import Cheque  # Assurez-vous que le modèle est bien importé
    corrected_text, is_correct = correct_text_using_best_method(image_path)
    
    # Extract data
    ocr_results = preprocess_and_extract_ocr(image_path)
    cheque_id = ocr_results['id']
    client_name = ocr_results['client']
    amount_digits = ocr_results['amount']
    date = ocr_results['date'] 
    cheque = Cheque(
        cheque_id=cheque_id,
        client_name=client_name,
        amount_digits=amount_digits,
        date=date,
        corrected_text=corrected_text,
        is_correct=is_correct
    )
    cheque.save()
    return cheque
