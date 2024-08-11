# ChequeFixerATB:Autocorrection of Bank Cheques

## Project Overview
This project focuses on developing an application for the detection and automatic correction of errors in handwritten amounts on bank cheques. The main objective is to automate cheque processing, enhancing the efficiency and accuracy of bank operations. The project combines Optical Character Recognition (OCR), machine learning, and Natural Language Processing (NLP) techniques to extract and correct handwritten information from cheques.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Segmentation and Extraction](#segmentation-and-extraction)
   - [KNN for Digit Extraction](#knn-for-digit-extraction)
   - [PaddleOCR for Cheque ID Extraction](#paddleocr-for-cheque-id-extraction)
   - [PyTesseract for Text Extraction](#pytesseract-for-text-extraction)
4. [Autocorrection](#autocorrection)
   - [Error Detection](#error-detection)
   - [NLP-based Correction](#nlp-based-correction)
   - [Sequence-to-Sequence Model](#sequence-to-sequence-model)
5. [Deployment](#deployment)
6. [Conclusion](#conclusion)

## Introduction
The primary challenge in cheque processing is accurately interpreting handwritten amounts, which are often prone to errors or illegible. This project aims to address these challenges by developing an automated solution that extracts amounts written in both words and digits, detects discrepancies, and corrects them using advanced NLP techniques.

## Data Preparation
The dataset consists of 112 cheque images from four different banks in India, created using various pen ink colors and written by multiple volunteers. Each cheque image is preprocessed using techniques such as:
- **Resizing:** Standardizing the image size for uniform processing.
- **Grayscale Conversion:** Simplifying the image to reduce computational complexity.
- **Binarization:** Converting the image to a binary format to distinguish text from the background.
- **Cleaning:** Removing noise and irrelevant details from the image.
- **Object Detection:** Isolating specific areas of interest, such as the cheque ID, amounts in digits, and amounts in words.

## Segmentation and Extraction
### KNN for Digit Extraction
For extracting the numerical amounts, we implemented a KNN (k-Nearest Neighbors) model. The digits are segmented from the cheque image, followed by data augmentation to improve the model’s accuracy. The KNN model then classifies the individual digits, enabling accurate extraction of the numerical amount.

### PaddleOCR for Cheque ID Extraction
PaddleOCR is utilized to extract the cheque ID, which is essential for identifying and tracking the cheques. PaddleOCR, a lightweight and efficient OCR model, is employed due to its high accuracy in recognizing printed text.

### PyTesseract for Text Extraction
For extracting the text written in words, we used PyTesseract, an open-source OCR tool that effectively handles handwritten and printed text. After segmenting the relevant parts of the cheque, PyTesseract is used to recognize and extract the amount written in words.

## Autocorrection
### Error Detection
The extracted amounts are cross-verified using rule-based validation and contextual checks. For instance, the numerical amount is compared with the literal amount to detect discrepancies. Additionally, amounts are validated against plausible ranges to catch errors.

### NLP-based Correction
We implemented a custom NLP model to correct errors in the handwritten text. The `num2words` library is used to convert numerical amounts into words for comparison. Dempster-Shafer theory is employed to handle uncertainty and provide probabilistic corrections.

### Sequence-to-Sequence Model
For more complex corrections, a sequence-to-sequence (Seq2Seq) model is integrated into the system. This model is particularly useful for correcting errors that involve entire sequences of words, offering a sophisticated approach to error correction.

## Deployment
The final application is deployed as a Django web application, featuring:
- **User Authentication:** Secure login and registration.
- **Cheque Processing Interface:** Upload, scan, and correct cheques.
- **Error Correction:** Automatic detection and correction of discrepancies.
- **Data Visualization:** A PowerBI dashboard for analyzing cheque processing statistics.

## Conclusion
This project successfully automates the detection and correction of errors in handwritten amounts on bank cheques, providing an efficient solution for banks. The integration of OCR, machine learning, and NLP technologies ensures accurate and reliable cheque processing, reducing manual effort and improving overall efficiency.

