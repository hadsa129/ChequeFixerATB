# ChequeFixerATB: Autocorrection of Bank Cheques

## Project Overview
ChequeFixerATB is an application designed to detect and automatically correct errors in handwritten amounts on bank cheques. This project aims to automate cheque processing, improving both efficiency and accuracy in bank operations. It leverages Optical Character Recognition (OCR), machine learning, and Natural Language Processing (NLP) techniques to accurately extract and correct handwritten cheque information.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Segmentation and Extraction](#segmentation-and-extraction)
   - [KNN for Digit Extraction](#knn-for-digit-extraction)
   - [PaddleOCR for Cheque ID Extraction](#paddleocr-for-cheque-id-extraction)
   - [PyTesseract for Text Extraction](#pytesseract-for-text-extraction)
4. [Autocorrection](#autocorrection)
5. [Deployment](#deployment)
6. [Conclusion](#conclusion)
7. [Demo Video](#demo-video)
8. [Internship Report](#internship-report)

## Introduction
Cheque processing involves interpreting handwritten amounts, which can be challenging due to variations in handwriting quality and legibility. ChequeFixerATB addresses these challenges by developing an automated solution to extract amounts in both digits and words, detect discrepancies, and apply corrections using advanced NLP techniques.

## Data Preparation
The dataset includes 112 cheque images from four banks in India. Each image is processed using:
- **Resizing:** Standardizes image size for uniform processing.
- **Grayscale Conversion:** Reduces complexity by converting the image to grayscale.
- **Binarization:** Converts the image to a binary format to enhance text-background differentiation.
- **Cleaning:** Removes noise and irrelevant details from the image.
- **Object Detection:** Isolates areas of interest, such as cheque IDs and amounts in digits and words.

## Segmentation and Extraction
### KNN for Digit Extraction
A KNN (k-Nearest Neighbors) model is used to extract numerical amounts from the cheque images. After segmenting the digits and performing data augmentation, the KNN model classifies each digit to determine the numerical amount.

### PaddleOCR for Cheque ID Extraction
PaddleOCR is utilized for extracting cheque IDs, crucial for tracking and identifying cheques. Its high accuracy in text recognition makes it ideal for this task.

### PyTesseract for Text Extraction
PyTesseract, an open-source OCR tool, is used for extracting handwritten amounts written in words. After segmenting relevant cheque sections, PyTesseract recognizes and extracts these amounts.

## Autocorrection
A custom NLP model corrects errors in the extracted handwritten text. The `num2words` library converts numerical amounts into words for comparison. Additionally, a Seq2Seq model handles complex corrections involving entire sequences of words, providing a robust solution for error correction.

## Deployment
The application is deployed as a Django web application, featuring:
- **User Authentication:** Secure login and registration.
- **Cheque Processing Interface:** For uploading, scanning, and correcting cheques.
- **Error Correction:** Automatic detection and correction of discrepancies.
- **Data Visualization:** Integration of a PowerBI dashboard for analyzing cheque processing statistics.

## Conclusion
ChequeFixerATB successfully automates the detection and correction of errors in handwritten cheque amounts. By integrating OCR, machine learning, and NLP technologies, the application enhances accuracy and efficiency in cheque processing, reducing manual effort for banks.

## Demo Video
Watch the demo video to see ChequeFixerATB in action:

[![Demo Video]([https://example.com/path/to/your_image.jpg](https://github.com/hadsa129/ChequeFixerATB/blob/main/application.png))](https://youtu.be/SGIrFg3bNRE)



## Internship Report
For a detailed overview of the project, including methodology, implementation, and results, refer to the full internship report:

[Download the Internship Report]([link_to_your_report](https://github.com/hadsa129/ChequeFixerATB/blob/main/Rapport_atb.pdf))
