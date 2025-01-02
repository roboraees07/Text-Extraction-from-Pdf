# PDF Text Extractor

A Python-based project designed to extract text from PDF files, utilizing a combination of libraries for PDF parsing, OCR, and image processing. This project is implemented in a Jupyter Notebook for easy execution and experimentation.

## Features

- Extracts text from PDF files using:
  - **PyPDF2**: For text-based PDFs.
  - **pdf2image** and **pytesseract**: For OCR (Optical Character Recognition) on scanned PDFs.
  - **OpenCV**: For image preprocessing.
- Supports handling both text-based and image-based PDFs.
- Includes Bag-of-Words (BoW) and TF-IDF analysis for textual data.
- Visualizes word frequencies and importance using bar charts and pie charts.

## Requirements

### Libraries
This project uses the following Python libraries:
- `PyPDF2`
- `pdf2image`
- `pytesseract`
- `opencv-python-headless`
- `scikit-learn`
- `matplotlib`
- `numpy`

Install the required libraries using:
```bash
pip install PyPDF2 pdf2image pytesseract opencv-python-headless scikit-learn matplotlib numpy
```

### Additional Setup
- **Tesseract OCR**:
  Ensure Tesseract OCR is installed on your system. You can download it from [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract).
- **Google Drive (Optional)**:
  If running on Google Colab, the script includes functionality to mount your Google Drive for accessing files.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-text-extractor.git
   cd pdf-text-extractor
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Pdf_Text_Extractor.ipynb
   ```

3. Follow the steps in the notebook to:
   - Mount Google Drive (if using Colab).
   - Install necessary libraries.
   - Load and process PDF files.

4. Run the Python script for advanced text extraction and analysis:
   ```python
   import os
   import time
   from pdf2image import convert_from_path
   import pytesseract
   from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
   from concurrent.futures import ThreadPoolExecutor, as_completed
   import matplotlib.pyplot as plt
   import numpy as np

   # Timing the process
   start_time = time.time()

   # File paths
   file_path = "/content/drive/MyDrive/Interview/PDF Documents/Test.pdf"  # Input PDF file path

   # Step 1: Extract text from PDF pages
   def process_page_as_text(page_index, image):
       """
       Function to process a single page image and extract text using OCR.
       :param page_index: Index of the page
       :param image: PIL image object
       :return: Extracted text
       """
       extracted_text = pytesseract.image_to_string(image)  # Extract text using Tesseract OCR
       return extracted_text  # Return the text extracted

   # Convert all PDF pages to images using pdf2image
   print("Converting PDF to images...")
   images = convert_from_path(file_path)
   text = ""
   with ThreadPoolExecutor() as executor:
       futures = [executor.submit(process_page_as_text, i, img) for i, img in enumerate(images)]
       for future in as_completed(futures):
           extracted_text = future.result()
           text += extracted_text + "\n"

   # Step 2: Perform BoW analysis
   vectorizer_bow = CountVectorizer()
   text_corpus = [text]
   bow_matrix = vectorizer_bow.fit_transform(text_corpus)

   # Create a dictionary of word frequencies
   word_frequencies_bow = {word: int(freq) for word, freq in zip(vectorizer_bow.get_feature_names_out(), bow_matrix.toarray()[0])}

   # Step 3: Perform TF-IDF analysis
   vectorizer_tfidf = TfidfVectorizer()
   tfidf_matrix = vectorizer_tfidf.fit_transform(text_corpus)

   # Create a dictionary of TF-IDF scores
   tfidf_scores = {word: round(float(score), 4) for word, score in zip(vectorizer_tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])}

   # Step 4: Visualize BoW as a bar chart
   def plot_bow_bar_chart(data, top_n=10):
       """
       Function to plot BoW data as a bar chart with patterns.
       :param data: Dictionary of words and their frequencies.
       :param top_n: Number of top words to display.
       """
       # Sort and select top_n words
       sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
       words, frequencies = zip(*sorted_data)

       # Define colors and patterns
       colors = plt.cm.tab10.colors
       patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

       # Plot
       fig, ax = plt.subplots(figsize=(10, 6))
       bars = plt.bar(words, frequencies, color=colors[:len(words)])
       for bar, pattern in zip(bars, patterns[:len(words)]):
           bar.set_hatch(pattern)
       plt.title("Top Words by Frequency (BoW)", fontsize=16)
       plt.xlabel("Words", fontsize=12)
       plt.ylabel("Frequency", fontsize=12)
       plt.xticks(rotation=45, ha="right")
       plt.tight_layout()
       plt.show()

   # Step 5: Visualize TF-IDF as a circular chart
   def plot_tfidf_pie_chart(data, top_n=10):
       """
       Function to plot TF-IDF data as a pie chart.
       :param data: Dictionary of words and their TF-IDF scores.
       :param top_n: Number of top words to display.
       """
       # Sort and select top_n words
       sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
       words, scores = zip(*sorted_data)

       # Define colors
       colors = plt.cm.viridis(np.linspace(0, 1, len(words)))

       # Plot
       fig, ax = plt.subplots(figsize=(8, 8))
       plt.pie(scores, labels=words, colors=colors, autopct='%1.1f%%', startangle=140)
       plt.title("Top Words by TF-IDF Scores", fontsize=16)
       plt.tight_layout()
       plt.show()

   # Plot BoW and TF-IDF
   plot_bow_bar_chart(word_frequencies_bow, top_n=10)
   plot_tfidf_pie_chart(tfidf_scores, top_n=10)

   # Timing end
   end_time = time.time()
   print(f"Processing completed in {end_time - start_time:.2f} seconds")
   ```

## Example

Below is an example workflow demonstrated in the notebook:

1. **Load PDF File**:
   ```python
   from PyPDF2 import PdfReader
   reader = PdfReader('sample.pdf')
   for page in reader.pages:
       print(page.extract_text())
   ```

2. **Convert PDF to Images**:
   ```python
   from pdf2image import convert_from_path
   images = convert_from_path('scanned.pdf')
   for image in images:
       image.show()
   ```

3. **Extract Text Using OCR**:
   ```python
   from pytesseract import image_to_string
   text = image_to_string(image)
   print(text)
   ```

4. **Advanced Text Analysis with BoW and TF-IDF**:
   ```python
   # Refer to the script example above for BoW and TF-IDF analysis
   ```

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request for any features or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Author

Developed by Muhammad Raees Azam.
