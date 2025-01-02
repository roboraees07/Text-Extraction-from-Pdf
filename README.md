# PDF Text Extractor

A Python-based project designed to extract text from PDF files, utilizing a combination of libraries for PDF parsing, OCR, and image processing. This project is implemented in a Jupyter Notebook for easy execution and experimentation.

## Features

- Extracts text from PDF files using:
  - **PyPDF2**: For text-based PDFs.
  - **pdf2image** and **pytesseract**: For OCR (Optical Character Recognition) on scanned PDFs.
  - **OpenCV**: For image preprocessing.
- Supports handling both text-based and image-based PDFs.

## Requirements

### Libraries
This project uses the following Python libraries:
- `PyPDF2`
- `pdf2image`
- `pytesseract`
- `opencv-python-headless`

Install the required libraries using:
```bash
pip install PyPDF2 pdf2image pytesseract opencv-python-headless
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

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request for any features or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Author

Developed by Muhammad Raees Azam.
