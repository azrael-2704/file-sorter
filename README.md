# Local File Organizer: AI File Management Run Entirely on Your Device, Privacy Assured

Tired of digital clutter? Overwhelmed by disorganized files scattered across your computer? Let AI do the heavy lifting! The Local File Organizer is your personal organizing assistant, using cutting-edge AI to bring order to your file chaos - all while respecting your privacy.

## How It Works 💡

Before:

```
/home/user/messy_documents/
├── IMG_20230515_140322.jpg
├── IMG_20230516_083045.jpg
├── IMG_20230517_192130.jpg
├── budget_2023.xlsx
├── meeting_notes_05152023.txt
├── project_proposal_draft.docx
├── random_thoughts.txt
├── recipe_chocolate_cake.pdf
├── scan0001.pdf
├── vacation_itinerary.docx
└── work_presentation.pptx

0 directories, 11 files
```

After:

```
/home/user/organized_documents/
├── Financial
│   └── 2023_Budget_Spreadsheet.xlsx
├── Food_and_Recipes
│   └── Chocolate_Cake_Recipe.pdf
├── Meetings_and_Notes
│   └── Team_Meeting_Notes_May_15_2023.txt
├── Personal
│   └── Random_Thoughts_and_Ideas.txt
├── Photos
│   ├── Cityscape_Sunset_May_17_2023.jpg
│   ├── Morning_Coffee_Shop_May_16_2023.jpg
│   └── Office_Team_Lunch_May_15_2023.jpg
├── Travel
│   └── Summer_Vacation_Itinerary_2023.docx
└── Work
    ├── Project_X_Proposal_Draft.docx
    ├── Quarterly_Sales_Report.pdf
    └── Marketing_Strategy_Presentation.pptx

7 directories, 11 files
```

## Updates 🚀

This repository was refactored to remove the Nexa SDK dependency and provide a fully local, lightweight inference pipeline using commonly-available Python libraries (scikit-learn, OpenCV, NLTK). The goal is to avoid heavy native build dependencies and allow offline execution on typical developer machines.

Key changes:
- Replaced large transformer/VLM model dependencies with a heuristic + TF-IDF extractive summarizer for text and OpenCV-based image analysis.
- Removed direct usage of the Nexa SDK and transformer models to avoid PyTorch/safetensors/Rust build issues.
- Simplified `local_ai.py` to use scikit-learn, NLTK, and OpenCV so installation works well inside conda environments.

Why this change: it reduces installation friction (no PyTorch/CUDA or Rust toolchain needed) and keeps all processing local and private.


## Roadmap 📅

- [ ] Copilot Mode: chat with AI to tell AI how you want to sort the file (ie. read and rename all the PDFs)
- [ ] Change models with CLI 
- [ ] ebook format support
- [ ] audio file support
- [ ] video file support
- [ ] Implement best practices like Johnny Decimal
- [ ] Check file duplication
- [ ] Dockerfile for easier installation
- [ ] People from [Nexa](https://github.com/NexaAI/nexa-sdk) is helping me to make executables for macOS, Linux and Windows

## What It Does 🔍

This intelligent file organizer harnesses the power of advanced AI models, including language models (LMs) and vision-language models (VLMs), to automate the process of organizing files by:


* Scanning a specified input directory for files.
* Content Understanding: 
  - **Textual Analysis**: Uses the [Llama3.2 3B](https://nexaai.com/meta/Llama3.2-3B-Instruct/gguf-q3_K_M/file) to analyze and summarize text-based content, generating relevant descriptions and filenames.
  - **Visual Content Analysis**: Uses the [LLaVA-v1.6](https://nexaai.com/liuhaotian/llava-v1.6-vicuna-7b/gguf-q4_0/file) , based on Vicuna-7B, to interpret visual files such as images, providing context-aware categorization and descriptions.

* Understanding the content of your files (text, images, and more) to generate relevant descriptions, folder names, and filenames.
* Organizing the files into a new directory structure based on the generated metadata.

The best part? All AI processing happens 100% on your local device using the [Nexa SDK](https://github.com/NexaAI/nexa-sdk). No internet connection required, no data leaves your computer, and no AI API is needed - keeping your files completely private and secure.


## Supported File Types 📁

- **Images:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- **Text Files:** `.txt`, `.docx`, `.md`
- **Spreadsheets:** `.xlsx`, `.csv`
- **Presentations:** `.ppt`, `.pptx`
- **PDFs:** `.pdf`

## Prerequisites 💻

- **Operating System:** Compatible with Windows, macOS, and Linux.
- **Python Version:** Python 3.12
- **Conda:** Anaconda or Miniconda installed.
- **Git:** For cloning the repository (or you can download the code as a ZIP file).

## Installation 🛠

> For SDK installation and model-related issues, please post on [here](https://github.com/NexaAI/nexa-sdk/issues).

### 1. Install Python

Before installing the Local File Organizer, make sure you have Python installed on your system. We recommend using Python 3.12 or later.

You can download Python from [the official website]((https://www.python.org/downloads/)).

Follow the installation instructions for your operating system.

### 2. Clone the Repository

Clone this repository to your local machine using Git:

```zsh
git clone https://github.com/QiuYannnn/Local-File-Organizer.git
```

Or download the repository as a ZIP file and extract it to your desired location.

### 3. Set Up the Python Environment

Create a new Conda environment named `local_file_organizer` with Python 3.12:

```zsh
conda create --name local_file_organizer python=3.12
```

Activate the environment:

```zsh
conda activate local_file_organizer
```

### 4. Install dependencies (conda + pip recommended)

We strongly recommend creating and using a Conda environment. Conda provides pre-built binary packages (numpy, OpenCV, SciPy, scikit-learn) which avoids compiling from source and prevents common installation errors on Windows.

1. Create and activate the conda environment:

```powershell
conda create --name file-sorter python=3.11 -y
conda activate file-sorter
```

2. Install heavy binary packages from conda-forge (these include numpy, OpenCV, SciPy, scikit-learn, Pillow, and Tesseract OCR engine):

```powershell
conda install -y -c conda-forge numpy=1.24.3 pillow=9.5.0 pandas scikit-learn scipy opencv nltk tesseract
```

3. Install the remaining Python packages via pip (inside the activated env):

```powershell
python -m pip install -r requirements.txt
```

Notes:
- We removed the Nexa SDK and heavy transformer runtimes to keep this project lightweight and easy to install locally.
- `tesseract` (OCR engine) is installed via conda; the Python wrapper `pytesseract` is listed in `requirements.txt` and will be installed with pip.

### 5. Running the Script 🎉

After completing the installation steps above, run:

```powershell
python main.py
```

This will analyze the files in the configured input directory and print/move files according to the rules in the project.

## Notes

- **SDK Models:**
  - The script uses `NexaVLMInference` and `NexaTextInference` models [usage](https://docs.nexaai.com/sdk/python-interface/gguf).
  - Ensure you have access to these models and they are correctly set up.
  - You may need to download model files or configure paths.


- **Dependencies:**
  - **pytesseract:** Requires Tesseract OCR installed on your system.
    - **macOS:** `brew install tesseract`
    - **Ubuntu/Linux:** `sudo apt-get install tesseract-ocr`
    - **Windows:** Download from [Tesseract OCR Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **PyMuPDF (fitz):** Used for reading PDFs.

- **Processing Time:**
  - Processing may take time depending on the number and size of files.
  - The script uses multiprocessing to improve performance.

- **Customizing Prompts:**
  - You can adjust prompts in `data_processing.py` to change how metadata is generated.

## License

This project is dual-licensed under the MIT License and Apache 2.0 License. You may choose which license you prefer to use for this project.

- See the [MIT License](LICENSE-MIT) for more details.