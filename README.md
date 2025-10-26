# Local File Organizer: Smart File Management Using Local Machine Learning

Bring order to your digital chaos with this intelligent file organizer that uses lightweight machine learning algorithms to automatically categorize and organize your files - all while running entirely on your local machine with no external dependencies.

## How It Works

The Local File Organizer processes your files using:

- **Text Analysis**: Uses TF-IDF vectorization and NLTK for understanding text content
- **Image Analysis**: Employs OpenCV and scikit-learn for visual content understanding
- **Smart Categorization**: Applies machine learning techniques to group similar files
- **Intelligent Renaming**: Generates descriptive filenames based on file content

Before:
```
/messy_folder/
├── IMG_20230515.jpg      # Picture of an apple
├── IMG_20230516.jpg      # Picture of office desk
├── notes_20230517.txt    # Meeting minutes
├── data_analysis.xlsx    # Sales data
└── document1.pdf         # Company policy
```

After:
```
/organized_folder/
├── Food_and_Produce/
│   └── Red_Apple_Photo_May15.jpg
├── Office/
│   ├── Workspace_Setup_May16.jpg
│   └── Meeting_Minutes_May17.txt
├── Financial/
│   └── Sales_Analysis_2023.xlsx
└── Documents/
    └── Company_Policy_Guidelines.pdf
```

## Features

- **100% Local Processing**: All analysis happens on your machine
- **No Internet Required**: Works completely offline
- **Privacy First**: No data leaves your computer
- **Resource Efficient**: Uses lightweight ML algorithms
- **Smart Object Detection**: Identifies common objects in images
- **Text Understanding**: Extracts key topics from documents
- **OCR Support**: Reads text from images and PDFs

## Supported File Types 📁

- **Images:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`
- **Documents:** `.txt`, `.docx`, `.md`, `.pdf`
- **Data Files:** `.xlsx`, `.csv`
- **Presentations:** `.ppt`, `.pptx`

## Installation 🛠️

### Prerequisites

- Python 3.11
- Conda package manager
- Windows, macOS, or Linux

### Setup Steps

1. Create and activate conda environment:
```powershell
conda create --name file-sorter python=3.11 -y
conda activate file-sorter
```

2. Install core dependencies:
```powershell
conda install -y -c conda-forge numpy=1.24.3 pillow=9.5.0 pandas scikit-learn scipy opencv nltk tesseract
```

3. Install remaining packages:
```powershell
python -m pip install -r requirements.txt
```

### Optional Components

- **Tesseract OCR**: Required for text extraction from images
  - Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

## Environment Variables

Before running the application, you need to create a `.env` file in the root directory of the project. This file is used to store your Gemini API key.

1.  **Create a `.env` file** in the root of the project.
2.  **Add the following line** to the `.env` file:

    ```
    GEMINI_API_KEY="your_api_key_here"
    ```

    Replace `"your_api_key_here"` with your actual Gemini API key.

## Usage

1. Activate the conda environment:
```powershell
conda activate file-sorter
```

2. Run the organizer:
```powershell
python main.py
```

3. Follow the prompts to select input and output directories

## How It Works Under the Hood

### Text Processing
- NLTK for tokenization and text preprocessing
- TF-IDF vectorization for content analysis
- Scikit-learn for text classification
- Extractive summarization for file descriptions

### Image Processing
- OpenCV for image analysis and object detection
- Color space analysis (HSV) for object recognition
- Shape detection for specific item recognition
- OCR integration for text in images

### File Organization
- Content-based clustering for folder creation
- Intelligent filename generation
- Metadata extraction and analysis
- Multi-threaded processing for performance

## Configuration

The organizer can be customized through:
- File type extensions in `file_utils.py`
- Processing rules in `data_processing_common.py`
- Output formatting in `output_filter.py`

## License

This project is dual-licensed under the MIT License and Apache 2.0 License.
See the [LICENSE](LICENSE) file for details.