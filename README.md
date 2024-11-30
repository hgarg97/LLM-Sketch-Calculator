# LLM Sketch Calculator

Gemini AI Math Notes is an interactive drawing application that allows users to draw mathematical equations on a canvas and get the result displayed in real-time. This tool leverages Optical Character Recognition (OCR) to interpret handwritten equations and a multimodal Generative AI model (Gemini API) to calculate the result.

---

## Features

- **Canvas Drawing:** Intuitive drawing interface using Tkinter to input equations.
- **Real-time Calculation:** Processes the drawn equation and provides the calculated result next to the equation.
- **Undo and Clear Options:** Allows undoing mistakes or clearing the canvas entirely.
- **Advanced Preprocessing:** Ensures better OCR accuracy by preprocessing drawn images.
- **AI Integration:** Uses the Gemini API for mathematical reasoning and calculation.

---

## Setup

### Prerequisites

- Python 3.8 or later
- Virtual environment (optional but recommended)
- API Key for **Gemini API** (Google Generative AI)

### Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following contents:

```txt
opencv-python
pillow
pytesseract
python-dotenv
google-generativeai
matplotlib
```

### Setting Up Environment Variables

1. Create a `.env` file in the root directory.
2. Add your Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Tesseract OCR

Ensure that Tesseract is installed on your system:

- For Ubuntu:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- For macOS:
  ```bash
  brew install tesseract
  ```
- For Windows, download the installer from [Tesseract's GitHub page](https://github.com/tesseract-ocr/tesseract).

---

## Running the Application

1. Clone this repository:
   ```bash
   git clone https://github.com/hgarg97/LLM-Sketch-Calculator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd LLM-Sketch-Calculator
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## Usage

1. Use the **Canvas** to draw mathematical equations.
2. Press `Enter` or click the **Calculate** button to process and solve the equation.
3. Use **Undo** or **Clear** for corrections.
4. The result will be displayed next to the `=` sign.

---

## Project Structure

```plaintext
Gemini-AI-Math-Notes/
├── main.py                # Main application code
├── .env                   # Environment variables
├── requirements.txt       # Required Python libraries
└── README.md              # Project documentation
```

---

## Example

1. Draw an equation such as `2 + 2 =` on the canvas.
2. Press **Enter** or click **Calculate**.
3. The application displays the result `4` next to the equals sign.

---

## Known Issues

- OCR may misinterpret complex or messy handwriting.
- Ensure `Tesseract` is correctly installed and configured.
- API usage is dependent on the allocated limits of your Gemini API key.

---

## Future Improvements

- Support for additional mathematical functions and symbols.
- Integration of a more robust handwriting recognition system.
- Multi-language support for equations and instructions.

---

## Acknowledgments

This project is inspired by [Apple's Math Notes demo](https://www.youtube.com/live/RXeOiIDNNek?si=zsfLkfVtCoCqk1ie&t=2806) from WWDC 2024.

Feel free to contribute and make this application even better! 🚀
