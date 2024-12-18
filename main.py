from io import BytesIO
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import os
from tkinter import font as tkFont
from dotenv import load_dotenv
import google.generativeai as genai

import cv2
import numpy as np

import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini AI Math Notes")

        self.canvas_width = 1200
        self.canvas_height = 800

        # Creating a frame for the buttons
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # Adding buttons to the button frame
        self.button_clear = tk.Button(button_frame, text="Clear", command=self.clear)
        self.button_clear.pack(side=tk.LEFT, padx=5, pady=5)

        self.button_undo = tk.Button(button_frame, text="Undo (Cmd/Ctrl Z)", command=self.undo)
        self.button_undo.pack(side=tk.LEFT, padx=5, pady=5)

        self.button_calculate = tk.Button(button_frame, text="Calculate", command=self.calculate)
        self.button_calculate.pack(side=tk.LEFT, padx=5, pady=5)

        # Creating the canvas
        self.canvas = tk.Canvas(root, bg='black', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.last_x, self.last_y = None, None
        self.current_action = []
        self.actions = []

        self.custom_font = tkFont.Font(family="Noteworthy", size=20)

        # Binding the Enter key to the calculate function
        self.root.bind("<Return>", self.handle_enter_key)

    def display_image(self, image, title="Image"):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def preprocess_image(self, pil_image):

        # Step 1: Convert PIL Image to OpenCV Format
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

        # Step 2: Apply Adaptive Thresholding for Better Contrast

        blurred_image = cv2.GaussianBlur(open_cv_image, (5, 5), 0)
        # self.display_image(blurred_image, title="Blurred Image")

        ret, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        # self.display_image(binary_image, title="Binary Image")

        binary_image_resized = cv2.resize(binary_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # self.display_image(binary_image_resized, title="Binary Image resized")

        # Step 4: Morphological Transformations for Noise Removal

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_image = cv2.dilate(binary_image_resized, kernel, iterations=1)
        # self.display_image(cleaned_image, title="Cleaned Image")

        # Step 5: Resize the Image for Better OCR Accuracy
        resized_image = cv2.resize(cleaned_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # self.display_image(resized_image, title="Resized Image")

        return resized_image


    def handle_enter_key(self, event):
        self.calculate()

    def calculate(self):

        # Preprocessing the image
        processed_image = self.preprocess_image(self.image)

       # Converting back to PIL for Tesseract
        pil_image = Image.fromarray(processed_image)
        
        # Defining the whitelist of characters to include only digits and operators
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789+-*/()=.'

        # Extracting text using Tesseract (only digits and operators)
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        print(f"Extracted text from canvas: {text}")

        # Querying the Gemini API
        answer = self.query_gemini(text)

        # Drawing the answer on the canvas
        if answer:
            self.draw_answer(answer)
        else:
            print("No answer generated.")

    def start_draw(self, event):
        self.current_action = []
        self.last_x, self.last_y = event.x, event.y

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.actions = []

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            line_id = self.canvas.create_line((self.last_x, self.last_y, x, y), fill='white', width=5)
            self.draw.line((self.last_x, self.last_y, x, y), fill='white', width=5)
            self.current_action.append((line_id, (self.last_x, self.last_y, x, y)))
        self.last_x, self.last_y = x, y

    def redraw_all(self):
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        for action in self.actions:
            for _, coords in action:
                self.draw.line(coords, fill='white', width=5)
                self.canvas.create_line(coords, fill='white', width=5)

    def reset(self, event):
        self.last_x, self.last_y = None, None
        if self.current_action:
            self.actions.append(self.current_action)

    def undo(self):
        if self.actions:
            last_action = self.actions.pop()
            for line_id, coords in last_action:
                self.canvas.delete(line_id)
            self.redraw_all()

    def draw_answer(self, answer):
        if not self.actions:
            return

        # Positioning the result next to the last equals sign
        last_action = self.actions[-1]
        last_coords = last_action[-1][-1]

        equals_x = last_coords[2]
        equals_y = last_coords[3]

        x_start = equals_x + 70
        y_start = equals_y - 20

        # Determining an appropriate font size based on canvas size and content
        scale_factor = 2
        font_size = int(self.canvas_width // scale_factor)
        font = ImageFont.truetype("arial.ttf", font_size)

        # Drawing the answer dynamically on both the canvas and PIL image
        self.canvas.create_text(x_start, y_start, text=answer, font=self.custom_font, fill="#FF9500")
        self.draw.text((x_start, y_start), answer, font=font, fill="#FF9500")


    def query_gemini(self, text):
        try:
            # Modify the prompt to explicitly ask for just the answer
            prompt = f'''You are mathematical calculator, the text provided to you is drawn by hand and extracted using OCR techniques, 
            we are drawing digits and operators in a human way ending with '=', return only the solved answer for {text} \n 
            If the text provided doesn't make sense return 'NaN' '''
            print(f"Query sent to Gemini: {prompt}")

            # Generate content using the documented method
            response = model.generate_content(prompt)
            print(f"Response from Gemini: {response}")

            # Extract and return the generated text
            if response:
                return response.text.strip()
            else:
                return "Error: No response generated."
        except Exception as e:
            print(f"Error querying Gemini API: {e}")
            return "Error"

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
