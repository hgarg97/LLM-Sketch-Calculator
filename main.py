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

        self.canvas = tk.Canvas(root, bg='black', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.last_x, self.last_y = None, None
        self.current_action = []
        self.actions = []

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack(side=tk.LEFT)

        self.button_undo = tk.Button(root, text="Undo (Cmd/Ctrl Z)", command=self.undo)
        self.button_undo.pack(side=tk.LEFT)

        self.button_calculate = tk.Button(root, text="Calculate", command=self.calculate)
        self.button_calculate.pack(side=tk.LEFT)

        self.custom_font = tkFont.Font(family="Noteworthy", size=100)

        # Bind the Enter key to the calculate function
        self.root.bind("<Return>", self.handle_enter_key)

    def preprocess_image(self, pil_image):
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        _, binary_image = cv2.threshold(open_cv_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Remove noise
        kernel = np.ones((1, 1), np.uint8)
        denoised_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Resize for better OCR
        resized_image = cv2.resize(denoised_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return resized_image

    def handle_enter_key(self, event):
        self.calculate()

    def start_draw(self, event):
        self.current_action = []
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            line_id = self.canvas.create_line((self.last_x, self.last_y, x, y), fill='white', width=5)
            self.draw.line((self.last_x, self.last_y, x, y), fill='white', width=5)
            self.current_action.append((line_id, (self.last_x, self.last_y, x, y)))
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None
        if self.current_action:
            self.actions.append(self.current_action)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.actions = []

    def undo(self):
        if self.actions:
            last_action = self.actions.pop()
            for line_id, coords in last_action:
                self.canvas.delete(line_id)
            self.redraw_all()

    def redraw_all(self):
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        for action in self.actions:
            for _, coords in action:
                self.draw.line(coords, fill='white', width=5)
                self.canvas.create_line(coords, fill='white', width=5)

    def calculate(self):
        # Preprocess the image
        processed_image = self.preprocess_image(self.image)

        # Convert back to PIL for Tesseract
        pil_image = Image.fromarray(processed_image)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(pil_image, config="--psm 7").strip()
        print(f"Extracted text from canvas: {text}")  # Debugging: Print extracted text

        # Query the Gemini API
        answer = self.query_gemini(text)

        # Draw the answer on the canvas
        if answer:
            self.draw_answer(answer)
        else:
            print("No answer generated.")


    def query_gemini(self, text):
        try:
            # Modify the prompt to explicitly ask for just the answer
            prompt = f"Return only the answer for: {text}"
            print(f"Query sent to Gemini: {prompt}")  # Debugging: Print the refined prompt

            # Generate content using the documented method
            response = model.generate_content(prompt)
            print(f"Response from Gemini: {response}")  # Debugging: Print the response

            # Extract and return the generated text
            if response:
                return response.text.strip()  # Ensure the response is clean
            else:
                return "Error: No response generated."
        except Exception as e:
            print(f"Error querying Gemini API: {e}")
            return "Error"


    def draw_answer(self, answer):
        if not self.actions:
            return

        # Position the result next to the last equals sign
        last_action = self.actions[-1]
        last_coords = last_action[-1][-1]

        equals_x = last_coords[2]
        equals_y = last_coords[3]

        x_start = equals_x + 70
        y_start = equals_y - 20

        # Draw the answer on the canvas
        self.canvas.create_text(x_start, y_start, text=answer, font=self.custom_font, fill="#FF9500")
        font = ImageFont.load_default()
        self.draw.text((x_start, y_start - 50), answer, font=font, fill="#FF9500")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
