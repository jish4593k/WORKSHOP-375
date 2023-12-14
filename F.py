import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import tkinter as tk
from PIL import Image, ImageTk

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

sor
transform = transforms.Compose([transforms.ToTensor()])

class FaceApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Face Detection App")

        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)

        self.model = model

        self.canvas = tk.Canvas(root, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.pack()

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        self.is_detecting = False
        self.detect()

    def start_detection(self):
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.start_button["text"] = "Stop Detection"
        else:
            self.start_button["text"] = "Start Detection"

    def detect(self):
        if self.is_detecting:
            ret, frame = self.cap.read()

            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor_image = transform(image).unsqueeze(0)
                with torch.no_grad():
                    prediction = self.model(tensor_image)

                self.photo = ImageTk.PhotoImage(image=image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(10, self.detect)

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

if name == "main":
    root = tk.Tk()
    app = FaceApp(root, model)
    root.mainloop()
