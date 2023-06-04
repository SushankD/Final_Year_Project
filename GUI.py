import tkinter as tk
from PIL import Image, ImageTk
import sys

class Application:
    def __init__(self, master):
        self.master = master
        master.title("Obstacle Detection")
        # master.attributes('-fullscreen', True)
        master.geometry("1300x700")

        # Set background image for the window
        background_image = Image.open("bg.jpg")  # Replace "background-image.jpg" with the path to your image
        resized_image = background_image.resize((1300, 700), Image.ANTIALIAS)  # Resize the image to match the window size
        self.background_image = ImageTk.PhotoImage(resized_image)
        self.background_label = tk.Label(master, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        heading_font_style = ("Helvetica", 36)
        button_font_style = ("Helvetica", 24)

        self.heading_label = tk.Label(master, text="Obstacle Detection System", font=heading_font_style, bg="#01011E", fg="white")
        self.heading_label.grid(row=0, column=0, columnspan=3, padx=375, pady=5, sticky=tk.W)

        start_image = Image.open("start.png")  # Replace "start-button.png" with the path to your image
        resized_start_image = start_image.resize((150, 75), Image.ANTIALIAS)  # Set the desired width and height
        self.start_button_image = ImageTk.PhotoImage(resized_start_image)
        self.start_button = tk.Button(master, image=self.start_button_image, command=self.start, bg="#01011E", fg="white", bd=0)
        self.start_button.image = self.start_button_image

        stop_image = Image.open("stop.png")  # Replace "start-button.png" with the path to your image
        resized_stop_image = stop_image.resize((150, 75), Image.ANTIALIAS)  # Set the desired width and height
        self.stop_button_image = ImageTk.PhotoImage(resized_stop_image)
        self.stop_button = tk.Button(master, image=self.stop_button_image, command=self.stop, bg="#01011E", fg="white", bd=0)
        self.stop_button.image = self.stop_button_image

        #self.stop_button = tk.Button(master, text="Stop", command=self.stop, font=button_font_style, bg="#01011E", fg="white")

        self.start_button.grid(row=1, column=0, padx=10, pady=250)
        self.stop_button.grid(row=1, column=2, padx=970, pady=250)

        self.started = False

    def start(self):
        print("Start button pressed")
        # obsdec = Laser()
        # obsdec.run()

    def stop(self):
        print("Stop button pressed")
        sys.exit(0)


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
