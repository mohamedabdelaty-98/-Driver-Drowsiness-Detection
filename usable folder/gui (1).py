# import PIL
# from PIL import Image,ImageTk
# import pytesseract
# import cv2
# from tkinter import *
#
#
# Gui_Win=tk.Tk()
# Gui_Win.title('Gui_Task_1')
# Gui_Win.geometry('800x900')
# Gui_Win.configure(bg='blue')
#
#
# width, height = 800, 600
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#
# root = Tk()
# root.bind('<Escape>', lambda e: root.quit())
# lmain = Label(root)
# lmain.pack()
#
# def show_frame():
#     _, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#     img = PIL.Image.fromarray(cv2image)
#     imgtk = ImageTk.PhotoImage(image=img)
#     lmain.imgtk = imgtk
#     lmain.configure(image=imgtk)
#     lmain.after(10, show_frame)
#
# show_frame()
# root.mainloop()


import sys
import cv2
import threading
import tkinter as tk
import tkinter.ttk as ttk
from queue import Queue
from PIL import Image
from PIL import ImageTk
from pygame import mixer
import usethemodel as pjbody



class App(tk.Frame):
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)
        self.is_running = False
        self.thread = None
        self.queue = Queue()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (800, 600), "white"))
        parent.wm_withdraw()
        parent.wm_title(title)
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.wm_protocol("WM_DELETE_WINDOW", self.on_destroy)
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)
        parent.wm_deiconify()

    def create_ui(self):
        self.button_frame = ttk.Frame(self)

        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(self.button_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.RIGHT)

        self.view = ttk.Label(self, image=self.photo)
        self.view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def on_destroy(self):
        self.stop()
        self.after(20)
        if self.thread is not None:
            self.thread.join(0.2)
        self.winfo_toplevel().destroy()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False

    def videoLoop(self, mirror=False):
        No=1
        cap = cv2.VideoCapture(No)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        drowsycounter = 0  # initialization of the drowsyness counting
        mixer.init()
        sound = mixer.Sound('alarm.mp3')  # this is the alarm sound
        while self.is_running:
            ret, to_draw = cap.read()

            # make it flipped like mirror
            to_draw = cv2.flip(to_draw, 1)

            # conition = 0 for high light
            # conition = 1 for good light
            # conition = 2 for bad light
            to_draw = pjbody.preprocess_the_frame(to_draw, condition=0)

            # get the latest frame with all the drawing on it after applying the models
            to_draw, drowsycounter = pjbody.project_func(to_draw, drowsycounter)

            # resize to a fixed size
            to_draw = cv2.resize(to_draw, (600, 600))

            # display////////////
            # cv2.imshow('', frame)

            # if his both eyes was closed for 7 frames so alarm play
            if drowsycounter >= 5:
                sound.play()

            if drowsycounter < 5:
                sound.stop()
            # self.label_frame = ttk.Frame(self)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break



            if mirror is True:
                to_draw = to_draw[:,::-1]
            image = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
            self.queue.put(image)
            self.event_generate('<<MessageGenerated>>')

    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)


def main(args):
    root = tk.Tk()
    app = App(root, "OpenCV Image Viewer")
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))