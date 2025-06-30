import tkinter as tk
import subprocess
import os
from PIL import Image, ImageTk
import threading
import queue

#script that creates a GUI application using Tkinter.
#it makes possible to run other scripts from the GUI 

class DocVQAApp:
    def __init__(self, master):
        self.master = master
        master.title("DocVQA")
        master.state('zoomed')
        
        self._load_background()
        self._load_bottom_right_image()
        self._create_widgets()

        self.result_label = tk.Label(master, text="", font=("Arial", 10), bg="#edf5fa")
        self.result_label.pack(pady=20)

    def _run_script(self, script_name, message):
        """Runs a Python script using subprocess and displays a message."""
        try:
            subprocess.run(["python", script_name], check=True)
            self.result_label.config(text=message)
            self.master.after(2000, lambda: self.result_label.config(text=""))  #clear msg after 2 seconds
        except subprocess.CalledProcessError as e:
            self.result_label.config(text=f"Error running {script_name}: {e}")
            self.master.after(7000, lambda: self.result_label.config(text=""))  #clear msg after 7 seconds
        except FileNotFoundError:
            self.result_label.config(text=f"Error: {script_name} not found.")
            self.master.after(5000, lambda: self.result_label.config(text=""))  #clear msg after 5 seconds

    def _load_background(self):
        """Loads and displays the background image."""
        try:
            image_path = os.path.join(os.path.dirname(__file__), "GUI_images/background_image.png")
            img_background = Image.open(image_path)
            self.background_tk = ImageTk.PhotoImage(img_background)
            background_label = tk.Label(self.master, image=self.background_tk)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)  #put image in background
            background_label.lower()  #send label to the back
        except FileNotFoundError:
            print(f"Error: Background image not found at {image_path}.")
            self.master.config(bg="lightgray")  #background color if image not found

    def _load_bottom_right_image(self):
        """Loads and displays the bottom right image."""
        try:
            image_path_bottom_right = os.path.join(os.path.dirname(__file__), "GUI_images/uab_green.png")
            img_bottom_right = Image.open(image_path_bottom_right)
            img_bottom_right_resized = img_bottom_right.resize((66, 22), Image.LANCZOS)
            self.bottom_right_tk = ImageTk.PhotoImage(img_bottom_right_resized)
            bottom_right_label = tk.Label(self.master, image=self.bottom_right_tk)
            bottom_right_label.place(relx=0.97, rely=0.93, anchor=tk.SE)
        except FileNotFoundError:
            print(f"Error: Bottom right image not found at {image_path_bottom_right}")

    def _open_analysis_window(self):
        """Opens a new window and displays output from the analysis script."""
        analysis_window = tk.Toplevel(self.master)
        analysis_window.title("Data Analysis Output")
        analysis_window.geometry("800x785")
        #open window  
        analysis_window.geometry("+{}+{}".format(self.master.winfo_screenwidth() - 800, 0))

        text_widget = tk.Text(analysis_window, wrap=tk.WORD, font=("Courier", 10))
        text_widget.pack(expand=True, fill="both", padx=10, pady=(10, 0))

        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        #close button
        close_button = tk.Button(analysis_window, text="Cerrar", font=("Arial", 12), bg="#ffdddd", command=lambda: close_analysis())
        close_button.pack(pady=10)

        #store process
        self.analysis_process = None

        def run_analysis_script():
            analysis_path = os.path.join(os.path.dirname(__file__), "main_data_analysis.py")
            self.analysis_process = subprocess.Popen(
                ["python", "-u", analysis_path, "--mode", "single"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            def enqueue_output():
                for line in self.analysis_process.stdout:
                    if line:
                        analysis_window.after(0, lambda l=line: safe_insert(l))
                self.analysis_process.stdout.close()
                self.analysis_process.wait()
                if analysis_window.winfo_exists():
                    analysis_window.after(500, analysis_window.destroy)

            def safe_insert(line):
                if text_widget.winfo_exists():
                    text_widget.insert(tk.END, line)
                    text_widget.see(tk.END)

            threading.Thread(target=enqueue_output, daemon=True).start()

        def close_analysis():
            if self.analysis_process and self.analysis_process.poll() is None:
                self.analysis_process.terminate()
            analysis_window.destroy()

        threading.Thread(target=run_analysis_script, daemon=True).start()



    def _create_widgets(self):
        """Creates and places the widgets for the application."""
        #title Label
        title = tk.Label(self.master, text="Human Eye-Tracking for Driving Explainability in DocVQA", font=("Arial", 24, "bold"), bg="#edf5fa")
        title.pack(pady=40)

        #consent agreement page button
        consent_agreement_path = os.path.join(os.path.dirname(__file__), "consent_GUI.py")
        button_consent = tk.Button(self.master, text="Consent agreement page", command=lambda: self._run_script(consent_agreement_path, "Running consent_GUI.py"), bg="#b5d6ed", font=("Arial", 16))
        button_consent.config(width=30, height=2)
        button_consent.pack(pady=25)

        #calibration button
        calibration_path = os.path.join(os.path.dirname(__file__), "open_tobii.py")
        button_calibrate = tk.Button(self.master, text="Calibrate Tobii Pro Spark", command=lambda: self._run_script(calibration_path, "Opening Tobii Pro Spark Manager"), bg="#c8e0f1", font=("Arial", 16))
        button_calibrate.config(width=30, height=2)
        button_calibrate.pack(pady=25)

        #trial button
        trial_path = os.path.join(os.path.dirname(__file__), "main_GUI.py")
        button_trial = tk.Button(self.master, text="Trial", command=lambda: self._run_script(trial_path, "Running main_GUI.py"), bg="#daebf6", font=("Arial", 16))
        button_trial.config(width=30, height=2)
        button_trial.pack(pady=25)

        #aalysis button
        button_analysis = tk.Button(self.master, text="Data Analysis", command=self._open_analysis_window, bg="#e8f3fa", font=("Arial", 16))
        button_analysis.config(width=30, height=2)
        button_analysis.pack(pady=25)

        #exit button
        button_exit = tk.Button(self.master, text="Exit", command=self.master.quit, bg="#f0f9ff", font=("Arial", 16))
        button_exit.config(width=10, height=1)
        button_exit.pack(pady=20)

        #bottom left corner text
        self.thesis = tk.Label(self.master, text="Bachelor's Thesis", font=("Arial", 11), bg="white")
        self.thesis.place(x=40, y=self.master.winfo_screenheight() - 100)

        self.degree = tk.Label(self.master, text="Artificial Intelligence Degree", font=("Arial", 11), bg="white")
        self.degree.place(x=40, y=self.master.winfo_screenheight() - 76)

        self.master.bind("<Configure>", self._update_name_position)

    def _update_name_position(self, event):
        """Updates the position of the bottom left text on window resize."""
        self.thesis.place(x=40, y=self.master.winfo_height() - 100)
        self.degree.place(x=40, y=self.master.winfo_height() - 76)

def main():
    root = tk.Tk()
    app = DocVQAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()