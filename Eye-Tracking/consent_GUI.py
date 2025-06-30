import tkinter as tk
from tkinter import ttk, messagebox
import json
import uuid
import os
import time
from datetime import datetime


#directory for participant data
DATA_DIR = "consent_participant_data"
os.makedirs(DATA_DIR, exist_ok=True)

#.txt file with participant IDs, if it does not exist create it 
PARTICIPANT_IDS_FILE = "participant_IDs.txt"
if not os.path.exists(PARTICIPANT_IDS_FILE):
    with open(PARTICIPANT_IDS_FILE, "w") as f:
        f.write("")


#lang dict
translations = {
    "English": {
        "title": "Eye-Tracking DocVQA Trial Consent",
        "consent_title": "CONSENT INFORMATION",
        "purpose_title": "Purpose of the Study:",
        "purpose_text": "This study aims to understand how eye movements relate to reading comprehension "
                        "when viewing images with embedded text. We hope to learn how people focus their "
                        "attention on different parts of an image and how this relates to understanding the "
                        "information presented. This research could contribute to improving reading "
                        "comprehension strategies.",
        "confidentiality_title": "Confidentiality and Data Security:",
        "confidentiality_text": "- Your eye-tracking data will be stored securely.\n"
                        "- It will be anonymized and you will be given a unique participant ID.\n"
                        "- Only authorized researchers will have access to the data.\n"
                        "- Data may be shared with collaborators or published in scientific journals, "
                        "but only in anonymized form.",
        "withdrawal_title": "Right to Withdraw:",
        "withdrawal_text": "You have the right to withdraw from the study at any time without penalty. "
                        "If you wish to withdraw, please inform the researcher. If you withdraw, your data will be deleted.",
        "acceptance_title": "Acceptance of Consent:",
        "acceptance_text": "By clicking 'I agree', you confirm that you have read and understood the information provided and agree to participate.",
        "next": "Next",
        "name": "Name:",
        "surname": "Surname:",
        "email": "Email:",
        "prefix": "Country prefix:",
        "phone": "Phone:",
        "age": "Age:",
        "age_options": ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        "gender": "Sex/Gender:",
        "gender_options": ["Male", "Female", "Prefer not to specify", "Other"],
        "english_level": "English Level (1-5):",
        "english_level_options": ["1", "2", "3", "4", "5"],
        "eye_condition": "Do you wear glasses/contact lens or have eye problems?",
        "eye_condition_options": ["Yes, glasses/contact lens", "Yes, eye problems", "Yes, both", "No"],
        "eye_problems": "Eye problems (if applicable):",
        "eye_problems_options": ["Myopia", "Hypermetropia", "Astigmatism", "Presbyopia", "Other", "None"],
        "lens":"Lens type (if applicable):",
        "lens_options": ["Single Vision", "Bifocals", "Multifocals", "Progressives", "Occupational", "Other"],
        "consent_agreement": "I have read and agree to participate.",
        "agree": "I agree",
        "submit": "Submit",
        "consent_required": "You must agree to participate before proceeding.",
        "submission_complete": "Your data has been recorded. Thank you!"
    },
    "Catalan": {
        "title": "Consentiment per a la prova DocVQA de seguiment ocular",
        "consent_title": "INFORMACIÓ SOBRE EL CONSENTIMENT",
        "purpose_title": "Propòsit de l'estudi:",
        "purpose_text": "Aquest estudi té com a objectiu comprendre com els moviments oculars es relacionen amb la "
                        "comprensió lectora en veure imatges amb text. "
                        "Esperem aprendre com les persones enfoquen la seva atenció en diferents parts d'una imatge "
                        "i com això influeix en la comprensió de la informació presentada. Aquesta investigació podria "
                        "contribuir a millorar les estratègies de comprensió lectora.",
        "confidentiality_title": "Confidencialitat i Privacitat de les dades:",
        "confidentiality_text": "- Les dades de seguiment ocular s'emmagatzemaran de manera segura.\n"
                        "- Les dades seran anononimitzades i se t'assignarà un ID únic de participant.\n"
                        "- Només investigadors autoritzats hi tindran accés.\n"
                        "- Les dades podrien ser compartides amb col·laboradors o publicades en revistes científiques, "
                        "però només en forma anonimitzada.",
        "withdrawal_title": "Dret a retirar-se:",
        "withdrawal_text": "Té dret a retirar-se de l'estudi en qualsevol moment sense penalització. "
                        "Si vol retirar-se, informi a l'investigador. Si es retira, les seves dades seran eliminades.",
        "acceptance_title": "Acceptació del consentiment:",
        "acceptance_text": "Fent clic a 'Hi estic d'acord', confirmes que has llegit i entès la informació proporcionada i acceptes participar.",
        "next": "Següent",
        "name": "Nom:",
        "surname": "Cognoms",
        "email": "Correu electrònic:",
        "prefix": "Prefixe del país:",
        "phone": "Telèfon:",
        "age": "Edat:",
        "age_options": ["Menys de 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        "gender": "Sexe/Gènere:",
        "gender_options": ["Home", "Dona", "Prefereixo no especificar", "Altres"],
        "english_level": "Nivell d'anglès (1-5):",
        "english_level_options": ["1", "2", "3", "4", "5"],
        "eye_condition": "Portes ulleres/lents de contacte o tens problemes de visió?",
        "eye_condition_options": ["Sí, ulleres/lents de contacte", "Sí, problemes de visió", "Sí, ambdues coses", "No"],
        "eye_problems": "Problemes de visió (si s'aplica):",
        "eye_problems_options": ["Miopia", "Hipermetropia", "Astigmatisme", "Presbícia", "Altres", "Cap"],
        "lens":"Tipus de lents (si s'aplica):",
        "lens_options": ["Monofocals", "Bifocals", "Multifocals", "Progressives", "Ocupacionals", "Altres"],
        "consent_agreement": "He llegit i accepto participar.",
        "agree": "Hi estic d'acord",
        "submit": "Enviar",
        "consent_required": "Has d'acceptar per participar abans de continuar.",
        "submission_complete": "Les teves dades s'han registrat correctament. Gràcies!"
    },
    "Spanish": {
        "title": "Consentimiento para la prueba DocVQA de seguimiento ocular",
        "consent_title": "INFORMACIÓN DE CONSENTIMIENTO",
        "purpose_title": "Propósito del estudio:",
        "purpose_text": "Este estudio tiene como objetivo comprender cómo los movimientos oculares "
                        "se relacionan con la comprensión lectora al ver imágenes con texto. "
                        "Esperamos aprender cómo las personas enfocan su atención en diferentes partes de una imagen "
                        "y cómo esto influye en la comprensión de la información presentada. Esta investigación podría "
                        "contribuir a mejorar las estrategias de comprensión lectora.",
        "confidentiality_title": "Confidencialidad y Privacidad de los datos:",
        "confidentiality_text": "- Sus datos de seguimiento ocular se almacenarán de forma segura\n"
                        "- Los datos seran anonimizados y se te asignara un ID único.\n"
                        "- Solo los investigadores autorizados tendrán acceso.\n"
                        "- Los datos podrían ser compartidos con colaboradores o publicados en revistas científicas, "
                        "pero solo de forma anonimizada.",
        "withdrawal_title": "Derecho a retirarse:",
        "withdrawal_text": "Tiene derecho a retirarse del estudio en cualquier momento sin penalización. "
                        "Si desea retirarse, informe al investigador. Si se retira, sus datos serán eliminados.",
        "acceptance_title": "Aceptación del consentimiento:",
        "acceptance_text": "Al hacer clic en 'Estoy de acuerdo', confirma que ha leído y entendido la información proporcionada y acepta participar.",
        "next": "Siguiente",
        "name": "Nombre:",
        "surname": "Apellidos:",
        "email": "Correo electrónico:",
        "prefix": "Prefijo del país:",
        "phone": "Teléfono:",
        "age": "Edad:",
        "age_options": ["Menos de 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        "gender": "Sexo/Género:",
        "gender_options": ["Hombre", "Mujer", "Prefiero no especificar", "Otro"],
        "english_level": "Nivel de inglés (1-5):",
        "english_level_options": ["1", "2", "3", "4", "5"],
        "eye_condition": "¿Usas gafas/lentillas o tienes problemas de visión?",
        "eye_condition_options": ["Sí, gafas/lentillas", "Sí, problemas de visión", "Sí, ambos", "No"],
        "eye_problems": "Problemas de visión (si aplica):",
        "eye_problems_options": ["Miopía", "Hipermetropía", "Astigmatismo", "Presbicia", "Otro", "Ninguno"],
        "lens":"Tipo de lentes (si aplica):",
        "lens_options": ["Monofocales", "Bifocales", "Multifocal", "Progresivas", "Ocupacionales", "Otras"],
        "consent_agreement": "He leído y acepto participar.",
        "agree": "Estoy de acuerdo",
        "submit": "Enviar",
        "consent_required": "Debes aceptar para participar antes de continuar.",
        "submission_complete": "Tus datos han sido registrados con éxito. ¡Gracias!"
    }
}


def generate_user_id(name):
    initials = "".join([word[0] for word in name.split()]).upper()  #get initials

    #check if initials already exist
    counter = 1
    base_initials = initials
    while check_initials_exist(initials):
        initials = f"{base_initials}{counter}"
        counter += 1

    now = datetime.now()
    date=now.strftime("%Y%m%d_%H%M%S") #date format YYYYMMDD_HHMMSS

    return f"{initials}-{date}"

def check_initials_exist(initials):
    """ Check if initials already exist in the participant IDs file """
    with open(PARTICIPANT_IDS_FILE, "r") as f:
        lines = f.readlines()
        for line in lines:
            existing_id = line.strip()
            if existing_id.startswith(initials + "-"):
                return True
    return False

class ParticipantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Consent Form")
        #self.root.geometry("1000x570")  

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 1000
        window_height = 670

        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)

        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        self.language_var = tk.StringVar(value="English")

        self.participant_data = {"participant_id": None}  

        self.create_language_selection_page()
    
    def set_ID(self):
        name = self.participant_data["name"] + " " + self.participant_data["surname"]
        self.participant_data["participant_id"] = generate_user_id(name)

        #save participant ID to .txt file
        with open(PARTICIPANT_IDS_FILE, "a") as f:
            f.write(self.participant_data["participant_id"] + "\n")

        #print the last item in the file
        with open(PARTICIPANT_IDS_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                print(f"Generated and saved Participant ID: {lines[-1]}")

    def create_language_selection_page(self):
        """ Language selection page """
        self.clear_window()

        tk.Label(self.root, text="Select Language / Selecciona Idioma:", font=("Arial", 16, "bold")).pack(pady=10)
        self.language_menu = ttk.Combobox(self.root, textvariable=self.language_var, values=list(translations.keys()), state="readonly", font=("Arial", 14))
        self.language_menu.pack(pady=10)

        start_button = tk.Button(self.root, text="Start / Iniciar", font=("Arial", 14), command=self.create_data_entry_page)
        start_button.pack(pady=10)

    
    def create_data_entry_page(self):
        """ Data entry page """
        self.clear_window()
        lang = translations[self.language_var.get()]
        first_questions = ["name", "email", "phone"]
        
        tk.Label(self.root, text=lang["name"], font=("Arial", 14)).grid(row=0, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        entry = tk.Entry(self.root, width=30, font=("Arial", 14))
        entry.grid(row=1, column=0, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, "name", entry)

        tk.Label(self.root, text=lang["surname"], font=("Arial", 14)).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(50, 10))
        entry = tk.Entry(self.root, width=30, font=("Arial", 14))
        entry.grid(row=1, column=1, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, "surname", entry)

        tk.Label(self.root, text=lang["prefix"], font=("Arial", 14)).grid(row=2, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        entry = tk.Entry(self.root, width=30, font=("Arial", 14))
        entry.grid(row=3, column=0, sticky=tk.W, pady=10, padx=(50,10))
        setattr(self, "prefix", entry)

        tk.Label(self.root, text=lang["phone"], font=("Arial", 14)).grid(row=4, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        entry = tk.Entry(self.root, width=30, font=("Arial", 14))
        entry.grid(row=5, column=0, sticky=tk.W, pady=10, padx=(50,10))
        setattr(self, "phone", entry)

        tk.Label(self.root, text=lang["email"], font=("Arial", 14)).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(50, 10))
        entry = tk.Entry(self.root, width=30, font=("Arial", 14))
        entry.grid(row=3, column=1, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, "email", entry)

        dropdown="age"
        tk.Label(self.root, text=lang[dropdown], font=("Arial", 14)).grid(row=4, column=1, sticky=tk.W, pady=2, padx=(50, 10))
        var = tk.StringVar()
        menu = ttk.Combobox(self.root, textvariable=var, values=lang[f"{dropdown}_options"], state="readonly", font=("Arial", 14))
        menu.grid(row=5, column=1, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, f"{dropdown}_var", var)

        dropdown="gender"
        tk.Label(self.root, text=lang[dropdown], font=("Arial", 14)).grid(row=6, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        var = tk.StringVar()
        menu = ttk.Combobox(self.root, textvariable=var, values=lang[f"{dropdown}_options"], state="readonly", font=("Arial", 14))
        menu.grid(row=7, column=0, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, f"{dropdown}_var", var)

        dropdown="english_level"
        tk.Label(self.root, text=lang[dropdown], font=("Arial", 14)).grid(row=6, column=1, sticky=tk.W, pady=2, padx=(50, 10))
        var = tk.StringVar()
        menu = ttk.Combobox(self.root, textvariable=var, values=lang[f"{dropdown}_options"], state="readonly", font=("Arial", 14))
        menu.grid(row=7, column=1, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, f"{dropdown}_var", var)

        dropdown="eye_condition"
        tk.Label(self.root, text=lang[dropdown], font=("Arial", 14)).grid(row=8, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        var = tk.StringVar()
        menu = ttk.Combobox(self.root, textvariable=var, values=lang[f"{dropdown}_options"], state="readonly", font=("Arial", 14))
        menu.grid(row=9, column=0, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, f"{dropdown}_var", var)

        #eye problems dropdown (conditional on eye_condition)
        tk.Label(self.root, text=lang["eye_problems"], font=("Arial", 14)).grid(row=8, column=1, sticky=tk.W, pady=2, padx=(50, 10))
        problems_options = lang["eye_problems_options"]
        self.eye_problems_vars = []

        for i, problem in enumerate(problems_options):
            var = tk.IntVar() 
            checkbox = tk.Checkbutton(self.root, text=problem, variable=var, font=("Arial", 14))
            checkbox.grid(row=9+i, column=1, sticky=tk.W, pady=2, padx=(50, 10))
            self.eye_problems_vars.append(var)

        dropdown="lens"
        tk.Label(self.root, text=lang[dropdown], font=("Arial", 14)).grid(row=10, column=0, sticky=tk.W, pady=2, padx=(50, 10))
        var = tk.StringVar()
        menu = ttk.Combobox(self.root, textvariable=var, values=lang[f"{dropdown}_options"], state="readonly", font=("Arial", 14))
        menu.grid(row=11, column=0, sticky=tk.W, pady=10, padx=(50, 10))
        setattr(self, f"{dropdown}_var", var)

  
        next_button = tk.Button(self.root, text=lang["next"], font=("Arial", 14), command=self.create_consent_agreement_page)
        next_button.grid(row=11+(len(problems_options)), column=0, columnspan=2, pady=10)
       

    def create_consent_agreement_page(self):
        """ Consent agreement page """
        self.save_participant_data()
        self.clear_window()
        lang = translations[self.language_var.get()]

        tk.Label(self.root, text=lang["title"], font=("Arial", 18, "bold")).pack(pady=10)

        #text frame
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=10, padx=20, fill="both", expand=True)

        #scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        #long text box
        consent_text = tk.Text(text_frame, wrap="word", font=("Arial", 14), padx=10, pady=10, height=15, width=80, yscrollcommand=scrollbar.set)
        consent_text.pack(fill="both", expand=True)

        #scrollbar linkage
        scrollbar.config(command=consent_text.yview)

        #config tags 
        consent_text.tag_configure("bold", font=("Arial", 15, "bold"))
        consent_text.tag_configure("italic", font=("Arial", 14, "italic"))
        consent_text.tag_configure("bold_italic", font=("Arial", 14, "bold", "italic"))

        # Insert formatted text with tags
        consent_text.insert("0.5", lang["consent_title"] , "bold")

        consent_text.insert("end", "\n\n" + lang["purpose_title"] + "\n", "bold_italic")
        consent_text.insert("end", lang["purpose_text"])

        consent_text.insert("end", "\n\n" + lang["confidentiality_title"] + "\n", "bold_italic")
        consent_text.insert("end", lang["confidentiality_text"])

        consent_text.insert("end", "\n\n" + lang["withdrawal_title"] + "\n", "bold_italic")
        consent_text.insert("end", lang["withdrawal_text"])

        consent_text.insert("end", "\n\n" + lang["acceptance_title"] + "\n", "bold_italic")
        consent_text.insert("end", lang["acceptance_text"], "italic") 

        consent_text.config(state="disabled")  #read-only mode

        #checkbox consent
        self.consent_var = tk.IntVar()
        consent_checkbox = tk.Checkbutton(self.root, text=lang["agree"], variable=self.consent_var, font=("Arial", 14))
        consent_checkbox.pack(pady=10)
    
        #submit button
        submit_button = tk.Button(self.root, text=lang["submit"], font=("Arial", 14), command=self.submit_data)
        submit_button.pack(pady=10)

        self.create_file()  #create file for participant data


    def save_participant_data(self):
        """ Save participant data """
        participant_eye_problems = []
        lang = translations[self.language_var.get()]
        eye_problem_options = lang["eye_problems_options"]
        for i, val in enumerate(self.eye_problems_vars):
            if val.get() == 1:
                participant_eye_problems.append(eye_problem_options[i])

        self.participant_data.update({
            "name": self.name.get(),
            "surname": self.surname.get(),
            "email": self.email.get(),
            "prefix": self.prefix.get(),
            "phone": self.phone.get(),
            "age": self.age_var.get(),
            "gender": self.gender_var.get(),
            "english_level": self.english_level_var.get(),
            "eye_condition": self.eye_condition_var.get(),
            "eye_problems": participant_eye_problems
        })

    def create_file(self):
        """ Save data to JSON file """
        self.set_ID()

        filename = os.path.join(DATA_DIR, f"{self.participant_data['participant_id']}.json")

        #save data to JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.participant_data, f, indent=4)

        print(f"Data saved to {filename}")  

    def submit_data(self):
        """ Save data and exit """
        messagebox.showinfo("Success", translations[self.language_var.get()]["submission_complete"])
        self.root.quit()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

root = tk.Tk()
app = ParticipantGUI(root)
root.mainloop()
