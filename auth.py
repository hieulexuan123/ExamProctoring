import tkinter as tk
from tkinter import filedialog, messagebox
import json
from PIL import Image, ImageTk
import face_recognition

class Authentication:
    def __init__(self, callback):
        self.root = tk.Tk()
        self.root.title("Exam Proctoring")
        self.root.geometry("600x600")
        self.root.configure(bg="#f0f0f0")

        self.filepath = None
        self.preview_image = None
        self.callback = callback

        self.createLoginPage()
        self.createRegistrationPage()

        self.showLoginPage()
        self.root.mainloop()

    def createLoginPage(self):
        self.login_frame = tk.Frame(self.root, bg="#f0f0f0", pady=20)

        tk.Label(self.login_frame, text="Username:", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.login_username_entry = tk.Entry(self.login_frame, font=("Arial", 12))
        self.login_username_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.login_frame, text="Password:", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.login_password_entry = tk.Entry(self.login_frame, font=("Arial", 12), show='*')
        self.login_password_entry.grid(row=1, column=1, padx=10, pady=5)

        login_button = tk.Button(self.login_frame, text="Login", font=("Arial", 12), bg="#28b463", fg="white", padx=10, pady=5, command=self.login)
        login_button.grid(row=2, column=0, columnspan=2, pady=10)

        switch_to_register_button = tk.Button(self.login_frame, text="Go to Register", font=("Arial", 10), command=self.showRegistrationPage)
        switch_to_register_button.grid(row=3, column=0, columnspan=2, pady=5)

    def createRegistrationPage(self):
        self.registration_frame = tk.Frame(self.root, bg="#f0f0f0", pady=20)

        tk.Label(self.registration_frame, text="Username:", font=("Arial", 12), bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.reg_username_entry = tk.Entry(self.registration_frame, font=("Arial", 12))
        self.reg_username_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.registration_frame, text="Password:", font=("Arial", 12), bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.reg_password_entry = tk.Entry(self.registration_frame, font=("Arial", 12), show='*')
        self.reg_password_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.registration_frame, text="Upload your image face:", font=("Arial", 12), bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        upload_button = tk.Button(self.registration_frame, text="Upload", font=("Arial", 10), command=self.uploadImage)
        upload_button.grid(row=2, column=1, padx=5, sticky="w")

        self.register_button = tk.Button(self.registration_frame, text="Register", font=("Arial", 12), bg="#5dade2", fg="white", padx=10, pady=5, command=self.register)
        self.register_button.grid(row=4, column=0, columnspan=3, pady=10)

        self.switch_to_login_button = tk.Button(self.registration_frame, text="Go to Login", font=("Arial", 10), command=self.showLoginPage)
        self.switch_to_login_button.grid(row=5, column=0, columnspan=3, pady=5)

    def uploadImage(self):
        filepath = filedialog.askopenfilename(title="Select Face Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if filepath:
            self.filepath = filepath
            self.show_preview(filepath)

    def show_preview(self, filepath):
        img = Image.open(filepath)
        img = img.resize((224, 224))
        self.preview_image = ImageTk.PhotoImage(img)

        preview = tk.Label(self.registration_frame, image=self.preview_image)
        preview.grid(row=3, column=0, columnspan=3, pady=10, sticky='nswe')

    def register(self):
        username = self.reg_username_entry.get()
        password = self.reg_password_entry.get()

        #Check empty fields
        if not username or not password or not self.filepath:
            messagebox.showwarning("Warning", "All fields are required!")
            return
        
        # Load db
        users_db = {}
        try:
            with open('users.json', 'r') as file:
                users_db = json.load(file)
        except FileNotFoundError:
            pass

        #Check duplicated username
        if username in users_db.keys():
            messagebox.showwarning("Warning", "Username is already used")
            return

        face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(self.filepath))
        print(face_encodings)
        if not face_encodings:
            messagebox.showwarning("Warning", "No face found in the image!")
            return
        
        users_db[username] = {
            'password': password,
            'face_encoding': face_encodings[0].tolist()
        }

        with open('users.json', 'w') as file:
            json.dump(users_db, file)

        messagebox.showinfo("Success", "User registered successfully!")
        self.showLoginPage()

    def login(self):
        username = self.login_username_entry.get()
        password = self.login_password_entry.get()

        try:
            with open('users.json', 'r') as file:
                users_db = json.load(file)
        except FileNotFoundError:
            return

        #Check username and password
        user = users_db.get(username)
        if not user or user['password'] != password:
            messagebox.showerror("Error", "Invalid username or password!")
            return

        messagebox.showinfo("Success", "Login successful!")
        self.root.destroy()
        self.callback(user)

    def showLoginPage(self):
        self.registration_frame.pack_forget()
        self.login_frame.pack(pady=10)

    def showRegistrationPage(self):
        self.login_frame.pack_forget()
        self.registration_frame.pack(pady=10)
