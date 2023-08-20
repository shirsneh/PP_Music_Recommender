import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import phe as paillier
import hashlib
from time import time
import json

class User:
    def __init__(self, user_id, init_profile, actual_ratings, mask, lr, pubkey, privkey,
                 pred_ratings, encrypted_password):
        self.user_id = user_id
        self.encrypted_password = encrypted_password
        self.user_profile = np.array(init_profile)
        self.actual_ratings = actual_ratings
        self.pred_ratings = pred_ratings
        self.mask = mask
        self.lr = lr
        self.pubkey = pubkey
        self.privkey = privkey

    def update_rating(self, song_id, rating):
        self.actual_ratings[song_id] = rating
        self.mask[song_id] = 1 if rating >= 3 else 0


class Server:
    def __init__(self, init_song_profile, lr, users_count, pubkey):
        self.song_profile = np.array(init_song_profile)
        self.lr = lr
        self.users_count = users_count
        self.pubkey = pubkey

    def update_song_profile(self, user_gradient):
        new_song_profile = (self.lr / self.users_count) * user_gradient
        self.song_profile = self.song_profile - new_song_profile

def update_ratings_in_json(user_id, song_id, actual_ratings, mask):
    with open('dataset.json', 'r') as json_file:
        data = json.load(json_file)

    for user in data["users"]:
        if user["type"] == "users" and user["user_id"] == user_id:
            user["actual_ratings"] = [int(r) for r in actual_ratings]
            user["mask"] = [int(m) for m in mask]
            break

    with open('dataset.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_data(pubkey, privkey, encrypt=False):
    with open('dataset.json') as f:
        data = json.load(f)

    users = {}
    for userData in data["users"]:
        user = User(userData["user_id"], np.array(userData["init_profile"]),
                    np.array(userData["actual_ratings"]), np.array(userData["mask"]), userData["lr"],
                    pubkey, privkey, np.array(userData["pred_ratings"]), userData["encrypted_password"])
        users[user.user_id] = user

    song_profiles = [song_data["song_profile"] for song_data in data["songs"]]
    lr = data["songs"][0]["lr"]

    if encrypt:
        song_profiles = [[pubkey.encrypt(value) for value in row] for row in song_profiles]
    song_profiles = np.array(song_profiles)

    server = Server(song_profiles, lr, len(users), pubkey)

    return users, server

def compute_gradient(user, song_profile, encrypt=False):
    if encrypt:
        new_song_profile = [[user.privkey.decrypt(value) for value in row] for row in song_profile]
        song_profile = np.array(new_song_profile)
    user.pred_ratings = np.dot(user.user_profile, song_profile.T)
    gradient = 2 * np.dot((user.pred_ratings - user.actual_ratings) * user.mask, song_profile)
    x = user.lr * gradient
    user.user_profile -= x

    if encrypt:
        encrypted_gradient = np.array([user.pubkey.encrypt(x) for x in gradient])
        return encrypted_gradient, compute_loss(user)
    else:
        return gradient, compute_loss(user)


def compute_loss(user):
    return np.sum(((user.actual_ratings - user.pred_ratings) * user.mask) ** 2)


def update_matrices(users, server):
    convergence_iters_threshold = 50
    iter = 0
    prev_loss = 0
    local_convergence = 0
    lossArray = []
    while True:
        startTime = time()
        total_loss = 0
        for user_id, user_data in users.items():
            gradient, loss = compute_gradient(user_data, server.song_profile, True)
            total_loss += loss
            server.update_song_profile(gradient)
        if np.abs(total_loss - prev_loss) < 0.001:
            local_convergence += 1
            if local_convergence > convergence_iters_threshold:
                print(f'Converged after {iter} iterations and {time() - startTime} seconds')
                break
        else:
            local_convergence = 0
            prev_loss = total_loss

        lossArray.append(total_loss)
        iter += 1
    print("Matrices are updated successfully :)")


class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Recommender")

        # Initialize Paillier encryption keys
        self.pubkey, self.privkey = paillier.generate_paillier_keypair()

        # Load data
        self.users, self.server = load_data(pubkey=self.pubkey, privkey=self.privkey, encrypt=True)

        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack(padx=100, pady=100)

        self.login_label = tk.Label(self.login_frame, text="User ID:")
        self.login_label.grid(row=0, column=0, sticky="e")

        self.login_entry = tk.Entry(self.login_frame)
        self.login_entry.grid(row=0, column=1)

        self.password_label = tk.Label(self.login_frame, text="Password:")
        self.password_label.grid(row=1, column=0, sticky="e")

        self.password_entry = tk.Entry(self.login_frame, show="*")
        self.password_entry.grid(row=1, column=1)

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login)
        self.login_button.grid(row=2, columnspan=2)

        self.join_button = tk.Button(self.login_frame, text="Join", command=self.show_join_popup)
        self.join_button.grid(row=3, columnspan=2)


    def login(self):
        user_id = self.login_entry.get()
        input_password = self.password_entry.get()

        if user_id.isdigit():
            user_id = int(user_id)
            users = self.users

            if user_id in users:
                user_data = users[user_id]
                encrypted_input_password = self.encrypt_password(input_password)  # Encrypt user input
                if encrypted_input_password == user_data.encrypted_password:
                    self.root.destroy()  # Close the login window
                    self.show_recommendations(user_id)  # Pass user_id instead of user_data
                else:
                    messagebox.showerror("Login Failed", "Invalid password.")
            else:
                messagebox.showerror("Login Failed", "User not found.")
        else:
            messagebox.showerror("Login Failed", "Invalid user id.")

    def show_join_popup(self):
        join_popup = tk.Toplevel(self.root)
        join_popup.title("Join")

        password_label = tk.Label(join_popup, text="Password:")
        password_label.pack()

        password_entry = tk.Entry(join_popup, show="*")
        password_entry.pack()

        confirm_password_label = tk.Label(join_popup, text="Confirm Password:")
        confirm_password_label.pack()

        confirm_password_entry = tk.Entry(join_popup, show="*")
        confirm_password_entry.pack()

        join_button = tk.Button(join_popup, text="Join",
                                command=lambda: self.create_user(password_entry.get(), confirm_password_entry.get()))
        join_button.pack()

    def create_user(self, password, confirm_password):
        if password == confirm_password:
            encrypted_password = self.encrypt_password(password)  # Encrypt with SHA-512 hash
            user_id = len(self.users) + 1
            init_profile = np.array( [0.1,0.2,0.3,0.3,0.7,0.6,0.5,0.3,0.3,0.7,0.6,0.5,0.5,0.6,0.6])
            actual_ratings = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            mask = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            lr = 0.01
            pred_ratings = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

            new_user = {
                "type": "users",
                "user_id": user_id,
                "encrypted_password": encrypted_password,
                "init_profile": init_profile.tolist(),
                "actual_ratings": actual_ratings.tolist(),
                "mask": mask.tolist(),
                "lr": lr,
                "pred_ratings": pred_ratings.tolist()
            }

            self.users[user_id] = User(user_id, init_profile, actual_ratings, mask, lr, self.pubkey, self.privkey,
                                       pred_ratings, encrypted_password)

            with open("dataset.json", "r") as json_file:
                data = json.load(json_file)

            data["users"].append(new_user)

            with open("dataset.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

            messagebox.showinfo("Join", "User joined successfully!")
        else:
            messagebox.showerror("Join Failed", "Passwords do not match.")

    def encrypt_password(self, password):
        # Encrypt password using SHA-512 hash
        hash_object = hashlib.sha512(password.encode())
        return hash_object.hexdigest()

    def show_recommendations(self, user_id):
        root = tk.Tk()
        root.title("Song Recommender")

        users = self.users
        if user_id in users:
            user_data = users[user_id]
            encrypted_password = user_data.encrypted_password
            actual_ratings = user_data.actual_ratings
            mask = user_data.mask
            self.server.update_song_profile(user_data.user_profile)

            rating_frame = tk.Frame(root)
            rating_frame.pack(padx=20, pady=20)

            best_songs_label = tk.Label(rating_frame, text="Recommended Songs")
            best_songs_label.pack()

            best_songs_tree = ttk.Treeview(rating_frame, columns=("Song",), show="headings")  # Remove "Rating" column
            best_songs_tree.heading("Song", text="Song")
            best_songs_tree.pack()

            # Call the update_recommendations method to get best suited songs
            recommended_songs = self.update_recommendations(user_data)

            for song_id, _ in recommended_songs[:10]:  # Use song_id from the tuple
                best_songs_tree.insert("", "end", values=("Song " + str(song_id),))

            update_button = tk.Button(rating_frame, text="Update Rating",
                                      command=lambda: self.show_update_rating_popup(user_data))
            update_button.pack()
            root.mainloop()
        else:
            messagebox.showerror("Error", "User not found.")

    def show_update_rating_popup(self, user_data):
        popup = tk.Toplevel()
        popup.title("Update Rating")
        song_id_label = tk.Label(popup, text="Song ID:")
        song_id_label.pack()
        song_id_entry = tk.Entry(popup)
        song_id_entry.pack()
        rating_label = tk.Label(popup, text="Rating:")
        rating_label.pack()
        rating_entry = tk.Entry(popup)
        rating_entry.pack()
        update_button = tk.Button(popup, text="Update",
                                  command=lambda: self.update_rating(user_data, int(song_id_entry.get()),
                                                                     int(rating_entry.get())))
        update_button.pack()
        popup.mainloop()

    def update_rating(self, user_data, song_id, new_rating):
        user_data.update_rating(song_id, new_rating)
        update_matrices(self.users, self.server)
        # Update the recommendation tables based on updated matrices
        user_id = user_data.user_id
        recommended_songs = self.update_recommendations(self.users[user_id])
        self.update_recommendation_table(recommended_songs)
        messagebox.showinfo("Update Rating", "Rating updated successfully!")

    def update_recommendation_table(self, recommended_songs):
        self.best_songs_tree.delete(*self.best_songs_tree.get_children())
        for song_id, _ in recommended_songs[:10]:  # Use song_id from the tuple
            self.best_songs_tree.insert("", "end", values=("Song " + str(song_id),))

    def update_recommendations(self, user_data):
        user_profile = user_data.user_profile
        song_profile = self.server.song_profile
        predicted_ratings = np.dot(user_profile, song_profile.T)
        unrated_songs = [i for i, mask_value in enumerate(user_data.mask) if mask_value == 0]
        recommended_songs = []
        for song_id in unrated_songs:
            recommended_songs.append((song_id, predicted_ratings[song_id]))
        return recommended_songs

if __name__ == "__main__":
    encrypt = True
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
