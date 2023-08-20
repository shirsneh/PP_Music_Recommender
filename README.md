# Federated Matrix Factorization Song Recommender

This project implements a privacy-preserving federated matrix factorization recommendation system using Python and the Tkinter library for GUI. The system enables users to receive song recommendations while keeping their data encrypted and secure. It utilizes the Paillier encryption scheme for secure computation and federated learning techniques to collectively improve recommendations without sharing raw user data.

## Features 
### User Profiles: 
Each user has an initial profile and can update their ratings for songs.
### Secure Encryption: 
User passwords are encrypted using SHA-512 hash, and Paillier encryption is used for secure data sharing.
### Federated Learning: 
The server and users collaboratively update song profiles and user ratings without directly sharing their data.
### Recommendation System: 
The system offers personalized song recommendations based on user profiles and their interactions.
### Graphical User Interface (GUI): 
The Tkinter-based GUI enables user login, signup, and interaction with the system.

## How to Run
1. Install the required Python packages using `pip`:

   ```bash
   pip install numpy phe

2. Run the application:

   ```bash
   python main.py

The GUI window will open, allowing users to log in, sign up, and receive song recommendations.

## Usage Instructions

1. **Login or Signup:** Users can log in using their user ID and password or sign up as new users.
   
2. **Recommendations:** After logging in, users can view recommended songs based on their profile.
   
3. **Update Ratings:** Users can update the ratings of songs they've interacted with.
   
4. **Secure Processing:** The system uses privacy-preserving techniques to update profiles and offer recommendations.
