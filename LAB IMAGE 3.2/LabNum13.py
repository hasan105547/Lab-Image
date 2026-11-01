# lab 15
# Write a MATLAB/Python program for text-to-speech conversion and record 
# speech signal.

import pyttsx3

# Create text-to-speech engine
engine = pyttsx3.init()

# Input text
text = "Hello, My name is Hasanuzzaman. I am a student of Information and Communication Engineering. "

# Speak and save speech
engine.say(text)
engine.save_to_file(text, "recoredspeech.wav")
engine.runAndWait()

print("Speech played and saved successfully.")