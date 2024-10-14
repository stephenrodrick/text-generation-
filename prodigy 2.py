import random
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox
import string
import json

def preprocess_text(text):
    """
    Preprocesses the input text by normalizing and removing punctuation.

    Args:
        text (str): The raw training text.

    Returns:
        str: The preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Remove extra whitespace
    text = ' '.join(text.strip().split())
    return text

def build_markov_model_word(text):
    """
    Builds a word-level Markov model based on the input text.

    Args:
        text (str): The training text to build the model from.

    Returns:
        dict: A nested dictionary representing the Markov model where
              markov_model[current_word][next_word] = probability.
    """
    markov_model = defaultdict(lambda: defaultdict(int))
    words = text.split()

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        markov_model[current_word][next_word] += 1

    # Convert counts to probabilities
    for current_word, transitions in markov_model.items():
        total = sum(transitions.values())
        for next_word in transitions:
            transitions[next_word] /= total  # Probability of next_word given current_word

    return markov_model

def generate_text_word(markov_model, start_word=None, length=50):
    """
    Generates text using the provided word-level Markov model.

    Args:
        markov_model (dict): The Markov model for text generation.
        start_word (str, optional): The starting word for text generation.
                                    If None, a random word is chosen.
        length (int): The desired number of words in the generated text.

    Returns:
        str: The generated text.
    """
    if not markov_model:
        raise ValueError("The Markov model is empty. Provide a non-empty training text.")

    if start_word:
        if start_word not in markov_model:
            messagebox.showwarning("Warning", f"Start word '{start_word}' not found in the training text. Choosing a random start word.")
            start_word = None

    if not start_word:
        start_word = random.choice(list(markov_model.keys()))

    current_word = start_word
    generated_words = [current_word]

    for _ in range(length - 1):
        transitions = markov_model.get(current_word, None)
        if not transitions:
            # If the current_word has no outgoing transitions, choose a new random start
            current_word = random.choice(list(markov_model.keys()))
            generated_words.append(current_word)
            continue

        # Choose the next word based on transition probabilities
        next_words = list(transitions.keys())
        probabilities = list(transitions.values())
        current_word = random.choices(next_words, weights=probabilities, k=1)[0]
        generated_words.append(current_word)

    return ' '.join(generated_words)

def save_markov_model(markov_model, filename):
    """
    Saves the Markov model to a JSON file.

    Args:
        markov_model (dict): The Markov model to save.
        filename (str): The filename to save the model to.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        # Convert defaultdict to regular dict for JSON serialization
        model_dict = {k: dict(v) for k, v in markov_model.items()}
        json.dump(model_dict, f, ensure_ascii=False, indent=4)

def load_markov_model(filename):
    """
    Loads the Markov model from a JSON file.

    Args:
        filename (str): The filename to load the model from.

    Returns:
        dict: The loaded Markov model.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
        # Convert back to defaultdict
        markov_model = defaultdict(lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in model_dict.items()})
    return markov_model

# Initialize global variable for the Markov model
markov_model_global = None

# GUI Functions
def generate():
    global markov_model_global
    training_text = text_input.get("1.0", tk.END)
    start = entry_start.get().lower() if entry_start.get() else None
    try:
        length = int(entry_length.get())
        if length <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Generated length must be a positive integer.")
        return

    if not training_text.strip():
        messagebox.showerror("Error", "Training text cannot be empty.")
        return

    # Preprocess the training text
    training_text = preprocess_text(training_text)

    # Build the word-level Markov model
    markov_model_global = build_markov_model_word(training_text)

    # Generate text
    try:
        generated = generate_text_word(markov_model_global, start_word=start, length=length)
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
        return

    # Output the generated text
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, generated)

def save_model():
    if not markov_model_global:
        messagebox.showerror("Error", "No model to save. Generate a model first.")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if filename:
        save_markov_model(markov_model_global, filename)
        messagebox.showinfo("Success", f"Model saved to {filename}.")

def load_model_gui():
    global markov_model_global
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if filename:
        try:
            markov_model_global = load_markov_model(filename)
            messagebox.showinfo("Success", f"Model loaded from {filename}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

# Set up the GUI
root = tk.Tk()
root.title("Word-Level Markov Chain Text Generator")

# Training Text Input
tk.Label(root, text="Training Text:").grid(row=0, column=0, padx=10, pady=10, sticky='nw')
text_input = tk.Text(root, height=10, width=70)
text_input.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

# Starting word
tk.Label(root, text="Starting Word:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
entry_start = tk.Entry(root, width=50)
entry_start.grid(row=1, column=1, padx=10, pady=10, sticky='w')

# Generated length
tk.Label(root, text="Generated Length (words):").grid(row=2, column=0, padx=10, pady=10, sticky='e')
entry_length = tk.Entry(root, width=50)
entry_length.insert(0, "50")
entry_length.grid(row=2, column=1, padx=10, pady=10, sticky='w')

# Generate button
tk.Button(root, text="Generate Text", command=generate).grid(row=3, column=1, padx=10, pady=10, sticky='w')

# Save and Load model buttons
tk.Button(root, text="Save Model", command=save_model).grid(row=3, column=2, padx=10, pady=10, sticky='w')
tk.Button(root, text="Load Model", command=load_model_gui).grid(row=3, column=3, padx=10, pady=10, sticky='w')

# Text output
tk.Label(root, text="Generated Text:").grid(row=4, column=0, padx=10, pady=10, sticky='nw')
text_output = tk.Text(root, height=10, width=70)
text_output.grid(row=4, column=1, columnspan=3, padx=10, pady=10)

root.mainloop()
