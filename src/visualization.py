import os
import matplotlib.pyplot as plt
import datetime


def plot_loss(losses, filename):
    directorio = "./plots/"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    filename = directorio + filename
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)


def plot_accuracy(accuracies, filename):
    directorio = "./plots/"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    filename = directorio + filename
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label="Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(filename)
