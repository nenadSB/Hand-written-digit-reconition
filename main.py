from train import train_model
from evaluate import evaluate_model

def main():
    """Main function to train and evaluate the model."""
    print("Training the model...")
    train_model()

    print("\nEvaluating the model...")
    evaluate_model()

if __name__ == "__main__":
    main()