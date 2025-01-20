import argparse
from text_anonymizer.preprocessing import apply_clean
from text_anonymizer.training import train_model
from text_anonymizer.model_io import save_model_weights, load_model_weights
from text_anonymizer.anonymizer import anonymize_with_syntf
from text_anonymizer.utils import load_dataset, load_input_text, save_output  

def main():
    parser = argparse.ArgumentParser(description="Social Media Text Anonymization")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'anonymize'], 
                        help="Mode: 'train' to train the model, 'anonymize' to anonymize text.")
    parser.add_argument('--data', type=str, help="Path to dataset for training or input text for anonymization.")
    parser.add_argument('--weights', type=str, help="Path to save or load model weights.")
    parser.add_argument('--output', type=str, help="Path to save the output (optional).")

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            print("Error: Please provide a dataset path with --data.")
            return

        # Load dataset and train model
        train_posts, test_posts = load_dataset(args.data)
        trainer = train_model(train_posts, test_posts, args.weights)

    elif args.mode == 'anonymize':
        if not args.data or not args.weights:
            print("Error: Please provide input text and model weights paths.")
            return

        # Load model weights and anonymize text
        weights = load_model_weights(args.weights)
        input_texts = load_input_text(args.data)
        anonymized_texts = anonymize_with_syntf(input_texts, weights)
        
        if args.output:
            save_output(anonymized_texts, args.output)
        else:
            print("Anonymized Texts:")
            for text in anonymized_texts:
                print(text)

if __name__ == "__main__":
    main()

