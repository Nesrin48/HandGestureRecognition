import argparse  # To create Parse argument
from utility_functions import *  # to use the functions of the classifier
import sys

# Function to get input argument


def get_input_args():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line argumentsusing add_argument() from ArguementParser method
    parser.add_argument('data_dir', type=str, default='handgesture_dataset/',
                        help='path to hangesture images folder')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')

    parser.add_argument('--epochs',  type=int, default=5,
                        help='number of epochs to train the model')

    return parser.parse_args()

# The main function to lunch the train and save the model


def main():
    print("\n############### Train ####################\n")
    # Get the arguments
    args = get_input_args()

    # Get hand gestures labales
    results_dic = get_hand_gesture_labels(args.data_dir)
    # Load the dataset
    X, Y = load_dataset(args.data_dir, results_dic)
    # Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(X, Y)
    # One hot encoding for lables
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    # Construction of model
    model = cnn_model()
    # Train the model
    train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs
    )
    # Test the model
    test_model(model, X_test, y_test)
    # Save the model
    save_model(args.save_dir, model)
    # complete.
    print('\n ## Training Successfully Completed! ## \n', flush=True)

if __name__ == "__main__":
    sys.stdout.flush()
    main()
