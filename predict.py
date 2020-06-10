import argparse
#  Import the functions to make the prediction
from utility_functions import (
    load_cnn_model,
    cat_to_name,
    process_image,
    predict,
    display_prediction
)

# Function to get input argument


def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line argumentsusing add_argument() from ArguementParser method
    parser.add_argument('image', type=str, help='image path')
    parser.add_argument('checkpoint', type=str, help='saved model path')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k most probable classes')
    parser.add_argument('--category_names', type=str, default=' ',
                        help='json file use a mapping of categories to real names')
    return parser.parse_args()


def main():

    print("\n############### Predict ####################\n")
    # Get the arguments
    args = get_input_args()

    # Get the saved model
    model = load_cnn_model(args.checkpoint)
    # Get the label mapping file
    cat_to_n = cat_to_name(args.category_names)

    # Display the prediction
    display_prediction(args.image, model, cat_to_n, args.top_k)

    print('\n ## Prediction Successfully Completed! ## \n', flush=True)


if __name__ == "__main__":
    main()
