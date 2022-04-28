import Scavetta_Victoria_text_cleaning as tc
import Scavetta_Victoria_models as m


def main():
    # Import the datasets and clean them
    tc.clean()

    # Train the models and make predictions
    m.make_predictions()


if __name__ == "__main__":
    main()
