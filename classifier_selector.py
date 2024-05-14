from intent_classifier import IntentClassifier
from gpt_intent_classifier import GPTIntentClassifier
from bert_intent_classifier import BERTIntentClassifier

def choose_intent_classifier(classifier_name: str = 'GPT',
                            model_name: str = "gpt-3.5-turbo",
                            classifier_type: str = 'zero-shot',
                            train_ds_path: str = "./data/atis/train.tsv",
                            test_ds_path: str = "./data/atis/test.tsv", 
                             tokenizer_name: str = 'bert-base-uncased') -> IntentClassifier:
    """
    Choose an intent classifier class based on the classifier_name.

    Parameters:
        classifier_name (str): Name of the classifier to choose. Options are "GPT" or "BERT".
        model_name (str): Name of the model to use. Defaults to "gpt-3.5-turbo".
        classifier_type (str): Type of classifier. Defaults to 'few-shot'.
        train_ds_path (str): Path to the training dataset. Defaults to "./data/atis/train.tsv".
        tokenizer_name (str): Name of the tokenizer to use. Defaults to an empty string.

    Returns:
        IntentClassifier: An instance of the selected intent classifier class.

    Raises:
        ValueError: If the provided classifier_name is unsupported.
    """
    if classifier_name == "GPT":
        # If classifier_name is "GPT", instantiate and return a GPTIntentClassifier
        return GPTIntentClassifier(model_name=model_name, classifier_type=classifier_type, train_ds_path=train_ds_path, test_ds_path=test_ds_path)
    elif classifier_name == "BERT":
        # If classifier_name is "BERT", instantiate and return a BERTIntentClassifier
        return BERTIntentClassifier()  # Note: BERTIntentClassifier implementation is not provided
    else:
        # If classifier_name is not supported, raise a ValueError
        raise ValueError("Unsupported classifier name: " + classifier_name)

# Example usage:
if __name__ == "__main__":
    classifier_name = "GPT"  # or "BERT"
   
    
    classifier = choose_intent_classifier(classifier_name=classifier_name)
    print(classifier.model_name)
    print(classifier.classifier_type)
    print(classifier.train_ds_path)
    print(classifier.test_ds_path)
    