from abc import ABC, abstractmethod
# from classifier_selector import choose_intent_classifier

class IntentClassifier(ABC):
    """
    Abstract base class for intent classifier models.

    Methods:
        is_ready() -> bool:
            Check if the classifier is ready to classify intents.

        classify_intent(prompt: str) -> str:
            Classify the intent of a given prompt.

        load() -> None:
            Load the intent classifier model.

        This class serves as a blueprint for implementing concrete classifier models.
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the classifier is ready to classify intents.

        Returns:
            bool: True if the classifier is ready, False otherwise.
        """
        pass

    @abstractmethod
    def classify_intent(self, prompt: str) -> str:
        """
        Classify the intent of a given prompt.

        Parameters:
            prompt (str): The prompt to classify.

        Returns:
            str: The classified intent.
        """
        pass
    
    @abstractmethod
    def load(self, test_size: int) -> None:
        """
        Load the intent classifier model from the specified path.

        Parameters:
            test_size (int): Evaluate the dataset on test_size cases
        """
        pass
    