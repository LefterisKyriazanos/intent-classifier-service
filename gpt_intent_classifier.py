from intent_classifier import IntentClassifier
from dotenv import load_dotenv
import os
import random
import openai
import pandas as pd
from typing import List, Dict
import pathlib
import jinja2
load_dotenv()

class GPTIntentClassifier(IntentClassifier):
    """
    Intent classifier using OpenAI's GPT models through OpenAI API.

    This classifier utilizes OpenAI's GPT models accessed through the OpenAI API.
    It can handle different GPT models and supports two classifier types: zero-shot and few-shot.

    Parameters:
        model_name (str): Name of the GPT model to use. Defaults to "gpt-3.5-turbo".
        classifier_type (str): Type of classifier to use. Options are "zero-shot" and "few-shot".
                               Defaults to 'few-shot'.
        train_ds_path (str): Path to the training dataset. Defaults to "./data/atis/train.tsv".

    Attributes:
        openai_api_key (str): API key for accessing the OpenAI API.
        model_name (str): Name of the GPT model being used.
        train_ds_path (str): Path to the training dataset.
        classifier_type (str): Type of classifier being used.

    Methods:
        is_ready() -> bool:
            Check if the classifier is ready to classify intents.

        classify_intent(prompt: str) -> str:
            Classify the intent of a given prompt.

        load_model(model_path: str) -> None:
            Load the intent classifier model from the specified path.

    Note:
        To use this class, you need to have an OpenAI API key. Update the 'openai_api_key' attribute
        with your API key before using the classifier.
    """
    def __init__(self, model_name = "gpt-3.5-turbo", classifier_type = 'few-shot', train_ds_path = "./data/atis/train.tsv"):
        
        super().__init__()  # Call superclass constructor
        self.model_name = model_name
        self.train_ds_path = train_ds_path
        self.classifier_type = classifier_type
        # self.prompt_template_file_path = prompt_template_file_path
        self.labeled_intents = []
        self.training_examples = []
        # self.training_prompts = []
        # self.training_intents = []
        # self.ds_columns = ds_columns
        
    def is_ready(self) -> bool:
        """Check if the classifier is ready to classify intents."""
        # Perform a test completion request to OpenAI API
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                  messages=[
                {"role": "system", "content": "Test"},
                {"role": "user", "content": 'Testing Model Readiness'}
            ], 
            )
            # If the completion request was successful, return True
            return True
        except Exception as e:
            # If there was an error, return False
            return False
        
    def create_label_intents(self, intents: list()) -> List[Dict[str, str]]:
        """
        Generate a list of labels and intents based on a list of unique intents.

        Parameters:
            intents (list): A list of unique intents.

        Returns:
            list: A list of dictionaries, where each dictionary contains a label, intent, and description (only for zero-shot classification).

        Example:
            >>> unique_intents = ['flight', 'airfare', 'ground_service']
            >>> labels = create_label_intents(unique_intents)
            >>> print(labels)
            [{'label': 0, 'intent': 'flight', 'description': 'This intent is about obtaining flight information'},
            {'label': 1, 'intent': 'airfare', 'description': 'This intent is about obtaining airfare information'},
            {'label': 2, 'intent': 'ground_service', 'description': 'This intent is about obtaining ground_service information'}]
        """
        labels = []
        label_counter = 0
        
        if self.classifier_type == 'few-shot':
            for intent in intents:
                labels.append({"label": label_counter, "intent": intent})
                label_counter += 1
        else: # prepare labels for zero-shot by adding "description"
            for intent in intents:
                labels.append({"label": label_counter, "intent": intent, "description": f"This intent is about obtaining {intent} information"})
                label_counter += 1
        
        return labels
    
    @staticmethod
    def pick_training_examples(examples: List[Dict[str, any]], max_per_label: int = 10) -> List[Dict[str, any]]:
        """
        Get a subset of examples for each label, limiting to a maximum number of examples per label.
        These Examples are later used as training data for the 'few-shot' classifier. 
        
        Parameters:
            examples (list): A list of examples, where each example is a dictionary containing text and label.
            max_per_label (int): Maximum number of examples per label.

        Returns:
            list: A list of dictionaries, containing a subset of examples for each label.

        Example:
            >>> training_examples = get_few_shot_examples(examples, max_per_label=10)
            >>> print(len(training_examples))
            30  # Assuming there are 3 unique labels with 10 examples each
        """
        training_examples = []
        label_count = {}
        
        # Initialize label_count with all possible labels as keys and values set to 0
        all_labels = set(example["label"] for example in examples)
        for label in all_labels:
            label_count[label] = 0

        # Shuffle examples to ensure randomness
        random.shuffle(examples)

        for example in examples:
            label = example["label"]
            if label not in label_count:
                label_count[label] = 0

            # Check if the maximum limit for the label has been reached
            if label_count[label] < max_per_label:
                training_examples.append(example)
                label_count[label] += 1

            # Break the loop if all labels reached the maximum limit
            if all(count >= max_per_label for count in label_count.values()):
                break

        return training_examples
    
    # @staticmethod
    def create_examples(self, training_ds: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Create a list of examples with text and corresponding labels based on user prompts and intents.
        Each training prompt is mapped to a target_label id (unique intent id).

        Parameters:
            training_ds: pandas DF with 2 (str) columns ['user_prompt', 'intents']

        Returns:
            list: A list of dictionaries, where each dictionary contains text and its corresponding label.

        Example:
            >>> training_prompts = ['ground transportation for las vegas', 'kansas city to las vegas economy']
            >>> training_intents = ['ground_service', 'flight']
            >>> labels = [{'label': 0, 'intent': 'ground_service'}, {'label': 1, 'intent': 'flight'}]
            >>> examples = create_examples(user_prompts, intents, labels)
            >>> print(examples)
            [{'text': 'ground transportation for las vegas', 'label': 0}, {'text': 'kansas city to las vegas economy', 'label': 1}]
        """
        examples = []
        training_prompts = training_ds['user_prompt'].tolist()
        training_intents = training_ds['intents'].tolist()
        
        for prompt, intent in zip(training_prompts, training_intents):
            label = next(item["label"] for item in self.labeled_intents if item["intent"] == intent)
            examples.append({"text": prompt, "label": label})
        
        return examples
    
    def get_prompt_template(self) -> str: 
        path = ''
        if self.classifier_type == 'few-shot':
            path = pathlib.Path('./data/few_shot_template.jinja2')
        else: # zero shot
            path = pathlib.Path('./data/zero_shot_template.jinja2')
            
        with path.open() as f:
            prompt_template = jinja2.Template(f.read())
            
        return prompt_template
    
    def construct_prompt_from_template(self, text_to_classify: str) -> str:
        prompt_template = self.get_prompt_template()
        prompt = ''
        if self.classifier_type == 'few-shot':
            
            prompt = prompt_template.render(
                examples=self.training_examples,
                labels=self.labeled_intents,
                text=text_to_classify,
            )
        else: # zero-shot prompt
             prompt = prompt_template.render(
                labels=self.labeled_intents,
                text=text_to_classify,
            )

        return prompt
    
    def get_prediction_labels(self, prompt: str) -> str:
    
        completion = openai.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        
        pred_labels = completion.choices[0].message.content
        return pred_labels

    def load(self, model_name: str = 'gpt-3.5-turbo', classifier_type: str = 'few-shot'):
        # Update Attributes
        self.model_name  = model_name
        self.classifier_type = classifier_type
        
        # Read the TSV file into a pandas DataFrame
        train_ds = pd.read_csv(self.train_ds_path, sep='\t', header=None, names= ['user_prompt', 'intents'])

        # Get unique intents list from training DS
        # To add/update intents, modify the dataset at self.train_ds_path with a few examples of the new intents
        # Future Improvement: uncouple the addition/update of intents from the training ds
        unique_intents = train_ds['intents'].unique().tolist()
        
        # Create a list of dicts containing basic information about each class (intent)
        # Returns (for few-shot): [{'label': 0, 'intent': 'flight'}, {'label': 1, 'intent': 'airfare'}, {'label': 2, 'intent': 'ground_service'}]
        # OR (for zero-shot): [{'label': 0, 'intent': 'flight', "description": "This intent is about obtaining flight information"}, {'label': 1, 'intent': 'airfare', "description": "This intent is about obtaining airfare information"}]
        self.labeled_intents = self.create_label_intents(intents= unique_intents)
        
            
        if self.classifier_type == 'few-shot':
            # Transform training data from dataframe to python list of dicts, with user_prompts as keys and intent_ids as values
            # Return example [{'text': 'ground transportation for las vegas', 'label': 0}, {'text': 'kansas city to las vegas economy', 'label': 1}]
            examples = self.create_examples(training_ds=train_ds)
            
            # Pick a representative subset of examples for each class equal to 'max_per_label'
            # We later pass these training_examples as reference to the 'few-shot' classifiers
            self.training_examples = GPTIntentClassifier.pick_training_examples(examples= examples, max_per_label= 3)

    
    def convert_pred_labels_to_intents(self, pred_labels: str) -> list():
        # Convert string to list of integers
        pred_labels = [int(x) for x in pred_labels.strip('[]').split(',')]
        # Map each integer to the corresponding label value
        predicted_intents = [self.labeled_intents[label]["intent"] for label in pred_labels]
        return predicted_intents

    @staticmethod
    def format_server_response(input_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Formats a list of 3 elements into a dictionary containing a list of dictionaries
        suitable for a server response in JSON format.

        Args:
            input_list (List[str]): A list of 3 elements.

        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary containing a list of dictionaries with 'label' keys.

        Example:
            >>> ServerResponseFormatter.format_server_response(['flight', 'flight+airfare', 'city'])
            {'intents': [{'label': 'flight'}, {'label': 'flight+airfare'}, {'label': 'city'}]}
        """
        return {'intents': [{'label': label} for label in input_list]}
    
    def classify_intent(self, user_text: str) -> str:
        # Use the GPT model to classify the intent of the prompt
        prompt = self.construct_prompt_from_template(text_to_classify=user_text)
        print(prompt)
        pred_labels = self.get_prediction_labels(prompt=prompt)
        ## VALIDATE RESPONSE
        pred_intents = self.convert_pred_labels_to_intents(pred_labels = pred_labels)
        # format response 
        formatted_response = GPTIntentClassifier.format_server_response(input_list=pred_intents)
        return formatted_response
    
    
    
    
def main():
    # classifier_name = "GPT"  # or "BERT"
    # model_name = "gpt-3.5-turbo"
    # model_name = "gpsdo"
    # classifier_type = 'zero-shot'
    # train_ds_path = "./data/atis/train.tsv"
    print('in main')
    # Instantiate a concrete subclass of IntentClassifier
    model = GPTIntentClassifier(classifier_type='few-shot')
    print(model.is_ready())
    print(model.classifier_type)
    classifier_type = 'zero-shot'
    model.load(classifier_type = classifier_type)
    print(model.model_name)
    print(model.classifier_type)
    formatted_response = model.classify_intent('I need to rent a car')
    
    print(formatted_response)
    
    

if __name__ == '__main__':
    main()
