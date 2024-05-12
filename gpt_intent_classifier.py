from intent_classifier import IntentClassifier
from dotenv import load_dotenv
import random
import openai
import pandas as pd
from typing import List, Dict
import pathlib
import jinja2
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string

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
    def __init__(self, model_name = "gpt-3.5-turbo", classifier_type = 'few-shot', train_ds_path = "./data/atis/train.tsv", test_ds_path= "./data/atis/test.tsv"):
        
        super().__init__()  # Call superclass constructor
        self.model_name = model_name
        self.train_ds_path = train_ds_path
        self.test_ds_path = test_ds_path
        self.classifier_type = classifier_type
        self.labeled_intents = []
        self.training_examples = []
       
        
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
            >>> unique_intents = ['flight', 'airfare', 'ground_service', 'flight+city']
            >>> labels = create_label_intents(unique_intents)
            >>> print(labels)
            [{'label': 0, 'intent': 'flight', 'description': 'This intent is about obtaining flight information'},
            {'label': 1, 'intent': 'airfare', 'description': 'This intent is about obtaining airfare information'},
            {'label': 2, 'intent': 'ground_service', 'description': 'This intent is about obtaining ground_service information'}],
            {'label': 3, 'intent': 'city', 'description': 'This intent is about obtaining city information'}]
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
    
    # @staticmethod
    def pick_training_examples(self, examples: List[Dict[str, any]], max_per_label: int = 10) -> List[Dict[str, any]]:
        """
        Get a subset of examples for each label, limiting to a maximum number of examples per label.
        These Examples are later used as training data for the 'few-shot' classifier. 
        
        Parameters:
            examples (list): A list of examples, where each example is a dictionary containing text and label.
            max_per_label (int): Maximum number of examples per label.

        Returns:
            list: A list of dictionaries, containing a subset of examples for each label.

        Example:
            >>> training_examples = get_few_shot_examples(examples, max_per_label=2)
            >>> print(len(training_examples))
            20  # Assuming there are 10 unique labels with 2 examples each
        """
        training_examples = []
        label_count = {}
        
        # Initialize label_count with all possible single labels as keys and values (count) set to 0
        for label in self.labeled_intents:
            label_count[label['label']] = 0

        # Shuffle examples to ensure randomness
        random.shuffle(examples)

        for example in examples:
            # training examples of multiple intents (0.5% of total) are excluded 
            # to avoid uneven correlations between classes
            if len(example["labels"]) > 1:
                continue
            else: # example has single target label
                 # Check if the maximum limit for the label has been reached
                if label_count[label['label']] < max_per_label:
                    training_examples.append(example)
                    label_count[label['label']] += 1
                
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
            training_ds: pandas DF with 2 (str) columns ['user_prompt', 'intents_list']

        Returns:
            list: A list of dictionaries, where each dictionary contains text and its corresponding label.

        Example:
            >>> training_prompts = ['ground transportation for las vegas', 'first flights and fares from pittsburgh to atlanta on a thursday']
            >>> training_intents = [['ground_service'], ['flight', 'airfare']]
            >>> labels = [{'label': 0, 'intent': 'ground_service'}, {'label': 1, 'intent': 'flight', 'label': 2, 'intent': 'airfare'}]
            >>> examples = create_examples(user_prompts, intents, labels)
            >>> print(examples)
            [{'text': 'ground transportation for las vegas', 'labels': [0]}, {'text': 'first flights and fares from pittsburgh to atlanta on a thursday', 'labels': [1, 2]}]
        """
        examples = []
        training_prompts = training_ds['user_prompt'].tolist()
        training_intents = training_ds['intents_list']
        for prompt, intent_list in zip(training_prompts, training_intents):
            labels = []
            for intent in intent_list: # iterate over list of intents 
                label = next(item["label"] for item in self.labeled_intents if item["intent"] == intent)
                labels.append(label)
            
            # preprocess training Data
            prompt = GPTIntentClassifier.preprocess_input(prompt)
            # append new example, eg. {'text': 'first flights and fares from pittsburgh to atlanta on a thursday', 'labels': [1, 2]}
            examples.append({"text": prompt, "labels": labels})
           
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
        # choose prompt template based on classifier_type
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

    @staticmethod
    def preprocess_input(input_text):
        """
        Preprocesses the input text by tokenizing, removing stopwords, and removing special characters.

        Parameters:
        input_text (str): The input text to preprocess.

        Returns:
        str: The preprocessed text.
        """
        # Tokenize the input text
        tokenized_text = nltk.word_tokenize(input_text)
        
        # Remove stopwords and special characters
        stop_words = set(stopwords.words('english'))
        special_characters = set(string.punctuation)
        preprocessed_text = [word for word in tokenized_text if word.lower() not in stop_words and word not in special_characters]
        
        # Join the words back into sentences
        preprocessed_text = ' '.join(preprocessed_text)
        
        return preprocessed_text
    
    
    def load(self, model_name: str = 'gpt-3.5-turbo', classifier_type: str = 'few-shot'):
        # Update Attributes
        self.model_name  = model_name
        self.classifier_type = classifier_type
        
        # Read the TSV file into a pandas DataFrame
        train_ds = pd.read_csv(self.train_ds_path, sep='\t', header=None, names= ['user_prompt', 'intents'])

        # Get unique intents list from training DS
        # To add/update intents, modify the dataset at self.train_ds_path with a few examples of the new intents
        # Future Improvement: uncouple the addition/update of intents from the training ds
        train_ds['intents_list'] = train_ds['intents'].str.split('+')

        # Explode the list into separate rows
        unique_intents = train_ds.explode('intents_list')['intents_list'].unique().tolist()
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
            self.training_examples = self.pick_training_examples(examples= examples, max_per_label= 3)
        
    
    def convert_pred_labels_to_intents(self, pred_labels: str) -> list():
        # Convert string to list of integers
        pred_labels = [int(x) for x in pred_labels.strip('[]').split(',')]
        # Map each integer to the corresponding label value
        predicted_intents = [self.labeled_intents[label]["intent"] for label in pred_labels]
        return predicted_intents
    
    @staticmethod
    def validate_response(input_string):
        """
        Validate if a string resembles a Python-like list of 3 integers.

        Args:
            input_string (str): The string to validate.

        Returns:
            bool: True if the input string resembles a Python-like list of 3 integers,
                  False otherwise.
        """
        # Define a regular expression pattern to match a Python-like list of 3 integers
        pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"

        # Use the fullmatch function to match the entire input string against the pattern
        match = re.fullmatch(pattern, input_string)

        # If a full match is found, return True
        if match:
            return True
        else:
            return False

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
    
    def classify_intent(self, text: str) -> str:
        
        # preprocess input
        processed_text = GPTIntentClassifier.preprocess_input(text)
        # Use the GPT model to classify the intent of the prompt
        prompt = self.construct_prompt_from_template(text_to_classify=processed_text)
        # print(prompt)
        pred_labels = self.get_prediction_labels(prompt=prompt)
       
        # validate response 
        if GPTIntentClassifier.validate_response(pred_labels): # response is in expected format
            pred_intents = self.convert_pred_labels_to_intents(pred_labels = pred_labels)
            # format response 
            formatted_response = GPTIntentClassifier.format_server_response(input_list=pred_intents)
            return formatted_response
        else: # malformed response
            print('malformed response: ', pred_labels)
            return 'Error: Malformed Response'
            
            
    
    @staticmethod
    def extract_predicted_intents(predicted_intents):
        """
        Extract intent labels from the predicted_intents list.

        Parameters:
        predicted_intents (list): A list of dictionaries where each dictionary contains a list of intents.
        Each dictionary represents the classifier's json response for a test user_query
        
        Example:
        predicted_intents = [{'intents': [{'label': 'city'}, {'label': 'airline'}, {'label': 'flight'}]}, 
                             {'intents': [{'label': 'flight_time'}, {'label': 'distance'}, {'label': 'flight_no'}]}]

        Returns:
        predicted_intent_lists (list of lists): A list of lists where each sublist represents the predicted labels for each test case.
        
        Example:
        predicted_intent_lists = [['city', 'airline', 'flight'], ['flight_time', 'distance', 'flight_no']]
        """

        predicted_intent_lists = []
        for intents_dict in predicted_intents:
            predicted_intent_labels = []
            for intent in intents_dict['intents']:
                predicted_intent_labels.append(intent['label'])
            predicted_intent_lists.append(predicted_intent_labels)

        return predicted_intent_lists
    
    @staticmethod
    def save_as_csv(df, file_path):
        """
        Save a DF as a csv.

        Parameters:
        df (pd.Dataframe): The file to be saved.
        file_path (str): The file path where the df will be saved.

        Returns:
        None
        """
        # Save DataFrame to file
        df.to_csv(file_path, sep='\t')
        
    def calculate_confusion_matrix(self, actual_intents, predicted_intents):
        """
        Calculate the confusion matrix.

        Given a prediction, all three prediction values, for example ['flight', 'flight_no', 'airport'], are counted for each test intent(s), such as ['flight_time','flight_no'].

        Example:
            For test case `show all flights and fares from denver to san francisco` where:
            
            - actual_intents: ['flight', 'airfare']
            - predicted intents: ['flight', 'flight_no', 'airport']
            
            We count all intents vs all predictions as `confusions` like this: 
            
            - 'flight' -> 'flight'       +1
            - 'flight' -> 'flight_no'    +1
            - 'flight' -> 'airport'      +1
            
            - 'airfare' -> 'flight'         +1
            - 'airfare' -> 'flight_no'      +1
            - 'airfare' -> 'airport'        +1

        Parameters:
            actual_intents (list of lists): A list of lists containing the actual intents for each test case.
            predicted_intents (list of lists): A list of lists where each sublist represents the top-3 predicted labels for each test case.

        Returns:
            confusion_matrix (pandas.DataFrame): The confusion matrix.
        """

        # Create a confusion matrix
        num_classes = len(self.labeled_intents)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        
        # Populate confusion matrix
        
        # Iterate predictions
        for i in range(len(actual_intents)):
            actual_intent = actual_intents[i]
            predicted_intent = predicted_intents[i]
            # will hold numeric representations of classes (label)
            # actual_labels  =  []
            # predicted_labels = []
            
            # obtain unique id of each actual label 
            # to populate numpy array
            # e.g. 'city'  -> 0
            for intent in actual_intent: 
                actual_label = next(item['label'] for item in self.labeled_intents if item['intent'] == intent)
        
                # perform the same operation for prediction data
                for pred in predicted_intent:
                    # e.g. 'flight'  -> 2
                    predicted_label = next(item['label'] for item in self.labeled_intents if item['intent'] == pred)

                    # append to matrix
                    confusion_matrix[actual_label][predicted_label] += 1

        # Convert confusion matrix to DataFrame
        df_confusion_matrix = pd.DataFrame(confusion_matrix, columns=[item['intent'] for item in self.labeled_intents], index=[item['intent'] for item in self.labeled_intents])

        return df_confusion_matrix

    def calculate_accuracy(self, actual_intents, predicted_intents):
        """
        Calculate the accuracy of the intent classifier.
        Accuracy measures the proportion of correctly classified cases from the total number of objects in the dataset.
        Per prediction we compare 2 lists, the test labels, in e.g. ['flight'] or ['flight', 'airline'] with the model's
        response, in e.g. ['flight', 'city', 'airline']. For each actual label present in the response we count +1 correct prediction.
        The total number of prediction is calculated as matched_intent_count + mismatched_intent_count.

        Parameters:
        actual_intents (list of lists): A list of lists containing the actual intents for each test case.
        predicted_intents (list of lists): A list of lists where each sublist represents the predicted labels for each test case.

        Returns:
        accuracy_df (pandas.DataFrame): A DataFrame containing the accuracy of the intent classifier.
        """

        matched_intent_count = 0
        mismatched_intent_count = 0

        # Calculate accuracy
        for i in range(len(actual_intents)):
            actual_intent = actual_intents[i]
            predicted_intent = predicted_intents[i]
            print(actual_intent, predicted_intent)
            for intent in actual_intent: # case might have more than 1 intents
                if intent in predicted_intent:  # Actual intent is among the predicted intents
                    matched_intent_count += 1
                else:
                    mismatched_intent_count += 1

        # Calculate accuracy score
        total_cases = matched_intent_count + mismatched_intent_count
        accuracy = matched_intent_count / total_cases
        
        # Create DataFrame with accuracy
        accuracy_df = pd.DataFrame({'accuracy': [accuracy]})

        return accuracy_df
    
    def calculate_precision(self, actual_intents, predicted_intents):
        """
        Calculate precision for each class.
        Precision is calculated as the fraction of instances 
        correctly classified as belonging to a specific class out of all instances 
        the model predicted to belong to that class (TP/(TP+FP)).

        Parameters:
        actual_intents (list): A list containing the actual intents for each test case.
        predicted_intents (list of lists): A list of lists where each sublist represents the predicted labels for each test case.
        labeled_intents (list of dicts): A list of dictionaries containing labeled intents.

        Returns:
        precision_df (pandas.DataFrame): A DataFrame containing precision for each class.
        """

        # Initialize dictionaries to store true positives (TP) and false positives (FP) for each class
        tp_dict = {intent['intent']: 0 for intent in self.labeled_intents}
        fp_dict = {intent['intent']: 0 for intent in self.labeled_intents}

        # Calculate true positives (TP) and false positives (FP) for each class
        for actual_intent, predicted_intent in zip(actual_intents, predicted_intents):
            for intent in predicted_intent:
                if intent in actual_intent:
                    tp_dict[intent] += 1  # True positive
                else:
                    fp_dict[intent] += 1  # False positive

        # Calculate precision for each class
        precision_dict = {}
        for intent in self.labeled_intents:
            tp = tp_dict[intent['intent']]
            fp = fp_dict[intent['intent']]
            precision = tp / (tp + fp) if tp + fp > 0 else 0  # Precision formula
            precision_dict[intent['intent']] = precision

        # Create DataFrame with precision for each class
        precision_df = pd.DataFrame({'intent': [intent['intent'] for intent in self.labeled_intents],
                                     'precision': [precision_dict[intent['intent']] for intent in self.labeled_intents]})
        
        return precision_df
    
    def calculate_recall(self, actual_intents, predicted_intents):
        """
        Calculate recall for each class. 
        Recall is calculated as the fraction of instances in a class that the model correctly classified 
        out of all instances in that class (TP/(TP+FN)).

        Parameters:
        actual_intents (list): A list containing the actual intents for each test case.
        predicted_intents (list of lists): A list of lists where each sublist represents the predicted labels for each test case.
        labeled_intents (list of dicts): A list of dictionaries containing labeled intents.

        Returns:
        recall_df (pandas.DataFrame): A DataFrame containing recall for each class.
        """

     
        # Initialize dictionaries to store true positives (TP) and false negatives (FN) for each class
        tp_dict = {intent['intent']: 0 for intent in self.labeled_intents}
        fn_dict = {intent['intent']: 0 for intent in self.labeled_intents}

        # Calculate true positives (TP) and false negatives (FN) for each class
        for actual_intent, predicted_intent in zip(actual_intents, predicted_intents):
            # for each class
            for intent in self.labeled_intents:
                intent_label = intent['intent']
                # class label is in actual_intent and was predicted
                if intent_label in actual_intent and intent_label in predicted_intent:
                    tp_dict[intent_label] += 1  # True positive
                elif intent_label in actual_intent and intent_label not in predicted_intent:
                    fn_dict[intent_label] += 1  # False negative

        # Calculate recall for each class
        recall_dict = {}
        for intent in self.labeled_intents:
            intent_label = intent['intent']
            tp = tp_dict[intent_label]
            fn = fn_dict[intent_label]
            recall = tp / (tp + fn) if tp + fn > 0 else 0  # Recall formula
            recall_dict[intent_label] = recall

        # Create DataFrame with recall for each class
        recall_df = pd.DataFrame({'intent': [intent['intent'] for intent in self.labeled_intents],
                                  'recall': [recall_dict[intent['intent']] for intent in self.labeled_intents]})

        return recall_df
    
    def process_evaluation_dataset(self, test_size: int = 100):
        
        # read test ds
        eval_ds = pd.read_csv(self.train_ds_path, sep='\t', header=None, names= ['user_prompt', 'actual_intents'])
    
        # preprocess
        eval_ds['actual_intents_list'] = eval_ds['actual_intents'].str.split('+')
    
        # # Explode the list into separate rows
        targets_test = eval_ds.explode('actual_intents_list')['actual_intents_list'].unique().tolist()
        
        # extract any unseen classes by the model
        model_known_targets = [intent['intent'] for intent in self.labeled_intents]
        model_unknown_targets  = set(targets_test) - set(model_known_targets)
        # print(model_unknown_targets)
        
        # test contains unseen classes
        if len(model_unknown_targets) > 0:
            # filter out data corresponding to unseen labels
            eval_ds = eval_ds[~eval_ds['actual_intents_list'].apply(lambda x: any(item in model_unknown_targets for item in x))]
            print('model_unknown_targets: ', model_unknown_targets) # log them later
        
        # adjust test_size 
        if eval_ds.shape[0] >= test_size: # else take max length
            # choose random sample equal to mx length
            eval_ds = eval_ds.sample(test_size)
        
        return eval_ds, model_unknown_targets
        
    def evaluate(self, test_size: int = 100):
        """
        Evaluate the performance of the intent classifier based on whether the actual intent is a subset of the predicted intents.

        Parameters:
        df : 
        Returns:
        
        """
        train_ds, model_unknown_targets =  self.process_evaluation_dataset(test_size)
        
        queries = train_ds['user_prompt'].tolist()
        # Iterate over the Series and convert each element to a list, eg. [['flight', 'airfare'], ['capacity']]
        actual_intents = [intent for intent in train_ds['actual_intents_list']]
        
        # initialize list to store valid responses
        predicted_intents = []
        
        # count valid vs invalid
        valid_res = 0
        invalid_res = 0
        for pos, query in enumerate(queries): 
            print('item: ', pos)
            response = self.classify_intent(query)
            # malformed response
            if response == 'Error: Malformed Response': 
                # remove case from evaluation
                actual_intents.pop(pos)
                invalid_res += 1
            # valid response
            else:
                # approve case for evaluation
                predicted_intents.append(response)
                valid_res += 1 

        # Extracting intent labels from the predicted_intents list
        # eg. extract ['flight', 'airfare', 'ground_service'] for each test query 
        # predicted_intent_lists is a list of lists holding information about all query predictions
        # eg. [['meal', 'flight', 'distance'], ['flight', 'city', 'flight_time']]
        predicted_intent_lists = GPTIntentClassifier.extract_predicted_intents(predicted_intents=predicted_intents)

        # calculate accuracy
        accuracy  = self.calculate_accuracy(actual_intents=actual_intents, predicted_intents=predicted_intent_lists)
        print(accuracy)
        # calculate precision
        precision = self.calculate_precision(actual_intents=actual_intents, predicted_intents=predicted_intent_lists)
        print(precision)
        # calculate recall 
        recall = self.calculate_recall(actual_intents=actual_intents, predicted_intents=predicted_intent_lists)
        print(recall)
        # Create a confusion matrix
        confusion_matrix = self.calculate_confusion_matrix(actual_intents=actual_intents, predicted_intents=predicted_intent_lists)
        
        # save as csv's
        # GPTIntentClassifier.save_as_csv(confusion_matrix, file_path=f'./model_evaluation/{self.classifier_type}_confusion_matrix.csv')
        # GPTIntentClassifier.save_as_csv(accuracy, file_path=f'./model_evaluation/{self.classifier_type}_accuracy.csv')
        # GPTIntentClassifier.save_as_csv(precision, file_path=f'./model_evaluation/{self.classifier_type}_precision.csv')
        # GPTIntentClassifier.save_as_csv(recall, file_path=f'./model_evaluation/{self.classifier_type}_recall.csv')
        
        return valid_res, invalid_res, accuracy, precision, recall, confusion_matrix
    
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
    # print(model.classifier_type)
    # classifier_type = 'zero-shot'
    model.load()
    print(model.model_name)
    # print(model.labeled_intents)
    print(model.classifier_type)
    # formatted_response = model.classify_intent('whats the meal like on flight 444')
    # print(model.labeled_intents)
    # print(formatted_response)
    # raise
    
    
    
    ### EVALUATION 
    # # print(test_ds)
    # test_ds_sample =  test_ds_unique_targets.sample(15)
    # print(test_ds_sample)
    model.evaluate(test_size=50)
    
    
    

if __name__ == '__main__':
    main()
