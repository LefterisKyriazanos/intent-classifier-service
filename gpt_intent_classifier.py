from intent_classifier import IntentClassifier
from dotenv import load_dotenv
import random
import openai
import pandas as pd
from typing import List, Dict, Tuple
import pathlib
import jinja2
import numpy as np
import re
import nltk
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
nltk.download('stopwords')  
nltk.download('punkt')  
from nltk.corpus import stopwords
import string
from fastapi import HTTPException

# load openai creds
load_dotenv()

class GPTIntentClassifier(IntentClassifier):
    """
    Initialize the GPTIntentClassifier.

    Args:
        model_name (str): Name of the GPT model to use. Defaults to "gpt-3.5-turbo".
        classifier_type (str): Type of classifier to use. Options are "zero-shot" and "few-shot".
                                Defaults to 'zero-shot'.
        train_ds_path (str): Path to the training dataset. Defaults to "./data/atis/train.tsv".
        test_ds_path (str): Path to the test dataset. Defaults to "./data/atis/test.tsv".

    Attributes:
        model_name (str): Name of the GPT model being used.
        classifier_type (str): Type of classifier being used.
        train_ds_path (str): Path to the training dataset.
        test_ds_path (str): Path to the test dataset.
        labeled_intents (list): List of labeled intents from the training dataset.
        training_examples (list): List of training examples from the training dataset.
        train_ds (DataFrame): DataFrame containing the training dataset.
        test_ds (DataFrame): DataFrame containing the test dataset.

    Note:
        To use this class, you need to have an OpenAI API key. Update the 'openai_api_key' attribute
        with your API key before using the classifier.
    """

    def __init__(self, model_name = "gpt-3.5-turbo", classifier_type = 'zero-shot', train_ds_path = "./data/atis/train.tsv", test_ds_path= "./data/atis/test.tsv"):
        
        super().__init__()  # Call superclass constructor
        
        # name & classifier
        self.model_name = model_name
        self.classifier_type = classifier_type
        
        # paths to train/test files (must be tsv)
        self.train_ds_path = train_ds_path
        self.test_ds_path = test_ds_path

        # prompt related attributes
        self.labeled_intents = []
        self.training_examples = []

        # datasets will be inialized based on user given train and test paths
        self.train_ds = pd.DataFrame()
        self.test_ds = pd.DataFrame()
        
    def is_ready(self, text: str = 'Athens airport to city center') -> bool:
        """Check if the classifier is ready to classify intents."""
        try:
            # test classifier / returns dict 
            response = self.classify_intent(text=text)
            if 'ERROR' in response.keys():
                return False
            else:
                return True 
        except Exception as e:
            # Pass the exception to FastAPI for handling
            raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(e)})
          
    def create_label_intents(self, intents: List[str]) -> List[Dict[str, str]]:
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
            
            for `zero-shot`:
            
            [{'label': 0, 'intent': 'flight', 'description': 'This intent is about obtaining flight information'},
            {'label': 1, 'intent': 'airfare', 'description': 'This intent is about obtaining airfare information'},
            {'label': 2, 'intent': 'ground_service', 'description': 'This intent is about obtaining ground_service information'}],
            {'label': 3, 'intent': 'city', 'description': 'This intent is about obtaining city information'}]
            
            for `few-shot`:
            
            [{'label': 0, 'intent': 'flight'},
            {'label': 1, 'intent': 'airfare'},
            {'label': 2, 'intent': 'ground_service'}],
            {'label': 3, 'intent': 'city'}]
            
        """
        
        labels = []
        label_counter = 0
        
        if self.classifier_type == 'few-shot':
            for intent in intents:
                # append label
                labels.append({"label": label_counter, "intent": intent})
                label_counter += 1
        else: # prepare labels for zero-shot by adding "description"
            for intent in intents:
                labels.append({"label": label_counter, "intent": intent, "description": f"This intent is about obtaining {intent} information"})
                label_counter += 1
        
        return labels
    
    def pick_training_examples(self, training_examples: List[Dict[str, any]], max_per_label: int = 2):
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
            
        Updates: self.training_examples attribute
        """
        label_count = {}
        
        # Initialize label_count with all possible single labels as keys and values (count) set to 0
        for label in self.labeled_intents:
            label_count[label['label']] = 0
            
        # Shuffle examples to ensure randomness
        random.shuffle(training_examples)

        for example in training_examples:
            # training examples of multiple intents (0.5% of total) are excluded 
            # to avoid uneven correlations between classes
            if len(example["labels"]) > 1: # example has single target label
                continue
            else: 
                 # Check if the maximum limit for the label has been reached
                if label_count[example["labels"][0]] < max_per_label:
                    self.training_examples.append(example)
                    label_count[example["labels"][0]] += 1
            
            # Break the loop if all labels (classes) reached the maximum limit
            if all(label_count[label['label']] >= max_per_label for label in self.labeled_intents):
                break


    def create_examples(self) -> List[Dict]:
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
        training_prompts = self.train_ds['user_prompt'].tolist()
        training_intents = self.train_ds['intents_list']
        for prompt, intent_list in zip(training_prompts, training_intents):
            labels = []
            for intent in intent_list: # iterate over list of intents 
                label = next(item["label"] for item in self.labeled_intents if item["intent"] == intent)
                labels.append(label)
            
            # preprocess training Data
            prompt = GPTIntentClassifier.preprocess_text(prompt)
            # append new example, eg. {'text': 'first flights and fares from pittsburgh to atlanta on a thursday', 'labels': [1, 2]}
            examples.append({"text": prompt, "labels": labels})
           
        return examples
    
    def get_prompt_template(self) -> str: 
        """
        Retrieves the prompt template based on the classifier type.

        Returns:
            str: The prompt template as a string.

        Note:
            It is assumed that the template files are always present and accessible.
        """
        path = ''
        if self.classifier_type == 'few-shot':
            path = pathlib.Path('./data/few_shot_template.jinja2')
        else: # zero shot
            path = pathlib.Path('./data/zero_shot_template.jinja2')
            
        with path.open() as f:
            prompt_template = jinja2.Template(f.read())
            
        return prompt_template
    
    def construct_prompt_from_template(self, text_to_classify: str) -> str:
        """
        Constructs a prompt using a template based on the classifier type and the provided text to classify.

        Args:
            text_to_classify (str): The text to classify.

        Returns:
            str: The constructed prompt.

        Note:
            It is assumed that the template files are always present and accessible.
        """
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
        """
        Gets prediction labels for the provided prompt from the OpenAI API.

        Args:
            prompt (str): The prompt to use for generating prediction labels.

        Returns:
            str: The prediction labels received from the OpenAI API.

        Note:
            This function makes an API request to OpenAI to generate prediction labels based on the provided prompt.
            The expected format of the output is a string representing a list of labels.
            Validation of the output format is performed later.
        """
        
        # Make an API request to OpenAI to generate prediction labels
        completion = openai.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        
        # Extract prediction labels from the completion response
        pred_labels = completion.choices[0].message.content
        # Return the prediction labels
        return pred_labels

    @staticmethod
    def preprocess_text(input_text: str) -> str:
        """
        Preprocesses the input text by stripping and removing special characters.

        Parameters:
        input_text (str): The input text to preprocess.

        Returns:
        str: The preprocessed text.
        """
        
        # strip leading and trailing spaces
        input_text_trimmed = input_text.strip()
        
        # Tokenize the input text
        # tokenized_text = nltk.word_tokenize(input_text_trimmed)
        
        # Remove stopwords and special characters
        # stop_words = set(stopwords.words('english'))
        special_characters = set(string.punctuation)
        preprocessed_text = [word for word in input_text_trimmed if word.lower() not in special_characters]
        
        # Join the words back into sentences
        preprocessed_text = ' '.join(preprocessed_text)
        
        return preprocessed_text
    
    
    def load(self, test_size: int = 40)  -> bool:        
        """
        Load function responsible for initializing the model before running the server.

        The load function is responsible for:
        1. Preprocessing of training and test datasets.
        2. Choosing training data (examples) in case a 'few-shot' classifier is selected.
        3. Creating JSON-like representations of classes (labeled intents).
        4. Choosing the most representative test dataset based on 'test_size' (usually 40-50 records).
        5. Creating evaluation metrics based on the model's performance on the test data.
        6. Deciding status ('SUCCESS', 'FAIL') based on evaluation thresholds.

        Returns:
            bool: True if the model loaded successfully, False otherwise.

        Notes:
            - To add/update intents, modify the dataset at self.train_ds_path with a few examples of the new intents.
            - Future Improvement: Uncouple the addition/update of intents from the training dataset.

        Explanation:
            This function initializes the model by performing several steps:
            1. It preprocesses the training and test datasets, including splitting the 'intents' column into a list of intents.
            2. If the classifier type is 'few-shot', it selects a representative subset of training examples for each class.
            3. It creates JSON-like representations of classes (labeled intents) based on unique intents extracted from the training dataset.
            4. It selects the most representative test dataset based on the specified 'test_size'.
            5. It evaluates the model's performance on the test dataset and computes evaluation metrics such as accuracy, precision, recall, and confusion matrix.
            6. It decides the status of the model ('SUCCESS', 'FAIL') based on evaluation thresholds, ensuring that the model meets specified performance criteria.

            New Intents (Classes) Addition:
            - To add new intents to the model, modify the dataset at self.train_ds_path by adding a few examples of the new intents. 
            - The `load` function automatically preprocesses the updated dataset and incorporates the new intents during model initialization.

            Exception Handling:
            - Exceptions that occur during the loading process are caught and logged within the load function.
            - If loading fails, the function returns False, indicating that the model was not loaded successfully.
            - The caller of the load function can handle the False return value and respond accordingly, such as logging an error or returning an appropriate HTTP response.

        Example Action:
        - The model is considered loaded successfully only if at least 99.5% of responses were successful (valid responses) &
        - The general accuracy of the model on the test dataset is above 80%.
        - If loading is successful, the function returns True.
        - If loading fails, the caller can handle the False return value appropriately.
        
        Raises:
            HTTPException: If an internal server error occurs during loading.

        """
        try:
            
            # Initialize datasets
            self.train_ds = pd.read_csv(self.train_ds_path, sep='\t', header=None, names= ['user_prompt', 'intents'])
            self.test_ds = pd.read_csv(self.test_ds_path, sep='\t', header=None, names= ['user_prompt', 'actual_intents'])
            # Get unique intents list from training DS
            self.train_ds['intents_list'] = self.train_ds['intents'].str.split('+')

            # Explode the list into separate rows & extract all unique classes
            unique_intents = self.train_ds.explode('intents_list')['intents_list'].unique().tolist()
            
            # Create a list of dicts containing basic information about each class (intent)
            # see doc for more information
            self.labeled_intents = self.create_label_intents(intents= unique_intents)
        
            if self.classifier_type == 'few-shot':
                
                # Transform training data from dataframe to python list of dicts, with user_prompts as keys and intent_ids as values
                examples = self.create_examples()
                # Pick a representative subset of examples for each class equal to 'max_per_label'
                self.pick_training_examples(training_examples=examples, max_per_label = 2)
            
            # evaluate before finishing loading
            valid_res, invalid_res, metrics_per_class, avg_evaluation_metrics = self.evaluate(test_size=test_size)
            
            #
            # action based on evaluation metrics
            # ...
            
            # example action
            # load successfully only when at least 95% of responses were successful and general accuracy over 85%
            if valid_res >= ((valid_res+invalid_res) * 0.95):
                if (avg_evaluation_metrics.loc[avg_evaluation_metrics['Metric'] == 'Accuracy', ['Value']].values[0] >= 0.80):              
                    return True
            else: # failed evaluation
                return False
        except Exception as e:
            # Pass the exception to FastAPI for handling
            raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(e)})
    
    def convert_pred_labels_to_intents(self, pred_labels: str) -> List:
        """
        Convert predicted labels to intents using a mapping from integer labels to intent values.

        Parameters:
            pred_labels (str): A string representing predicted labels as integers, separated by commas and enclosed in square brackets.

        Returns:
            list: A list of intents corresponding to the predicted labels.

        Explanation:
            This function takes a string `pred_labels` representing predicted labels as integers, separated by commas,
            and enclosed in square brackets. It first converts the string into a list of integers using list comprehension.
            Then, it maps each integer label to the corresponding intent value using a mapping defined in `self.labeled_intents`.
            Finally, it returns a list of intents corresponding to the predicted labels.

            Example:
                If `pred_labels` is '[0,1,2]', and `self.labeled_intents` is [{'label': 0, 'intent': 'Greeting'}, 
                {'label': 1, 'intent': 'Goodbye'}, {'label': 2, 'intent': 'Question'}], the function will return 
                ['Greeting', 'Goodbye', 'Question'].
        """
        # Remove leading and trailing whitespace, and strip the square brackets
        pred_labels_cleaned = pred_labels.strip().strip('[]')
        # Split the string by comma and strip whitespace from each element
        pred_labels_list = [int(label.strip()) for label in pred_labels_cleaned.split(',')]
        # Map each integer to the corresponding label value
        predicted_intents = [self.labeled_intents[label]["intent"] for label in pred_labels_list]
        return predicted_intents
    
    def validate_response(self, input_string) -> bool:
        """
        Validate if a string resembles a Python-like list of 3 integers and if all integers are valid labels.

        Args:
            input_string (str): The string to validate.

        Returns:
            bool: True if the input string resembles a Python-like list of 3 integers, and all integers are valid labels.
                False otherwise.
        """
        # extract valid (expected) labels
        valid_labels = [item['label'] for item in self.labeled_intents]
        # trim string "  Hello, world!   " -> "Hello, world!"
        trimmed_string = input_string.strip()
        # Define a regular expression pattern to match a Python-like list of 3 integers
        pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"

        # Use the fullmatch function to match the entire input string against the pattern
        match = re.fullmatch(pattern, trimmed_string)

         # Check if the input string matches the pattern
        if match:
            # Extract integers from the matched string
            integers = [int(x) for x in re.findall(r'\d+', trimmed_string)]
            # Check if all integers are valid labels
            if all(label in valid_labels for label in integers):
                return True

        # invalid response 
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
            >>> ServerResponseFormatter.format_server_response(['flight', 'airfare', 'city'])
            {'intents': [{'label': 'flight'}, {'label': 'airfare'}, {'label': 'city'}]}
        """
        return {'intents': [{'label': label} for label in input_list]}
    
    def classify_intent(self, text: str) -> Dict:
        """
        Classify the intent of the input text using the GPT model and return a formatted response.

        Parameters:
            text (str): The input text to classify.

        Returns:
            Dict: A dictionary representing the formatted server response containing the classified intents.
            
            Returns {'error': 'BAD RESPONSE'} if model output was unexpected. This case does NOT raise an exception, 
            as during the evaluation we want to log malformed responses.  

        Raises:
            HTTPException: If an internal server error occurs during processing.

        Example:
            {
                "intents": [
                    {"label": "flight"},
                    {"label": "aircraft"},
                    {"label": "capacity"}
                ]
            }

        Explanation:
            This function preprocesses the input text, classifies its intent using the GPT model, and formats
            the response into a dictionary suitable for a server response. It follows these steps:
            
            1. Preprocesses the input text.
            2. Constructs a prompt from the preprocessed text.
            3. Uses the GPT model to predict the intent labels for the prompt.
            4. Validates the format of the predicted labels.
            5. Converts the predicted labels to actual intent values.
            6. Formats the response into a dictionary with keys representing the intents.

            Input Processing:
            - The input text is preprocessed using the preprocess_text method from the GPTIntentClassifier class.

            Model Inference:
            - The preprocessed text is used to construct a prompt, which is then passed to the GPT model for inference.
            - The model predicts a list of integers representing the intent labels.

            Response Validation:
            - The predicted labels are validated to ensure they are in the expected format (a Python-like list of 3 integers).

            Exception Handling:
            - If the response from the model is malformed or does not match the expected format, an error is logged,
            and an error response is returned.
            - The caller of the classify_intent function can handle the error response appropriately, such as logging
            an error or returning an appropriate HTTP response.
        """
        try: 
            # preprocess input
            processed_text = GPTIntentClassifier.preprocess_text(text)
            # Use the GPT model to classify the intent of the prompt
            prompt = self.construct_prompt_from_template(text_to_classify=text)

            # expecting python-like list response of 3 integers
            pred_labels = self.get_prediction_labels(prompt=prompt)
        
            # validate response's format
            if self.validate_response(pred_labels): # response is in expected format and contains only valid labels
                # process response and convert to actual labels 
                pred_intents = self.convert_pred_labels_to_intents(pred_labels= pred_labels)
                # format response 
                formatted_response = GPTIntentClassifier.format_server_response(input_list=pred_intents)
                return formatted_response
            else: # bad response
                print('bad response: ', pred_labels)
                return {'ERROR': 'BAD RESPONSE'}
            
        except Exception as e:
            # Pass the exception to FastAPI for handling
            raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(e)})
            
    
    @staticmethod
    def extract_predicted_intents(predicted_intents: List[Dict]) -> List[List[str]]:
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
    def save_as_csv(df: pd.DataFrame, file_path: str):
        """
        Save a DF as a csv.

        Parameters:
        df (pd.Dataframe): The file to be saved.
        file_path (str): The file path where the df will be saved.

        Returns:
        None
        """
        # Save DataFrame to file
        df.to_csv(file_path)
        
    def calculate_custom_confusion_matrix(self, actual_intents: List[List[str]], predicted_intents: List[List[str]]) -> pd.DataFrame:
        """
        Calculate a custom confusion matrix. This matrix can provide insight into the underlying relationships between different classes. 

        Given a prediction, all three prediction values, for example ['flight', 'flight_no', 'airport'], are counted for each test intent(s), such as ['flight_time','flight_no'].

        Example:
            For test case `show all flights and fares from denver to san francisco` where:
            
            - actual_intents: ['flight', 'airfare']
            - predicted intents: ['flight', 'flight_no', 'airport']
            
            We count all intents vs all predictions as `confusions` like this: 
            
            - 'flight' -> 'flight'       +1 ('flight' is a correct prediction)
            - 'flight' -> 'flight_no'    +1 ('flight_no' is a false prediction)
            - 'flight' -> 'airfare'     +1  ('airfare' is a correct prediction)

            - 'airfare' -> 'flight'         +1 ('flight' is a correct prediction)
            - 'airfare' -> 'flight_no'      +1 ('flight_no' is a false prediction)
            - 'airfare' -> 'airfare'       +1  ('airfare' is a correct prediction)
            
        This approach enables us to discern not only how frequently certain classes appear together    
        but also to identify any underlying associations the model implicitly recognizes.  
        The final matrix is a n*n, where n is the number of unique classes. 
        Each cell represents how many times the actual label (row) was confused/correlated with the pred label (column)

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
            
            # iterate intents
            for intent in actual_intent: 
                # extract label
                actual_label = next(item['label'] for item in self.labeled_intents if item['intent'] == intent)
        
                # iterate pred labels
                for pred in predicted_intent:
                    # extract label, e.g. 'flight'  -> 2
                    predicted_label = next(item['label'] for item in self.labeled_intents if item['intent'] == pred)
                    # append to matrix if 
                    confusion_matrix[actual_label][predicted_label] += 1
                

        # Convert confusion matrix to DataFrame
        df_confusion_matrix = pd.DataFrame(confusion_matrix, columns=[item['intent'] for item in self.labeled_intents], index=[item['intent'] for item in self.labeled_intents])

        return df_confusion_matrix

    @staticmethod
    def calculate_accuracy(actual_intents: List[List[str]], predicted_intents: List[List[str]]) -> Tuple[float, pd.DataFrame]:
        """
        Calculate the accuracy of the intent classifier.
        Accuracy measures the proportion of correctly classified cases from the total number of objects in the dataset.
        Per prediction we compare 2 lists, the test labels, in e.g. ['flight'] or ['flight', 'airline'] with the model's
        response, in e.g. ['flight', 'city', 'airline']. We count correct predictions if actual_intent is a subset of the 
        prediction.

        Parameters:
        actual_intents (list of lists): A list of lists containing the actual intents for each test case.
        predicted_intents (list of lists): A list of lists where each sublist represents the predicted labels for each test case.

        Returns:
        accuracy: float that represents the average accuracy of the intent classifier.
        incorrect_predictions: pd.dataframe containing the incorrect predictions.  
        """

        matched_intent_count = 0
        mismatched_intent_count = 0
        # List to collect rows for incorrect predictions
        incorrect_predictions_list = []
        
        # Calculate accuracy
        for i in range(len(actual_intents)):
            actual_intent = actual_intents[i]
            predicted_intent = predicted_intents[i]
            print(actual_intent, predicted_intents[i])
            if (all(x in predicted_intent for x in actual_intent)): # actual intents are subset of prediction
                matched_intent_count += 1
            else:
                mismatched_intent_count += 1
                # Collect the incorrect prediction data
                incorrect_predictions_list.append({
                    'Actual Intents': actual_intent,
                    'Predicted Intents': predicted_intent
                })
        
        # Convert the list of incorrect predictions to a DataFrame
        incorrect_predictions = pd.DataFrame(incorrect_predictions_list)


        # Calculate accuracy score
        total_cases = matched_intent_count + mismatched_intent_count
        accuracy = matched_intent_count / total_cases
        
        return accuracy, incorrect_predictions
    
    def sample_evaluation_data(self, test_size: int = 40):
       
        """
        Sample data from the evaluation dataset ensuring each class is represented equally.

        Parameters:
            test_size (int): The desired number of total samples to include in the evaluation dataset.

        Explanation:
            This function aims to create a representative evaluation dataset by sampling from the original test dataset.
            It ensures that each class in the dataset is equally represented in the sampled data (if possible). The function accepts the
            desired 'test_size' parameter, indicating the number of samples to include in the evaluation dataset.

            If the 'test_size' is greater than the number of unique classes in the dataset, the function iterates over each
            unique class. For each class, if the number of samples is less than or equal to 'samples_per_class', all samples
            from that class are included. Otherwise, it randomly selects 'samples_per_class' samples from that class.

            If the resulting 'sample_ds' dataset is less than 'test_size', the function randomly selects the remaining
            samples from other classes.

            If the 'test_size' is less than the number of unique classes, the function chooses one example from each class
            and then samples a subset of classes to create the evaluation dataset.

            Finally, the sampled evaluation dataset is assigned to 'self.test_ds' for further evaluation.
        """
        
        # Initialize an empty DataFrame to store the sampled data
        sample_ds = pd.DataFrame()
        classes_count = self.test_ds['actual_intents'].nunique()
        # Calculate the number of samples per class to ensure equal representation
        samples_per_class = test_size // classes_count
        
        # print statements
        print('\ntest size: ', test_size)
        print('unique eval classes: ', classes_count)
        print('eval samples per class: ', samples_per_class)
        print('any remaining test records will be sampled randomly')
        
        if samples_per_class > 0: # test_size is greater than class count

            # Iterate over each unique class
            for unique_test_class_label in self.test_ds['actual_intents'].unique():
                # Extract samples belonging to the current class
                class_samples = self.test_ds[self.test_ds['actual_intents'] == unique_test_class_label]

                # If the number of samples for this class is less than or equal to samples_per_class, include all class samples
                if len(class_samples) <= samples_per_class:
                    sample_ds = pd.concat([sample_ds, class_samples])
                else:
                    # Randomly select samples  from this class
                    sampled_indices = np.random.choice(class_samples.index, samples_per_class, replace=False)
                    sampled_class_samples = class_samples.loc[sampled_indices]
                    sample_ds = pd.concat([sample_ds, sampled_class_samples])

            # If the total number of samples in sample_ds is less than test_size, continue adding samples from other classes
            remaining_samples = test_size - len(sample_ds)
            if remaining_samples > 0:
                # Exclude already selected indices
                remaining_indices = self.test_ds.index.difference(sample_ds.index)
                # Randomly select remaining samples
                additional_samples = self.test_ds.loc[np.random.choice(remaining_indices, remaining_samples, replace=False)]
                sample_ds = pd.concat([sample_ds, additional_samples])
            
        else: # test len is less than class count
            # choose one example from each class 
            sample_ds = self.test_ds.drop_duplicates(subset=['actual_intents'], keep='first')
            # choose a subset of classes to test, equal to test_size
            sample_ds = sample_ds.sample(test_size)
            
        # assign sampled test set to test_ds
        self.test_ds = sample_ds

    def process_evaluation_dataset(self, test_size: int = 100) -> Tuple[List[str], List[List[str]]]:
        """
        Process the evaluation dataset for intent classification.

        Parameters:
            test_size (int): The number of samples to include in the evaluation dataset. Defaults to 100.

        Preprocessing Steps:
            1. Splitting Actual Intents:
                - Splits the 'actual_intents' column into lists using str.split('+').
                - Creates a new column 'actual_intents_list' containing lists of intents for each user prompt.

            2. Extracting Unique Targets:
                - Extracts unique intents present in the 'actual_intents_list' column.
                - Determines the full range of intents present in the evaluation dataset.

            3. Identifying Unseen Classes:
                - Compares the unique intents with the intents known to the model ('model_known_targets').
                - Identifies any intents present in the evaluation dataset but not known to the model as unseen classes.

            4. Filtering/Log Unseen Classes:
                - Removes data corresponding to unseen classes from the evaluation dataset.
                - Ensures compatibility between the evaluation dataset and the model's known intents.

            5. Adjusting Test Sample:
                - If the processed dataset exceeds the specified 'test_size', samples a subset to match the desired size.
               -  Ensures that the evaluation dataset adequately represents the full range of classes for robust evaluation.

            Operations assign a new value to the 'self.test_ds' attribute.
        """
        # preprocess
        self.test_ds['actual_intents_list'] = self.test_ds['actual_intents'].str.split('+')
    
        # # Explode the list into separate rows
        targets_test = self.test_ds.explode('actual_intents_list')['actual_intents_list'].unique().tolist()
        
        # extract any unseen classes by the model
        model_known_targets = [intent['intent'] for intent in self.labeled_intents]
        model_unknown_targets  = set(targets_test) - set(model_known_targets)
        
        # test contains unseen classes
        if len(model_unknown_targets) > 0:
            # filter out data corresponding to unseen labels
            self.test_ds = self.test_ds[~self.test_ds['actual_intents_list'].apply(lambda x: any(item in model_unknown_targets for item in x))]
            print('model_unknown_targets: ', model_unknown_targets) # log them later
        
        # adjust test_size 
        if self.test_ds.shape[0] >= test_size: # else take max length
            # Balance test dataset ('self.test_ds' attribute)
            self.sample_evaluation_data(test_size=test_size)
        
        # (list) format:  [['find a flight from memphis to tacoma dinner'], ['show flights from burbank to milwaukee for today']]
        queries = self.test_ds['user_prompt'].tolist()
        # Iterate over the Series and convert each element to a list with format:[['flight', 'airfare'], ['capacity']]
        actual_intents = [intent for intent in self.test_ds['actual_intents_list']]
        
        return queries, actual_intents
        
    def predict_test_set(self, queries: List[str]) -> Tuple[List, int, int, List]:
        """
        Get predictions for the given set of queries and count valid vs invalid responses.

        Parameters:
            queries (list): A list of queries to be evaluated.

        Returns:
            tuple: A tuple containing:
                - predicted_intents (list): A list of valid predicted intents.
                - valid (int): The count of valid responses.
                - invalid (int): The count of invalid/malformed/unexpected responses.
                - positions (list): list of positions (indices) where response was invalid
        """
        
        predicted_intents = []
        valid_res = 0
        invalid_res = 0
        positions = []
        
        # Test cases one by one
        for pos, query in enumerate(queries): 
            response = self.classify_intent(query)
            # bad response
            if 'ERROR' in response.keys(): 
                # Count invalid
                invalid_res += 1
                # remember positions of failed responses
                positions.append(pos)
            # Valid response
            else:
                # Approve case for evaluation
                predicted_intents.append(response)
                valid_res += 1 

        return predicted_intents, valid_res, invalid_res, positions

    def evaluate(self, test_size: int = 100) -> Tuple[int, int, pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the performance of the intent classifier based on whether the actual intent is a subset of the predicted intents.

        Parameters:
            test_size (int): The number of queries to use for evaluation. Defaults to 100.

        Returns:
            tuple: A tuple containing the following evaluation metrics and data:
                - valid_res (int): The number of valid responses.
                - invalid_res (int): The number of invalid responses.
                - metrics_per_class_df (pd.dataframe): Evaluation metrics per class
                - average_metrics_df (pd.dataframe): Evaluation metrics (AVG)
        """
        queries, actual_intents = self.process_evaluation_dataset(test_size)
    
        # initialize list to store valid responses
        predicted_intents = []
        
        # get predictions for the dataset
        predicted_intents, valid_res, invalid_res, positions = self.predict_test_set(queries)
        
        # valid & invalid response count
        print('\nvalid,  invalid')
        print(valid_res, invalid_res)


        if valid_res > 0: # we can evaluate 
           
            # Prepare Data
            # Remove items from actual_intents where response was invalid
            if positions:
                for pos in sorted(positions, reverse=True): # delete from back to front, preventing unwanted index shifting
                    del actual_intents[pos]
            
            # Extract the label values
            # eg. extract ['flight', 'airfare', 'ground_service'] for each test query 
            y_pred_intents = GPTIntentClassifier.extract_predicted_intents(predicted_intents=predicted_intents)

            # Fit the MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            y_combined = actual_intents + y_pred_intents
            mlb.fit(y_combined)
            
            # Transform the Labels
            y_true_binarized = mlb.transform(actual_intents)
            y_pred_binarized = mlb.transform(y_pred_intents)

            # Compute metrics for each class
            # use zero_division = 0 to avoid inflating the metrics when a class is never predicted
            precision_per_class = precision_score(y_true_binarized, y_pred_binarized, average=None, zero_division=0)
            recall_per_class = recall_score(y_true_binarized, y_pred_binarized, average=None, zero_division=0)
            f1_per_class = f1_score(y_true_binarized, y_pred_binarized, average=None, zero_division=0)

            # Compute confusion matrix (per class)
            confusion_matrices = multilabel_confusion_matrix(y_true_binarized, y_pred_binarized)
            # Aggregate all class matrices
            overall_confusion_matrix = np.sum(confusion_matrices, axis=0)
            # create a class vs class confusion matrix
            custom_confusion_matrix = self.calculate_custom_confusion_matrix(actual_intents=actual_intents, predicted_intents=y_pred_intents)

            # AVG metrics 
            # calculate avg accuracy and log incorrect predictions (df)
            accuracy_avg, incorrect_predictions  = GPTIntentClassifier.calculate_accuracy(actual_intents=actual_intents, predicted_intents=y_pred_intents)

            # using 'macro' average 
            # calculates the metrics for each class n independently and then averages them
            precision_avg_macro = precision_score(y_true_binarized, y_pred_binarized, average='macro', zero_division=0)
            recall_avg_macro = recall_score(y_true_binarized, y_pred_binarized, average='macro', zero_division=0)
            f1_avg_macro = f1_score(y_true_binarized, y_pred_binarized, average='macro', zero_division=0)

            # using 'micro' average
            # calculates metrics globally by counting the total true positives (TP), false positives (FP), 
            # and false negatives (FN) across all classes.
            precision_avg_micro = precision_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)
            recall_avg_micro = recall_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)
            f1_avg_micro = f1_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)

            # using 'weighted' average
            # like macro averaging, but then it takes a weighted average of these metrics based on the number of true instances
            # (support) for each class
            precision_avg_weighted = precision_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0)
            recall_avg_weighted = recall_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0)
            f1_avg_weighted = f1_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0)

            # Get the class labels
            class_labels = mlb.classes_

            # Print metrics for each class
            for i, label in enumerate(class_labels):
                print(f"\nClass '{label}':")
                print(f"  Precision: {precision_per_class[i]}")
                print(f"  Recall: {recall_per_class[i]}")
                print(f"  F1 Score: {f1_per_class[i]}")
                print(f"  Confusion Matrix:")
                print(confusion_matrices[i])

            # Print average metrics
            print("\nAverage Metrics:")
            print("Overall Confusion Matrix:")
            print(overall_confusion_matrix)
            print(f"  Accuracy: {accuracy_avg}")
            print(f"  (macro) Precision: {precision_avg_macro}")
            print(f"  (micro) Precision: {precision_avg_micro}")
            print(f"  (weighted) Precision: {precision_avg_weighted}")
            print(f"  (macro) Recall: {recall_avg_macro}")
            print(f"  (micro) Recall: {recall_avg_micro}")
            print(f"  (weighted) Recall: {recall_avg_weighted}")
            print(f"  (macro) F1 Score: {f1_avg_macro}")
            print(f"  (micro) F1 Score: {f1_avg_micro}")
            print(f"  (weighted) F1 Score: {f1_avg_weighted}")

            # Create a DataFrame to store the metrics
            metrics_per_class_df = pd.DataFrame({
                'Class': class_labels,
                'Precision': precision_per_class,
                'Recall': recall_per_class,
                'F1 Score': f1_per_class
            })
            # Create DataFrame to store average metrics
            average_metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision (macro)', 'Precision (micro)', 'Precision (weighted)', 
                        'Recall (macro)', 'Recall (micro)', 'Recall (weighted)', 
                        'F1 Score (macro)', 'F1 Score (micro)', 'F1 Score (weighted)'],
                'Value': [accuracy_avg, precision_avg_macro, precision_avg_micro, precision_avg_weighted, 
                        recall_avg_macro, recall_avg_micro, recall_avg_weighted, 
                        f1_avg_macro, f1_avg_micro, f1_avg_weighted]
            })
                        
            # export
            # if running with Docker files will be stored inside the container directory
            GPTIntentClassifier.save_as_csv(custom_confusion_matrix, file_path=f'./model_evaluation/{self.classifier_type}_custom_confusion_matrix.csv')
            GPTIntentClassifier.save_as_csv(metrics_per_class_df, file_path=f'./model_evaluation/{self.classifier_type}_metrics_per_class.csv')
            GPTIntentClassifier.save_as_csv(average_metrics_df, file_path=f'./model_evaluation/{self.classifier_type}_average_metrics.csv')
            GPTIntentClassifier.save_as_csv(incorrect_predictions, file_path=f'./model_evaluation/{self.classifier_type}_incorrect_predictions.csv')
            
        return valid_res, invalid_res, metrics_per_class_df, average_metrics_df
        
    
def main():
    print('in main')
    # Instantiate a concrete subclass of IntentClassifier
    model = GPTIntentClassifier(classifier_type='few-shot')
    print(model.classifier_type)
    # classifier_type = 'zero-shot'
    model.load()
    
    print('model loaded successfuly (main)')
    # print(model.training_examples)
    print(model.model_name)
    # print(model.labeled_intents)
    print(model.classifier_type)
    print('model is ready: ', model.is_ready())
   
   
    
    
    
    

if __name__ == '__main__':
    main()
