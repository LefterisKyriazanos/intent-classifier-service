import os
import argparse
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from gpt_intent_classifier import GPTIntentClassifier
from fastapi.exceptions import RequestValidationError
from typing import Optional



app = FastAPI()

# Instantiate a concrete subclass of IntentClassifier
# See IntentClassifier (parent Abstract Class) for more information
# Inialize GPTIntentClassifier
model = GPTIntentClassifier()


class Item(BaseModel):
    """
    Represents an item with a text attribute.

    Attributes:
        text (Optional[str]): The text associated with the item. It's optional,
            allowing it to be None, but if present, it should be a string.
            The length of the text is constrained to be between 1 and 500 characters.
    """
    text: Optional[str] = Field(None, min_length=1, max_length=500)
    
@app.get('/ready')
def ready():
    """
    Check if the model is ready.

    Returns:
        A JSON response indicating status.
    """
    if model.is_ready():
        return {'status': 'OK'}
    else:
        raise HTTPException(status_code=500, detail= {"label": "INTERNAL_ERROR", "message": "Not Ready"})
    
@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc):
    """
    Exception handler for RequestValidationError.
    
    This method catches RequestValidationError raised during request validation 
    and responds with a 500 Internal Server Error status code instead of the predefined fastapi error for value validation.
    
    See Item class for more information. 

    Parameters:
        request (Request): The request object.
        exc (RequestValidationError): The RequestValidationError instance.

    Returns:
        HTTPException: An HTTPException with a 500 status code and error detail.
    
    Example Requests Handled:
        - Missing required fields in the request body.
        - Incorrect data types for fields.
        - Invalid field values failing validation criteria.
        - Incorrect data format in the request.

    """
    raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(exc)})

@app.post('/intent')
async def intent(query: Item = None):
    """
    Endpoint for classifying the intent of a text message.

    Parameters:
    - query (Item): A Pydantic model representing the request body.
    
    Request body should be a JSON object with a single key 'user_query' containing the text message to classify.
    Valid Payload: {'text': 'show me ground transportation in phoenix'}
    
    Returns:
    - JSONResponse: A JSON response with the predicted intents.

    Possible Error Responses:
    - 400 Bad Request: If the request is missing a body or the body is missing a 'user_query' field.
    - 500 Internal Server Error: If any other exception occurs during processing.

    Example Response:
    ```json
    {
        "intents": [
            {"label": "flight"},
            {"label": "aircraft"},
            {"label": "capacity"}
        ]
    }
    ```

    Example Usage:
    ```python
    import requests

    url = 'http://localhost:8000/intent'
    payload = {'text': 'Your text message here'}
    response = requests.post(url, json=payload)
    print(response.json())
    ```
    """
    try:
        if query is None:
            # Raise a 400 Bad Request error if the request is missing a body
            raise HTTPException(status_code=400, detail={"label": "BODY_MISSING", "message": "Request doesn't have a body."})
        elif query.text == None: # string is None (text not passed)
            # Raise a 400 Bad Request error if the 'text' field is missing value
            raise HTTPException(status_code=400, detail={"label": "TEXT_MISSING", "message": "\"text\" missing from request body."})
        else:
            # intent classification logic / returns Dict
            response = model.classify_intent(query.text)
            # response is valid
            if 'ERROR' not in response.keys():
                # return valid response
                return JSONResponse(content=response)
            else:
                # response failed to adhere to expected format
                raise HTTPException(status_code=500, detail= {"label": "INTERNAL_ERROR", "message": "Invalid Model Response"})
    except HTTPException as http_error:
        # re-raise the caught HTTPException, allowing FastAPI to handle it
        # and generate an appropriate HTTP response (code, message) based on the root exception 
        raise http_error
    except Exception as e:
        # Catch any other exceptions and return 500 Internal error
        raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(e)})

def main():
    # define arguments
    arg_parser = argparse.ArgumentParser()
    
    # select classifier Class (e.g. 'GPT' or 'Bert', not used at the moment
    arg_parser.add_argument('--classifier', type=str, default=os.getenv('classifier', default='GPT'), help='Classifier Class. Default: GPT')
    
    # model arguments 
    arg_parser.add_argument('--model', type=str, default=os.getenv('model', default= 'gpt-3.5-turbo'), help='Model name. Default: gpt-3.5-turbo')
    arg_parser.add_argument('--classifier_type', type=str, default=os.getenv('classifier_type', default='zero-shot'), help='few-shot or zero-shot. Default: zero-shot')
    arg_parser.add_argument('--train_ds_path', type=str, default=os.getenv('train_ds_path', default='./data/atis/train.tsv'), help='Relative path to train tsv')
    arg_parser.add_argument('--test_ds_path', type=str, default=os.getenv('test_ds_path', default='./data/atis/test.tsv'), help='Relative path to test tsv')
    
    # port 
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number. Default: 8080')
    
    # parse arguments
    args = arg_parser.parse_args()
    
    # update model attributes
    model.model_name = args.model
    model.classifier_type = args.classifier_type
    model.train_ds_path = args.train_ds_path
    model.test_ds_path  = args.test_ds_path
    
    # print statements
    print('\nclass (classifier): ', args.classifier)
    print('model_name: ', model.model_name)
    print('model_classifier_type: ', model.classifier_type)
    print('model_train_ds_path: ', model.train_ds_path)
    print('model.test_ds_path: ', model.test_ds_path)
    print('\n')
    
    
    # Prepare and evaluate the model
    # Ensures that the server is only started when the model is successfully loaded
    if model.load(test_size=50):
        print('\nModel loaded successfully!\n')
        print('\nStarting Server..\n')
        # If the model is successfully loaded, run the server
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Server is not started 
        # If loading fails, raise an HTTPException with an appropriate error message
        raise HTTPException(status_code=500, detail="Model did not Load Successfully")

if __name__ == '__main__':
    main()
    
