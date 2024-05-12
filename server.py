import os
import argparse
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from gpt_intent_classifier import GPTIntentClassifier
from intent_classifier import IntentClassifier
from fastapi.exceptions import RequestValidationError
from typing import Optional



app = FastAPI()

classifier_name = "GPT"  # or "BERT"
model_name = "gpt-3.5-turbo"
classifier_type = 'few-shot'
train_ds_path = "./data/atis/train.tsv"
tokenizer_name = '' # optional (for use with a non OpenAI API classifier)
    
# Instantiate a concrete subclass of IntentClassifier
model = GPTIntentClassifier()


class Item(BaseModel):
    # TextWithLength = constr(min_length=1, max_length=50)
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
        raise HTTPException(status_code=423, detail='Not ready')
    
@app.get('/classifier_type')
def get_classifier_type():
    """
    Check the classifier_type (few-shot or zero-shot).

    Returns:
        A JSON response indicating classifier_type.
    """
    if model.is_ready():
        return {'classifier_type': model.classifier_type}
    else:
        raise HTTPException(status_code=423, detail='Not ready')

@app.get('/model_name')
def get_model_name():
    """
    Check the model name.

    Returns:
        A JSON response indicating the model name.
    """
    if model.is_ready():
        return {'model_name': model.model_name}
    else:
        raise HTTPException(status_code=423, detail='Not ready')

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Exception handler for RequestValidationError.

    This method catches RequestValidationError raised during request validation 
    and responds with a 500 Internal Server Error status code instead of the predefined fastapi error for value validation.

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
            # intent classification logic 
            response = model.classify_intent(query.text)
            # response =  'Error: Malformed Response'
            if response != 'Error: Malformed Response':
                return JSONResponse(content=response)
            else:
                raise HTTPException(status_code=500, detail= {"label": "INTERNAL_ERROR", "message": "Model output did not adhere to specified format (python list)"})
    except HTTPException as http_error:
        # re-raise the caught HTTPException, allowing FastAPI to handle it
        # and generate an appropriate HTTP response (code, message) based on the root exception 
        raise http_error
    except Exception as e:
        # Catch any other exceptions and return 500 Internal error
        raise HTTPException(status_code=500, detail={"label": "INTERNAL_ERROR", "message": str(e)})

@app.post("/evaluate")
async def evaluate_model(file):
    #### TEST FILE PREPROCESSING

    
    accuracy, precision, recall = model.evaluate(file)
    return {"accuracy": accuracy, "precision": precision, "recall": recall}

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--classifier_type', type=str, default=os.getenv('classifier_type', 'few-shot'), help='few-shot or zero-shot')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')

    args = arg_parser.parse_args()

    # model configuration
    model.load(model_name = args.model, classifier_type= args.classifier_type)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()
    
