# Pollfish - Machine Learning Engineer

We're excited that you want to join the Pollfish team. If you have any questions regarding this task, please don't hesitate to ask.

## Brief Task Description

Your task is to implement a zero or few-shot intent classifier that can be used to provide inferencing service via an HTTP Service. During the classifier development, you should take into consideration the following points: 

  - new set of intents (handled dynamically by jinja2)
  - cost (reduce tokens, reduce calls)
  - token size (remove stopwords, )
  - latency (???)
  - and parsable output (handle output)

The boilerplate for the Service is implemented in file `server.py` and you'll have to implement the API function for inferencing as per the API documentation provided below. The classifier interface has been defined in `intent_classifer.py`. **You can modify/refactor it according to your needs.**

To create a zero or few-shot classifier, we highly recommend utilizing GPT-3.5-turbo through the OpenAI API. By signing up for an account at https://openai.com/, you will receive $18 in credits to develop your classifier.

Please also provide any Jupyter notebook(s) in which you delve into your task, data, and prompt development. **Include some test results to demonstrate the functionality of your classifier**.

## Implementation Notes / Requirements

- ATIS data can be used for developing the classifier. You'll find the data files in `data/atis` directory. Files are TSV files where the first column is the text and the second column is the intent label.
- The given codebase contains one bug (that we know of). You need to find and fix this bug.
- Your classification component should be architected in a way that facilitates the addition/implementation of new intent classifier models.
- Your service needs to adopt the following API Documentation.
- Please provide a _**private**_ GitHub repository where all of your code should be. Please also provide a README.md file with all the instructions, requirements, etc. to run the solution.

## API Documentation
API documentation for intent classification service.

### `POST /intent`
Responds intent classification results for the given query utterance.

#### Request
JSON request with MIME-Type of `application/json` and body:
- **text** `string`: Input sentence intent classification

Example request
```json
{
 "text": "find me a flight that flies from memphis to tacoma"
}
```

#### Responses

JSON response with body:
- **intents** `[Prediction]`: An array of **top 3 intent prediction results**. See `Prediction` type below.

`Prediction` is a JSON object with fields:
- **label** `string`: Intent label name

Example response
```json
{
 "intents": [{
   "label": "flight"
 }, {
   "label": "aircraft"
 }, {
   "label": "capacity"}]
}
```

#### Error codes

All exceptions are JSON responses with HTTP status code other than 2XX, error label and human-readable error message.

##### 400 Body missing

Given when the request is missing a body.
```json
{
 "label": "BODY_MISSING",
 "message": "Request doesn't have a body."
}
```

##### 400 Text missing

Given when the request has a body but the body is missing a text field.
```json
{
 "label": "TEXT_MISSING",
 "message": "\"text\" missing from request body."
}
```

##### 500 Internal error

Given with any other exception. Human readable message includes the exception text.
```json
{
 "label": "INTERNAL_ERROR",
 "message": "<ERROR_MESSAGE>"
}
```

## Evaluation Criteria

The provided solution will be evaluated according to the following criteria:

  - **Scenario fitness:** How does your solution meet the requirements?
  - **Modularity:** Can your code easily be modified? How much effort is needed to add a new kind of ML model to your inference service?
  - **Code readability and comments:** Is your code easily comprehensible and testable?
  - **Bonus:** Any additional creative features: Docker files, architectural diagrams for model or service, Swagger, model performance metrics, etc.
  - **Research:** Can your code easily adapt to a new set of intents? Is your prompt cost-effective? How do you deal with large token sizes? Does your prompt ensure parsable output? How do you handle parsing errors?
  - **Quality assurance:** **Ensure your code is thoroughly tested** to validate its correctness and functionality. Your solution should be robust and able to handle various inputs and edge cases to demonstrate its reliability.