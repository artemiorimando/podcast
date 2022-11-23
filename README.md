# podcast repo

This repo is to demonstrate a two-stage ML web api solution that takes the title and transcript body of a podcast and outputs the top 5 highlights.

## Two-stage ML solution:
1) Okapi Best Matching (BM25) 
2) MSMARCO Distilbert

BM25 is used to rank the sentences in the transcript that represent the entire transcript. A distilbert model is then used to re-rank the BM25 sentences as the most relevant. The concept of re-rank here can be built upon as I only compute the cosine similarity between top 10 BM25 output embeddings with the title embedding. Ideally, I would build a BERT classification model that predicts whether a sentence in the transcript is relevant or not.

The app can be tested by cloning this repo via command line, moving to the cloned directory in local computer, and running the DockerFile.
```
docker build -t podcast -f Dockerfile .

docker run -p 8000:8000 podcast
```

## /predict POST Endpoint
FastAPI is used to build the web app. You can view it in main.py script in the src folder. The endpoint takes in two fields, "title" and "transcript" and outputs two fields, "title" and "summary".

## Example Request Body
Note that the 'transcript' field is not the full body of text - truncated for display here.

```
{"title": "invisible matters of time", "transcript": "This is ninety nine percent, invisible, I'm Roman Mars for the most board. We take time for granted. Maybe we don't have enough of it, but at least we know how it works. At least you know most of the time a lot of what we think about time and and how we keep track of. It is relatively recent and some aspects that we take for granted aren't actually all that universal and today we're going to be talking to a few of my young colleagues for a set of many stories about our evolving relationship with time and to get US started..."}
```

## Example Response Body
```
{
  "title": "invisible matters of time",
  "summary": [
    "At least you know most of the time a lot of what we think about time and and how we keep track of.",
    "I mean when you think about the history of the implementation of one time zone.",
    "It is interesting to me that the time of day actually depended on the ethnicity of who you were asking.",
    "Then I started noticing that the time that they used the numbers they used for time were two hours off that of facing time.",
    "So time is just one example of how these you know intimate parts of weaker culture being suppressed, it's a sign of them, potentially being separatists of being."
  ]
}
```
