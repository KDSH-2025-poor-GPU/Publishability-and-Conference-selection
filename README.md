# Instructions for running

## Step 1: Properly placing your credentials

Place your `credentials.json` in the directories `./conference_selection/live_document_indexing/` and `./conference_selection/main/`

Place a `.env` file in the directories `./conference_selection` and `./conference_selection/main/` \
such that the file contains your Gemini API keys like the following:
```dotenv
API_KEY=...
```

## Step 2: Running the solution for Task 1

The notebook `KDSH_Task_1.ipynb` can be run for testing the solution of Task 1.
In the end, the trained model will be stored in `./conference_selection/main/decisionclassifier.joblib`

## Step 3: Running the solution for Task 2

In one terminal, execute:
```bash
cd conference_selection/live_document_indexing/
python app.py
```

**Note**: Please note that this `app.py` must be run before moving to the next step. Please keep this terminal open.

In another terminal, execute:
```bash
cd conference_selection/main/
python main.py
```

The results will be stored in `./conference_selection/main/results.csv`
