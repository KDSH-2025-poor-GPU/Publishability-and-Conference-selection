# Instructions for running

## Step 1: Properly placing your credentials

Place your `credentials.json` in the directories `./conference_selection/live_document_indexing/` and `./conference_selection/main/` (for accessing Google Drive folders)

## Step 2: Running the solution for Task 1

The notebook `KDSH_Task_1.ipynb` can be run for testing the solution of Task 1.
In the end, the trained model will be stored in `./conference_selection/main/decisionclassifier.joblib`

## Step 3: Running the solution for Task 2

Execute the following:
```bash
export API_KEY=...
cd conference_selection/main/
docker-compose up --build
```

(where `API_KEY` is your Google Gemini API Key)

The results will be stored in `/app/results.csv` of the `main_service` container.
