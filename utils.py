import pandas as pd
import os, shutil


def make_submission(preds):
    os.makedirs("submission", exist_ok=True)
    open("submission/original_notebook.ipynb", "w").close()
    df = pd.DataFrame(preds, columns=["label"])
    df.to_csv("submission/submission.csv", index=False)
    shutil.make_archive("submission", "zip", "submission")

    shutil.rmtree("submission")

    print("Submission file generated at submission.zip")
