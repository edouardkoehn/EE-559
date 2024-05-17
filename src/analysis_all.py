from src.utils import ROOT_DIR
import subprocess
import os

PATH_FCM_ANALYSIS = os.path.join(ROOT_DIR, "src", "analysis_fcm.py")
PATH_LLAVA_ANALYSIS = os.path.join(ROOT_DIR, "src", "analysis_llava.py")
PATH_BERT_ANALYSIS = os.path.join(ROOT_DIR, "src", "analysis_bert.py")
subprocess.run(["python", PATH_FCM_ANALYSIS])
subprocess.run(["python", PATH_LLAVA_ANALYSIS])
subprocess.run(["python", PATH_BERT_ANALYSIS])
