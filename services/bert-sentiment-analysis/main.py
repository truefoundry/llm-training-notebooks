
import os
import gradio as gr
import mlfoundry

from mlfoundry.integrations.transformers import HF_MODEL_PATH
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


mlf_client = mlfoundry.get_client()

run = mlf_client.get_run_by_fqn(os.environ.get('RUN_FQN'))
downloaded = run.download_artifact_deprecated(HF_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(downloaded, use_fast=True)

config = AutoConfig.from_pretrained(downloaded)
model = AutoModelForSequenceClassification.from_pretrained(downloaded, config=config)
model = model.eval()


pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def predict(input):
  output = pipe(input)
  return {obj['label']: obj['score'] for obj in output[0]}

iface = gr.Interface(fn=predict, inputs="text", outputs="label", title="Sentiment Analysis")
iface.launch(server_name="0.0.0.0", server_port=8080)
