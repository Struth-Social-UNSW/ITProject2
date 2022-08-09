from transformers import pipeline
import json

pipe = pipeline(task='feature-extraction', model='digitalepidemiologylab/covid-twitter-bert-v2')
out = pipe(f"In places with a lot of people, it's a good idea to wear a mask")
print(json.dumps(out, indent=4))