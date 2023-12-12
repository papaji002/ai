import openai
openai.api_key = "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"
promt = "what is full form of CPU"
res = openai.Completion.create(engine="davinci", prompt = promt)
genTxt = res.choices[0].text.strip()
print('Result:\n',genTxt)

"""OUTPUT:-
   Result:
   ?
   A full form of CPU is Central Processing Unit which is also commanly
"""