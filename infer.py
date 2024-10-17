from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import pandas as pd
import csv

AZURE_SEARCH_SERVICE: str = ""
AZURE_OPENAI_ACCOUNT: str = ""
AZURE_DEPLOYMENT_MODEL: str = ""
AZURE_OPENAI_KEY: str = ""
service_endpoint = AZURE_SEARCH_SERVICE
index_name = ""
key = ""

search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

def CheckAnswer(strComment, strAnswer):

    GROUNDED_PROMPT="""
    You are a friendly assistant that fact checks answers from comments.
    Confirm the answer to the comment with only the informaion provided in the sources.
    fact check the answer ONLY with the facts listed in the sources below.
    If there isn't enough information below, say you don't know.
    Do not generate answers that don't use the sources below.
    Comment: {comment}
    Answer: {answer} 
    Sources:\n{sources}   
    """
    search_results = search_client.search(
        search_text=strComment,
        top=5,
        select="chunk"
    )
    sources_formatted = "\n".join([f'{document["chunk"]}' for document in search_results])

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,  
        api_version="2024-07-01-preview",
        azure_endpoint=AZURE_OPENAI_ACCOUNT
    )

    chat_completion = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": GROUNDED_PROMPT.format(answer=strAnswer, sources=sources_formatted, comment=strComment)
            }
        ],
        temperature=0.7,
        top_p=0.5
    )

    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

def read_csv(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=[0], delimiter=',')
    return df

file_path = './src/rfp.csv'
df = read_csv(file_path)

data = {}
data['Comment'] = ""
data['Answer'] = ""
data['AI Review'] = ""

with open('out.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerow(data.keys())
    for index, row in df.iterrows():
        data['Comment'] = row[0]
        data['Answer'] = row[1]
        data['AI Review'] = CheckAnswer(row[0], row[1])
        writer.writerow(data.values())
        