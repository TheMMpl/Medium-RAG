import csv
from vectordb import Memory
import pandas as pd

def prepare_data():
    dataset = pd.read_csv('medium.csv', sep=',', skipinitialspace=True)
    return dataset


#print(df["Text"][2])
def initalize_db(df):
    #
    mem=Memory(memory_file="test.txt",chunking_strategy={"mode": "sliding_window", "window_size": 128, "overlap": 16},embeddings="normal")
    text=df["Text"].to_list()
    print("ziu")
    mem.save(text)
    return mem

def load_db():
    return Memory(memory_file="test.txt",chunking_strategy={"mode": "sliding_window", "window_size": 128, "overlap": 16},embeddings="normal")

#df=prepare_data()
#mem=initalize_db(df)
mem=load_db()
print(mem.search("What is the knn algorithm?"))