import spacy
from spacy import displacy

# Carregar o modelo treinado
nlp = spacy.load("C:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\results")


with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_021.txt", encoding="utf-8") as file:
    texto = file.read()

# Processar o texto com o modelo treinado
doc = nlp(texto)

# Iterar sobre as entidades identificadas no texto
for entidade in doc.ents:
    print(entidade.text, entidade.label_)