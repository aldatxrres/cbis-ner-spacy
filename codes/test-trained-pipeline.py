import spacy
from spacy import displacy

colors = {
    "PROCEDIMENTO": "linear-gradient(90deg, #AA9CFC, #FC9CE7)",
    "DOENÇA": "linear-gradient(90deg, #FC9C9C, #FCFC9C)", 
    "MEDICAMENTO": "linear-gradient(90deg, #9CFC9C, #9CFCFC)",
    "SINTOMA": "linear-gradient(90deg, #E0551B, #E0A238)",
    "ESPECIALIDADE": "linear-gradient(90deg, #38E0A2, #6DE0DC)",
    "REAÇÃO": "linear-gradient(90deg, #E05138, #E0A76E)",
    "DIAGNÓSTICO": "linear-gradient(90deg, #851BE0, #E0806E)",
}

# Carregar o modelo treinado
nlp = spacy.load("C:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\results")

with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_034.txt", encoding="utf-8") as file:
    texto = file.read()

# Processar o texto com o modelo treinado
doc = nlp(texto)
displacy.serve(doc, style='ent', page=True, options={"colors": colors})

# Iterar sobre as entidades identificadas no texto
# for entidade in doc.ents:
#     print(entidade.text, entidade.label_)