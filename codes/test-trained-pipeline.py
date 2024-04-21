import spacy
from spacy import displacy

colors = {
    "PROCEDIMENTO": "linear-gradient(90deg, #AA9CFC, #FC9CE7)",
    "DOENCA": "linear-gradient(90deg, #FC9C9C, #FCFC9C)", 
    "MEDICAMENTO": "linear-gradient(90deg, #9CFC9C, #9CFCFC)",
    "SINTOMA": "linear-gradient(90deg, #E0551B, #E0A238)",
    "ESPECIALIDADE": "linear-gradient(90deg, #38E0A2, #6DE0DC)",
    "REACAO": "linear-gradient(90deg, #E05138, #E0A76E)",
    "DIAGNOSTICO": "linear-gradient(90deg, #851BE0, #E0806E)",
}

nlp = spacy.load("C:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\trained_pipeline")

with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_039.txt", encoding="utf-8") as file:
    texto = file.read()

doc = nlp(texto)
displacy.serve(doc, style='ent', page=True, options={"colors": colors})