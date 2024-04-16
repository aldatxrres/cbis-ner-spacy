import medspacy
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
from spacy import displacy
from spacy.matcher import Matcher

nlp = medspacy.load()
print(nlp.pipe_names)

with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_035.txt", encoding="utf-8") as file:
    text = file.read()

colors = {
    "PROBLEM": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "MEDICATION": "linear-gradient(90deg, #9cfc9c, #9cfcfc)",
    "DISEASE": "linear-gradient(90deg, #fc9c9c, #fcfc9c)"
}

matcher = Matcher(nlp.vocab)

# usando o LOWER para que a comparação seja feita em minúsculo

patterns = [
    [{"LOWER": "leucodérmica"}],
    [{"LOWER": "hipertensão"}],
    [{"LOWER": "pneumonia"}],
    [{"LOWER": "diabetes"}, {"LOWER": "mellitus"}, {"LOWER": "tipo"}, {"LOWER": "2"}],
    [{"LOWER": "doença"}, {"LOWER": "cerebrovascular"}],
    [{"LOWER": "febre"}],
    [{"LOWER": "amoxicilina"}],
    [{"LOWER": "clemastina"}],
    [{"LOWER": "hidrocortisona"}],
    [{"LOWER": "dislipidemia"}]
]

for pattern in patterns:
    matcher.add("CUSTOM_ENTITIES", [pattern])

doc = nlp(text)
visualize_ent(doc)
displacy.serve(doc, style='ent', page=True, options={"colors": colors})


