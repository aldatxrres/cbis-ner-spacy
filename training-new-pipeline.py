import spacy
from spacy.training import Example

train_data = [
    ("Sexo feminino, 73 anos, leucodérmica, com antecedentes pessoais conhecidos de hipertensão arterial, diabetes mellitus tipo 2 insulinotratada, dislipidemia e doença cerebrovascular.", {"entities": [(24, 36, "DOENCA"), (78, 99, "DOENCA"), (100, 124, "DOENCA"), (142, 154, "DOENCA")]}),
    ("Recorreu ao serviço de urgência por tosse produtiva com expectoração purulenta e febre (38,1ºC) com 5 dias de evolução.", {"entities": [(36, 41, "SINTOMA"), (56, 78, "SINTOMA"), (81, 86, "SINTOMA")]}),
    ("Foi-lhe prescrita amoxicilina/ácido clavulânico, sendo a primeira administração por via endovenosa, no serviço de urgência.", {"entities": [(18, 29, "MEDICAMENTO"), (30, 47, "MEDICAMENTO")]})
]

nlp = spacy.blank("pt")
ner = nlp.add_pipe("ner")

for text, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

optimizer = nlp.begin_training()

for itn in range(100):
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

nlp.to_disk("C:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\results")