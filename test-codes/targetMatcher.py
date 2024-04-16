import medspacy
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
from spacy import displacy

nlp = medspacy.load()
print(nlp.pipe_names)

with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_010.txt", encoding="utf-8") as file:
    text = file.read()

colors = {
    "PROBLEM": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "MEDICATION": "linear-gradient(90deg, #9cfc9c, #9cfcfc)",
    "DISEASE": "linear-gradient(90deg, #fc9c9c, #fcfc9c)"
}

target_matcher = nlp.get_pipe("medspacy_target_matcher")
target_rules = [
    TargetRule("leucodérmica", "PROBLEM"),
    TargetRule("hipertensão arterial", "PROBLEM"),
    TargetRule("pneumonia", "PROBLEM"),
    TargetRule("diabetes mellitus tipo 2", "DISEASE"),
    TargetRule("doença cerebrovascular", "PROBLEM"),
    TargetRule("expectoração purulenta", "PROBLEM"), 
    TargetRule("febre", "PROBLEM"), 
    TargetRule("amoxicilina", "MEDICATION"), 
    TargetRule("clemastina", "MEDICATION"), 
    TargetRule("hidrocortisona", "MEDICATION"), 
    TargetRule("dislipidemia", "DISEASE")
]
target_matcher.add(target_rules)

doc = nlp(text)
visualize_ent(doc)

displacy.serve(doc, style='ent', page=True, options={"colors": colors})


# Como os problemas foram resolvidos: instalei o IPython, importei o displacy (que resolve o problema de exibição gráfica) 

# Sugestões:

# fazer uma função que reconheça o final das palavras terminadas em (ite - inflamações, itis - inflamações, oma - tumores, ectomia - retirada de órgãos, ectasia - dilatação de órgãos, emia - sangue, lise - destruição, malácia - amolecimento, megalia - aumento, nefro - rim, nefrite - inflamação do rim, nefrose - doença do rim, nefrolitíase - cálculos renais, nefroma - tumor do rim, nefropatia - doença do rim, nefroscopia - exame do rim, nefrostomia - drenagem do rim, nefrotomia - incisão do rim, nefroureterectomia - retirada do rim e ureter e etc...)

# separar medicamentos por classes (anti-inflamatórios, antibióticos, antivirais, antifúngicos, antiparasitários, antineoplásicos, anticoagulantes, anticonvulsivantes, antipsicóticos, antidepressivos, ansiolíticos) 