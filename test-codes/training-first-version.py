import spacy 
import random 
from spacy import displacy
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags


nlp = spacy.blank(); 

nlp = spacy.load("pt_core_news_sm")

ner = nlp.add_pipe("ner-pt")

ner.add_label("DOENÇA")
ner.add_label("MEDICAMENTO")
ner.add_label("SINTOMA")
ner.add_label("PROCEDIMENTO")
ner.add_label("REAÇÃO")
ner.add_label("PROBLEMA")

train_data = [
    ("Sexo feminino, 73 anos, leucodérmica, com antecedentes pessoais conhecidos de hipertensão arterial, diabetes mellitus tipo 2 insulinotratada, dislipidemia e doença cerebrovascular. Doente negou alergias medicamentosas conhecidas. Recorreu ao serviço de urgência por tosse produtiva com expectoração purulenta e febre (38,1ºC) com 5 dias de evolução. Analiticamente com aumento dos parâmetros inflamatórios e, radiologicamente, com condensação do lobo inferior esquerdo, a favorecer o diagnóstico de pneumonia adquirida na comunidade. Foi-lhe prescrita amoxicilina/ácido clavulânico, sendo a primeira administração por via endovenosa, no serviço de urgência. Aproximadamente 1 minuto após ingestão do fármaco, apresentou rash cutâneo generalizado e alteração do estado de consciência, com saturação periférica de oxigênio, em ar ambiente, de 67%; pressão arterial 87×50mmHg; e frequência cardíaca de 110bpm. Foi medicada com clemastina 2mg e hidrocortisona 200mg, com evolução desfavorável para parada cardiorrespiratória, com posterior recuperação de pulso após Suporte Avançado de Vida, necessidade de entubação orotraqueal e ventilação mecânica invasiva. Eletrocardiograma com evidência de supradesnivelamento do segmento ST no território inferior (Figura 1). Realizou coronariografia urgente, que revelou doença aterosclerótica difusa, com ausência de lesões obstrutivas (Figura 2). Verificou-se ainda, na sala de hemodinâmica, a resolução espontânea do supradesnivelamento do segmento ST-T. Analiticamente, apresentava-se com pico de troponina I 2,046μg/L, creatinoquinase (CK) total 647U/L e CK-MB 55U/L. Após contato, a família mencionou alergia prévia à penicilina, que a doente desconhecia. Doseamento da triptase nas primeiras 6 horas após o choque: 132ng/mL (fortemente positivo). Foi admitida provável síndrome de Kounis tipo 2 em contexto de toma de amoxicilina/ácido clavulânico. Doente permaneceu 29 horas sob ventilação mecânica, com boa evolução clínica posterior. Teve alta com indicação para evitar antibióticos betalactâmicos e foi referenciada à consulta de imunoalergologia.", 
     {"entities": [
         (78, 98, "PROBLEMA"),  # hipertensão arterial
         (100, 131, "DOENÇA"),  # diabetes mellitus tipo 2 
         (142, 154, "DOENÇA"),  # dislipidemia
         (157, 179, "DOENÇA"),  # doença cerebrovascular
         (311, 316, "PROBLEMA"),  # febre
         (498, 508, "DOENÇA"),  # pneumonia 
         (552, 563, "MEDICAMENTO"),  # amoxicilina/ácido clavulânico
         (720, 745, "REAÇÃO"),  # rash cutâneo generalizado
         (924, 934, "MEDICAMENTO"),  # clemastina
         (941, 955, "MEDICAMENTO"),  # hidrocortisona
         (1192, 1226, "PROCEDIMENTO"),  # supradesnivelamento do segmento ST
         (1661, 1671, "MEDICAMENTO"),  # penicilina
         (1813, 1838, "REAÇÃO"),  # síndrome de Kounis tipo 2
     ]}
    )
]

print(train_data)

with open("c:\\Users\\aldat\\OneDrive\\Documents\\[TCC] NER\\Projeto spaCy\\cbis-ner-spacy\\clinical-reports\\cc_032.txt", encoding="utf-8") as file:
    text = file.read()

for text, annotations in train_data:
    doc = nlp.make_doc(text)
    entities = annotations.get("entities")
    tags = offsets_to_biluo_tags(doc, entities)
    print("Texto:", text)
    print("Tags BILOU:", tags)

examples = []
for text, annots in train_data:
    examples.append(Example.from_dict(nlp.make_doc(text), annots))

nlp.begin_training()
for _ in range(10): 
    random.shuffle(examples)
    for example in examples:
        nlp.update([example], drop=0.5)

doc = nlp(text)


colors = {
    "PROBLEMA": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "MEDICAMENTO": "linear-gradient(90deg, #9cfc9c, #9cfcfc)",
    "DOENÇA": "linear-gradient(90deg, #fc9c9c, #fcfc9c)",
    "PROCEDIMENTO": "linear-gradient(90deg, #9cfcfc, #9cfc9c)",
    "REAÇÃO": "linear-gradient(90deg, #fc9cfc, #fc9c9c)"
}

displacy.render(doc, style="ent", page=True, options={"colors": colors})