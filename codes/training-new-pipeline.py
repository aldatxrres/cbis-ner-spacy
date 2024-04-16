import spacy
from spacy.training import Example

train_data = [
    ("Sexo feminino, 73 anos, leucodérmica, com antecedentes pessoais conhecidos de hipertensão arterial, diabetes mellitus tipo 2 insulinotratada, dislipidemia e doença cerebrovascular. Doente negou alergias medicamentosas conhecidas.", {"entities": [(15, 22, "IDADE"), (24, 36, "DOENÇA"), (78, 98, "DOENÇA"), (100, 124, "DOENÇA"), (125, 140, "MEDICAMENTO"), (142, 154, "DOENÇA"), (157, 179, "DOENÇA")]}),
("Recorreu ao serviço de urgência por tosse produtiva com expectoração purulenta e febre (38,1ºC) com 5 dias de evolução.", {"entities": [(36, 41, "SINTOMA"), (56, 78, "SINTOMA"), (81, 86, "SINTOMA")]}),
("Analiticamente com aumento dos parâmetros inflamatórios e, radiologicamente, com condensação do lobo inferior esquerdo, a favorecer o diagnóstico de pneumonia adquirida na comunidade.", {"entities": [(19, 55, "SINTOMA"), (81, 118, "SINTOMA"), (149, 158, "DOENÇA")]}),
("Foi-lhe prescrita amoxicilina/ácido clavulânico, sendo a primeira administração por via endovenosa, no serviço de urgência.", {"entities": [(18, 29, "MEDICAMENTO"), (30, 47, "MEDICAMENTO"), (84, 98, "PROCEDIMENTO")]}),
("Aproximadamente 1 minuto após ingestão do fármaco, apresentou rash cutâneo generalizado e alteração do estado de consciência, com saturação periférica de oxigênio, em ar ambiente, de 67%; pressão arterial 87×50mmHg; e frequência cardíaca de 110bpm.", {"entities": [(42, 49, "MEDICAMENTO"), (62, 87, "REAÇÃO"), (90, 124, "REAÇÃO"), (130, 186, "SINTOMA"), (188, 214, "SINTOMA"), (218, 247, "SINTOMA")]}),
("Foi medicada com clemastina 2mg e hidrocortisona 200mg, com evolução desfavorável para parada cardiorrespiratória, com posterior recuperação de pulso após Suporte Avançado de Vida, necessidade de entubação orotraqueal e ventilação mecânica invasiva.", {"entities": [(17, 31, "MEDICAMENTO"), (34, 54, "MEDICAMENTO"), (87, 113, "REAÇÃO"), (155, 179, "PROCEDIMENTO"), (196, 217, "PROCEDIMENTO"), (220, 248, "PROCEDIMENTO")]}),
("Eletrocardiograma com evidência de supradesnivelamento do segmento ST no território inferior (Figura 1).", {"entities": [(0, 17, "PROCEDIMENTO"), (35, 92, "SINTOMA")]}),
("Realizou coronariografia urgente, que revelou doença aterosclerótica difusa, com ausência de lesões obstrutivas (Figura 2).", {"entities": [(9, 24, "PROCEDIMENTO"), (46, 75, "DOENÇA"), (93, 111, "SINTOMA")]}),
("Verificou-se ainda, na sala de hemodinâmica, a resolução espontânea do supradesnivelamento do segmento ST-T.", {"entities": [(31, 43, "PROCEDIMENTO"), (71, 107, "SINTOMA")]}),
("Analiticamente, apresentava-se com pico de troponina I 2,046μg/L, creatinoquinase (CK) total 647U/L e CK-MB 55U/L.", {"entities": [(35, 64, "SINTOMA"), (66, 99, "SINTOMA"), (102, 113, "SINTOMA")]}),
("Após contato, a família mencionou alergia prévia à penicilina, que a doente desconhecia.", {"entities": [(51, 61, "MEDICAMENTO")]}),
("Doseamento da triptase nas primeiras 6 horas após o choque: 132ng/mL (fortemente positivo).", {"entities": [(14, 22, "MEDICAMENTO")]}),
("Foi admitida provável síndrome de Kounis tipo 2 em contexto de toma de amoxicilina/ácido clavulânico.", {"entities": [(22, 47, "DOENÇA"), (71, 82, "MEDICAMENTO"), (83, 100, "MEDICAMENTO")]}),
("Doente permaneceu 29 horas sob ventilação mecânica, com boa evolução clínica posterior.", {"entities": [(31, 50, "PROCEDIMENTO")]}),
("Teve alta com indicação para evitar antibióticos betalactâmicos e foi referenciada à consulta de imunoalergologia.", {"entities": [(36, 64, "MEDICAMENTO"), (97, 113, "ESPECIALIDADE")]})
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