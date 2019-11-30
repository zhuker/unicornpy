# pip install spacy
# python -m spacy download en_core_web_sm
import spacy
from spacy import displacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
# Process whole documents
text = (u"""
Agios Pharmaceuticals Receives Funding to Support Research into Novel Therapeutics and Diagnostics for Brain Cancer
USA
Published on December 23, 2009
Agios Pharmaceuticals, a Cambridge, MA-based biopharmaceutical company focused on discovering and developing novel drugs in the emerging field of cancer metabolism, has received an undisclosed amount of funding from Accelerate Brain Cancer Cure (ABC2), a non-profit organization that supports brain cancer research.
These new funds will enable new research investigating IDH1 gene mutations in brain cancer, with the goal of supplementing Agios’s ongoing research into the development of new IDH1 therapeutics and diagnostics.
Recent research by Agios scientists published in the journal Nature established that the mutated form of the enzyme IDH1 produces a metabolite, 2-hydroxyglutarate (2HG), which may contribute to the formation and malignant progression of gliomas, the most common type of brain cancer*. This discovery creates the opportunity for therapeutic intervention in brain cancer and other cancers where IDH1 mutations are present using new drugs that can target the IDH1 metabolic pathway and prevent the buildup of 2HG. 2HG also represents a potential metabolic biomarker that may enable rapid diagnosis and earlier treatment of this form of glioma.
Commenting on the funding, David Schenkein, M.D., Chief Executive Officer at Agios, said: “We are delighted to be collaborating with ABC2 on this important research into the role of the IDH1 mutation in brain cancer.
“ABC2 has a track record of partnering with biotech companies to bring drugs to patients more quickly. Agios is aggressively developing new therapeutics specifically targeting IDH1 which we hope can have a profound impact on the lives of patients with brain cancer”, he added.
*Dang et al. Cancer-associated IDH1 mutations produce 2-hydroxyglutarate. Nature 2009;In press. DOI: 10.1038/nature08617 
""")
doc = nlp(text)
# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
html = spacy.displacy.render(doc, style='ent', minify=True, page=True, options={'compact': True})
with open('render_ent_sm.html', 'w') as f:
    f.write(html)

# spacy.displacy.serve(doc, style='ent')
