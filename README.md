# PolyToxiQ : A Polymer Toxicity Prediction Tool using PSMILE Strings

## About This Project
        
This application predicts the toxicity level of polymers based on their PSMILES string representation using transfer learning techniques and Tox21 molecular fingerprinting. 
        
### Methodology
- **AutoGluon & Scikit-learn**: We used AutoGluon's TabularPredictor to build a robust machine learning model that classifies polymers into different toxicity levels ( High, Medium, and Low). The model was trained on a carefully curated dataset 
          of Tox 21 datsets (4974) with their known toxicity properties and level of concern(LoC).
        
- **Cosine Similarity**: We calculate the cosine similarity between the PolyBERT Generated fingerprint of the input polymer and those in our reference database of Tox21 Molecule Fingerprints. This metric measures how similar two molecular structures are in their vector space  representation, with values ranging from 0 (completely different) to 1 (identical).
        
- **Zero-Shot Transfer Learning**: Our approach leverages transfer learning principles that allow us to make 
          predictions on novel polymer structures that weren't present in the training data using Transfer learning of pre-trained Autogluon Model of Tox21 Molecule dataset.        

### Toxicity Classification Levels:
        
# Polymers are classified into three concern levels depending on their toxicity properties or Hazard Criteria (0<= Hazard Criteria <= 8):

- **Persistent**, **Bioaccumulative(BIOACCUM)** ,**carcinogenicity(CARCINOGEN)**, **mutagenic(MUTA)**, **reproductive toxicity(REPROTOX)**, **specific target organ toxicity(STOT)**, **Endocrine Disrutive Chemicals(EDC)**, and **aquatic toxicity(AQUATOX)**
        
# Toxicity Classification Levels:
                    
- **High** ☣️☣️☣️☣️☣️: **(4 < Hazard Criteria <=8)** These polymers may pose significant health or environmental risks and require strict handling protocols.
        
- **Medium** ☣️☣️☣️: **(2 < Hazard Criteria <4)** Moderately concerning toxicity that requires proper handling and disposal procedures.
        
- **Low** ☣️: **(0 <= Hazard Criteria <4)** Minimal toxicity concern under normal usage conditions.
        
  ### References & Further Reading
        
   - [CompTOX21 Data Base and Challange](https://comptox.epa.gov/dashboard/chemical-lists/tox21sl)
   - [REACH REGULATION Article 57 (ANNEX XIV) TOXICITY CRITERIA]()
   - [Polymer Fingerprint and PSMILE](https://psmiles.readthedocs.io/en/latest/#what-is-a-psmiles-string)
   - [GitHUb Repository: PSMILES - Fun with P🙂s strings](https://github.com/Ramprasad-Group/psmiles)
   - [List of chemicals with high hazards for categorisation](https://www.industrialchemicals.gov.au/help-and-guides/list-chemicals-high-hazards-categorisation)
   - [Transfer Learning](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00375)
   - [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
   - [RDKit Documentation](https://www.rdkit.org/docs/index.html)
   - [polyBERT SentenceTransformers ](https://kuenneth.uni-bayreuth.de/en/projects/index.html)
   - [SMILES Notation for Chemical Structures](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
   - 
   ![image](https://github.com/user-attachments/assets/2c23e890-fef5-4c7a-ad2a-ba037456ad91)

      
