import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components
import requests
import base64
from io import BytesIO
import re
import ast

# Import RDKit for chemical structure rendering and fingerprint generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem # Not directly used in current logic, but for general RDKit ops
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.warning("RDKit is not installed. Install it with: pip install rdkit-pypi")

# Import PSMILES library for polymer processing
try:
    from psmiles import PolymerSmiles as PS
    PSMILES_AVAILABLE = True
except ImportError:
    PSMILES_AVAILABLE = False
    st.warning("PSMILES library is not installed. Install it with: pip install psmiles")

# Import streamlit-ketcher for molecular drawing
try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False
    st.warning("streamlit-ketcher is not installed. Install it with: pip install streamlit-ketcher")

# Import your config
import config

# --- Page Configuration ---
st.set_page_config(
    page_title="PolyToxiQ: Polymer Toxicity Prediction Tool",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Model and Data Loading ---
@st.cache_resource
def load_polybert_model():
    """Loads the polyBERT SentenceTransformer model."""
    try:
        model = SentenceTransformer('kuelumbus/polyBERT')
        return model
    except Exception as e:
        st.error(f"Failed to load polyBERT model: {e}")
        return None

@st.cache_data
def load_training_data():
    """Loads the training data with pre-calculated fingerprints."""
    data_path = config.get_data_path()
    try:
        df = pd.read_csv(data_path)
        # Ensure 'fingerprints' column is parsed as actual lists
        df['fingerprints'] = df['fingerprints'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Rename columns to match expected names in the app
        if 'CANONICALISED_SMILES' in df.columns:
            df.rename(columns={'CANONICALISED_SMILES': 'smiles'}, inplace=True)
        if 'LOC' in df.columns: # Assuming 'LOC' is the toxicity level column
            df.rename(columns={'LOC': 'loc'}, inplace=True)

        return df
    except Exception as e:
        st.error(f"Failed to load training data from {data_path}: {e}")
        return pd.DataFrame()

# Commented out: AutoGluon Model Loading (for future DNN replacement)
# @st.cache_resource
# def load_autogluon_model():
#     """Loads the pre-trained AutoGluon model."""
#     model_path = config.get_model_path()
#     try:
#         from autogluon.tabular import TabularPredictor
#         predictor = TabularPredictor.load(model_path)
#         return predictor
#     except Exception as e:
#         st.error(f"Failed to load AutoGluon model from {model_path}: {e}")
#         return None

polybert_model = load_polybert_model()
training_data_df = load_training_data()
# autogluon_predictor = load_autogluon_model() # Commented out

# --- Helper Functions (Chemical Processing & Visualization) ---

def mol_to_svg(mol, width=300, height=300):
    """Converts an RDKit molecule object to an SVG string."""
    if not RDKIT_AVAILABLE or mol is None:
        return "<span>RDKit or valid molecule not available for rendering.</span>"
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        return f"<span>Error rendering molecule: {e}</span>"

def get_image_download_link(img_data, filename, text):
    """Generates a download link for an image."""
    b64 = base64.b64encode(img_data).decode()
    return f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}">{text}</a>'

def validate_psmiles(psmiles_string):
    """
    Validates a PSMILES string and canonicalizes it using psmiles.
    Returns canonical PS object and RDKit mol, or None and error message.
    """
    if not PSMILES_AVAILABLE:
        return None, None, "PSMILES library not available."
    if not psmiles_string:
        return None, None, "Please enter a PSMILES string."

    # Initial check for presence of two '*'
    if psmiles_string.count('*') != 2:
        return None, None, "PSMILES must contain exactly two '*' characters."
    
    # Try creating a PS object and canonicalizing
    try:
        ps = PS(psmiles_string)
        canonical_ps_obj = ps.canonicalize # This returns a PolymerSmiles object
        
        # Convert the PolymerSmiles object to a string explicitly
        canonical_smiles_str = str(canonical_ps_obj) 
        
        # Validate canonical SMILES with RDKit
        mol = Chem.MolFromSmiles(canonical_smiles_str)
        if mol is None:
            return None, None, "Canonicalized PSMILES resulted in an invalid RDKit molecule. Check structure."
        
        return canonical_ps_obj, mol, None # Return the object and its string representation
    except Exception as e:
        return None, None, f"Error validating or canonicalizing PSMILES: {e}"

def create_dimer(ps_obj):
    """Creates a dimer from a PSMILES object."""
    try:
        if not PSMILES_AVAILABLE:
            return None, None, "PSMILES library not available."

        dimer_ps_obj = ps_obj.dimer(0) # This returns a PolymerSmiles object
        dimer_smiles_str = str(dimer_ps_obj) # Convert the PolymerSmiles object to a string explicitly
        dimer_mol = Chem.MolFromSmiles(dimer_smiles_str)
        return dimer_ps_obj, dimer_mol, None
    except Exception as e:
        return None, None, f"Error creating dimer: {e}"

def generate_fingerprint(smiles_list, model):
    """Generates polyBERT fingerprints for a list of SMILES strings."""
    if model is None:
        return None, "PolyBERT model not loaded."
    try:
        embeddings = model.encode(smiles_list)
        return embeddings.tolist(), None
    except Exception as e:
        return None, f"Error generating fingerprints: {e}"

def subtract_fingerprints(fp_dimer, fp_polymer):
    """Subtracts polymer fingerprint from dimer fingerprint to get monomer fingerprint."""
    if fp_dimer is None or fp_polymer is None:
        return None, "Fingerprints are missing for subtraction."
    try:
        # Ensure they are numpy arrays for element-wise subtraction
        fp_dimer_arr = np.array(fp_dimer)
        fp_polymer_arr = np.array(fp_polymer)
        monomer_fp = (fp_dimer_arr - fp_polymer_arr).tolist()
        return monomer_fp, None
    except Exception as e:
        return None, f"Error during fingerprint subtraction: {e}"

def find_best_match(query_fp, training_df):
    """Finds the best matching molecule in the training data using cosine similarity."""
    if query_fp is None or training_df.empty:
        return None, "Training data not loaded or query fingerprint missing."
    
    # Ensure query_fp is a 2D array for sklearn's cosine_similarity
    if isinstance(query_fp, list):
        query_fp = np.array(query_fp).reshape(1, -1)
    else:
        return None, "Invalid query fingerprint format."

    best_match = None
    max_similarity = -1

    for index, row in training_df.iterrows():
        try:
            # Ensure ref_fp is a 2D array
            ref_fp = np.array(row['fingerprints']).reshape(1, -1)
            similarity = cosine_similarity(query_fp, ref_fp)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = row.to_dict()
                best_match['similarity_score'] = similarity
        except Exception as e:
            # Log error or skip row if fingerprint is malformed
            # st.error(f"Error processing row {index} in training data: {e}. Skipping.") # Comment out for cleaner UI
            continue
            
    return best_match, None

# Commented out: AutoGluon Prediction Function (for future DNN replacement)
# def predict_toxicity_ml(fingerprint_list, predictor):
#     """Predicts toxicity level using the AutoGluon model."""
#     if predictor is None:
#         return "N/A", "AutoGluon model not loaded."
#     if fingerprint_list is None:
#         return "N/A", "Fingerprint is missing for ML prediction."
#
#     try:
#         # AutoGluon expects a DataFrame as input
#         fp_df = pd.DataFrame([fingerprint_list], columns=[f'fp_{i}' for i in range(len(fingerprint_list))])
#         prediction = predictor.predict(fp_df)
#         return prediction.iloc[0], None
#     except Exception as e:
#         return "Error", f"Error during ML prediction: {e}"

def get_toxicity_level_info(loc):
    """Provides detailed info for each toxicity level."""
    if loc is None:
        loc = "unknown"

    loc = str(loc).lower()
    info = {
        "high": {
            "label": "High Toxicity ☣️ ☣️ ☣️ ☣️ ☣️",
            "color": "#FF4B4B",  # Red
            "description": "As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as Highly concenerned Chemicals. This polymer may pose significant health or environmental risks. Proper handling protocols and protective measures are strongly recommended.",
            "recommendations": "⚠️ Always follow safety guidelines when working with this material."
        },
        "medium": {
            "label": "Medium Toxicity ⚠️☣️ ☣️ ☣️",
            "color": "#FFD700", # Gold/Yellow
            "description": "As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as Moderately concern  Chemicals. This polymer has moderate toxicity concerns. It requires proper handling and disposal procedures.",
            "recommendations": "⚠️ Basic protective equipment is recommended when working with this material."
        },
        "low": {
            "label": "Low Toxicity ☣️",
            "color": "#5cb85c",  # Green
            "description": "As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as low concern Chemicals. This polymer has minimal toxicity concerns under normal usage conditions.",
            "recommendations": "⚠️ Standard laboratory safety practices are recommended as a precaution."
        },
        "unknown": {
            "label": "Toxicity Unknown ❓",
            "color": "#808080", # Grey
            "description": "Toxicity information could not be determined or is not available for the best matching reference.",
            "recommendations": "Proceed with caution. Treat as potentially hazardous until further information is available. Conduct further research or testing."
        }
    }
    return info.get(loc, info["unknown"])

def get_reach_hazard_properties(loc):
    """Provides dummy REACH hazard property info based on LOC."""
    properties = {
        "Carcinogenicity": "Not Fulfilled",
        "Mutagenicity": "Not Fulfilled",
        "Reproductive Toxicity": "Not Fulfilled",
        "PBT (Persistent, Bioaccumulative, Toxic)": "Not Fulfilled",
        "vPvB (very Persistent, very Bioaccumulative)": "Not Fulfilled",
        "Endocrine Disrupting Properties": "Not Fulfilled",
        "Respiratory Sensitisation": "Not Fulfilled",
    }
    loc = str(loc).lower()
    if loc == "high":
        properties["Carcinogenicity"] = "Under_investigation"
        properties["Mutagenicity"] = "Under_investigation"
        properties["Reproductive Toxicity"] = "Under_investigation"
    elif loc == "medium":
        properties["Respiratory Sensitisation"] = "Under_investigation"

    return properties

def create_confidence_meter(confidence_score):
    """Creates a Plotly gauge chart for confidence."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score (%)"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence_score}}))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_similarity_visualization(monomer_fp, best_match_fp):
    """
    Placeholder for similarity visualization (e.g., t-SNE, UMAP).
    For now, a simple bar chart comparing first few dimensions.
    """
    if monomer_fp is None or best_match_fp is None:
        st.info("No fingerprints available for visualization.")
        return

    num_dims = min(len(monomer_fp), len(best_match_fp), 20) # Show first 20 dimensions
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Dim {i+1}' for i in range(num_dims)],
        y=monomer_fp[:num_dims],
        name='Query Monomer FP',
        marker_color='skyblue'
    ))
    fig.add_trace(go.Bar(
        x=[f'Dim {i+1}' for i in range(num_dims)],
        y=best_match_fp[:num_dims],
        name='Best Match FP',
        marker_color='salmon'
    ))
    fig.update_layout(
        title='Comparison of Monomer and Best Match Fingerprints (First 20 Dims)',
        xaxis_title='Fingerprint Dimension',
        yaxis_title='Value',
        barmode='group',
        height=400
    )
    return fig

def get_table_download_link(df, filename="report.csv"):
    """Generates a link to download a dataframe as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Full Analysis Report (CSV)</a>'


# --- Main Streamlit Application ---
def main():
    st.title("PolyToxiQ: A Polymer Toxicity Prediction Tool ☣️")

    # Initialize session state for analysis results
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}
    if 'similarity_results' not in st.session_state:
        st.session_state.similarity_results = None

    tab1, tab2, tab3 = st.tabs(["🎨 Visualization", "🔍 Similarity Analysis", "☣️ Toxicity Class Prediction"])

    with tab1:
        st.header("Draw Your Polymer & Visualize Structures")

        # Option to choose input method
        input_method = st.radio(
            "Choose PSMILES Input Method:",
            ("Draw with Ketcher", "Enter PSMILES Manually"),
            key="input_method_selector"
        )

        psmiles_input_value = "" # This will hold the actual value used for analysis

        col_ketcher, col_examples = st.columns([0.7, 0.3])

        # with col_examples:
        #     st.subheader("Polymer Examples")
        #     example_polymers = {
        #         "Polyethylene": "[*]CC[*]",
        #         "Polystyrene": "C(c1ccccc1)(*)C(*)",
        #         "Poly(methyl methacrylate)": "CC(C(=O)OC)(*)C(*)",
        #         "Poly(vinyl chloride)": "C(Cl)(*)C(*)",
        #         "Poly(ethylene glycol)": "[*]OCCO[*]"
        #     }
        #     # Add an empty option
        #     selected_example = st.selectbox(
        #         "Select an example polymer (this will pre-fill the input below):",
        #         [""] + list(example_polymers.keys()),
        #         key="example_select"
        #     )

        #     if selected_example and selected_example != st.session_state.get('last_selected_example_psmiles', ''):
        #         st.session_state.example_psmiles_for_input = example_polymers[selected_example]
        #         st.session_state.last_selected_example_psmiles = selected_example
        #         # Trigger a rerun so the input fields update immediately
        #         st.rerun()
        #     elif not selected_example: # If user clears selection, clear example_psmiles_for_input
        #          if 'example_psmiles_for_input' in st.session_state:
        #              del st.session_state.example_psmiles_for_input
        #              st.session_state.last_selected_example_psmiles = '' # Reset tracking
        #              st.rerun() # Rerun to clear the input field

        with col_ketcher:
            if input_method == "Draw with Ketcher":
                st.write("Draw your polymer structure with two `*` symbols to define the repeating unit.")
                # Pre-fill Ketcher if an example was selected, otherwise use its current state
                ketcher_default = st.session_state.get('example_psmiles_for_input', '')
                if 'ketcher_editor' in st.session_state and st.session_state.ketcher_editor and not ketcher_default:
                     ketcher_default = st.session_state.ketcher_editor # Preserve user's Ketcher drawing if no new example

                psmiles_input_value = st_ketcher(
                    ketcher_default,
                    key="ketcher_editor",
                    height=450
                )
            else: # Enter PSMILES Manually
                st.write("Copy and paste your PSMILES string below. Ensure it contains exactly two `*` symbols.")
                # Pre-fill text area if an example was selected, otherwise use its current state
                manual_default = st.session_state.get('example_psmiles_for_input', '')
                if 'manual_psmiles_text_area' in st.session_state and st.session_state.manual_psmiles_text_area and not manual_default:
                    manual_default = st.session_state.manual_psmiles_text_area # Preserve user's manual input if no new example

                psmiles_input_value = st.text_area(
                    "Enter PSMILES:",
                    value=manual_default,
                    height=150,
                    key="manual_psmiles_text_area"
                )

        st.markdown(f"**Current PSMILES Input:** `{psmiles_input_value}`" if psmiles_input_value else "**Waiting for PSMILES input...**")

        st.markdown("---")

        if st.button("Analyze Polymer"):
            st.session_state.analysis_data = {} # Reset previous analysis
            st.session_state.similarity_results = None
            
            # Use psmiles_input_value for analysis
            if psmiles_input_value: # Only proceed if there's an input
                with st.spinner("Processing Polymer... This may take a moment."):
                    # 1. Validate and Canonicalize PSMILES
                    canonical_ps_obj, canonical_mol, validation_error = validate_psmiles(psmiles_input_value)
                    if validation_error:
                        st.error(f"PSMILES Validation Error: {validation_error}")
                        st.session_state.analysis_data['status'] = 'failed'
                        st.stop()

                    canonical_smiles_str = str(canonical_ps_obj) # Get string representation
                    st.session_state.analysis_data['input_psmiles'] = psmiles_input_value # Store the actual input
                    st.session_state.analysis_data['canonical_psmiles'] = canonical_smiles_str

                    # 2. Create Dimer
                    dimer_ps_obj, dimer_mol, dimer_error = create_dimer(canonical_ps_obj) # Pass the PolymerSmiles object
                    if dimer_error:
                        st.error(f"Dimer Creation Error: {dimer_error}")
                        st.session_state.analysis_data['status'] = 'failed'
                        st.stop()
                    dimer_smiles_str = str(dimer_ps_obj) # Get string representation
                    st.session_state.analysis_data['dimer_psmiles'] = dimer_smiles_str

                    # 3. Generate Fingerprints
                    polymer_fp, poly_fp_error = generate_fingerprint([canonical_smiles_str], polybert_model)
                    dimer_fp, dimer_fp_error = generate_fingerprint([dimer_smiles_str], polybert_model)

                    if poly_fp_error or dimer_fp_error:
                        st.error(f"Fingerprint Generation Error: {poly_fp_error or dimer_fp_error}")
                        st.session_state.analysis_data['status'] = 'failed'
                        st.stop()
                    
                    st.session_state.analysis_data['polymer_fp'] = polymer_fp[0] # Get the list out of the outer list
                    st.session_state.analysis_data['dimer_fp'] = dimer_fp[0]

                    # 4. Subtract Fingerprints to get Monomer FP
                    monomer_fp, subtract_error = subtract_fingerprints(
                        st.session_state.analysis_data['dimer_fp'],
                        st.session_state.analysis_data['polymer_fp']
                    )
                    if subtract_error:
                        st.error(f"Fingerprint Subtraction Error: {subtract_error}")
                        st.session_state.analysis_data['status'] = 'failed'
                        st.stop()
                    st.session_state.analysis_data['monomer_fp'] = monomer_fp

                    # 5. Find Best Match in Training Data
                    if not training_data_df.empty:
                        best_match_result, match_error = find_best_match(monomer_fp, training_data_df)
                        if match_error:
                            st.error(f"Similarity Matching Error: {match_error}")
                            st.session_state.analysis_data['status'] = 'failed'
                            st.stop()
                        st.session_state.similarity_results = best_match_result
                    else:
                        st.warning("Training data not loaded. Cannot perform similarity analysis.")
                        st.session_state.similarity_results = None

                    st.session_state.analysis_data['status'] = 'completed'
                    st.success("Analysis Complete! Proceed to 'Similarity Analysis' and 'Toxicity Class Prediction' tabs.")
            else:
                st.warning("Please provide a PSMILES string to analyze.")


            if st.session_state.analysis_data.get('status') == 'completed':
                st.subheader("Structural Visualizations")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Canonicalized Polymer Structure")
                    st.image(mol_to_svg(Chem.MolFromSmiles(st.session_state.analysis_data['canonical_psmiles'])), use_column_width=True)
                    st.markdown(f"**Canonical PSMILES:** `{st.session_state.analysis_data['canonical_psmiles']}`")
                
                with col2:
                    st.markdown("#### Generated Dimer Structure")
                    st.image(mol_to_svg(Chem.MolFromSmiles(st.session_state.analysis_data['dimer_psmiles'])), use_column_width=True)
                    st.markdown(f"**Dimer PSMILES:** `{st.session_state.analysis_data['dimer_psmiles']}`")

                st.markdown("---")
                with st.expander("🔬 Fingerprint Information (600 Dimensions)", expanded=False):
                    st.write("These are the generated polyBERT fingerprints (first 10 dimensions shown).")
                    
                    st.subheader("Polymer Fingerprint")
                    st.json(st.session_state.analysis_data['polymer_fp'][:10]) # Show first 10
                    
                    st.subheader("Dimer Fingerprint")
                    st.json(st.session_state.analysis_data['dimer_fp'][:10]) # Show first 10
                    
                    st.subheader("Derived Monomer Fingerprint (Polymer - Dimer)")
                    st.json(st.session_state.analysis_data['monomer_fp'][:10]) # Show first 10
            
        # --- NEW SECTIONS AT THE END OF TAB 1 ---
        # Project background information
        with st.expander("📚 Project Background & Information", expanded=False):
            st.markdown("""
            ## About This Project
            
            This application predicts the toxicity level of polymers based on their PSMILES string representation using 
            transfer learning techniques and Tox21 molecular fingerprinting. 
            
            ### Methodology
            - **AutoGluon & Scikit-learn**: We used AutoGluon's TabularPredictor to build a robust machine learning model 
              that classifies polymers into different toxicity levels (High, Medium and Low). The model was trained on a carefully curated dataset 
              of Tox 21 datasets (4974) with their known toxicity properties and level of concern (LoC).
            #### Fingerprint Generation
            
            - **PolyBERT Embeddings**: We use the polyBERT sentence transformer to generate vector representations of polymer structures.
            - **Vector Similarity**: Cosine similarity measures how similar two molecular structures are in their vector space representation.
            - **Advanced Processing**: PSMILES strings undergo validity checking, canonicalization, and attachment point handling.
            """)

        # Add Polymer Structure Examples (descriptive)
        with st.expander("🧪 Polymer Structure Examples", expanded=False):
            st.markdown("""
            ### Polymer SMILES Notation
            
            In polymer SMILES (PSMILES), the `[*]` symbol represents connection points for repeating units in the polymer chain.
            Below are examples of common polymer structures and how they're represented in PSMILES notation.
            """)
            st.markdown("""
            For more information on PSMILES notation, see the [PSMILES documentation](https://psmiles.readthedocs.io/en/latest/).
            """)

        # Add about Ketcher information
        with st.expander("🎨 About Ketcher", expanded=False):
            st.markdown("""
            ## Ketcher Chemical Structure Editor
            
            Ketcher is an open-source web-based chemical structure editor incorporating high performance, good portability, 
            light weight, and ability to easily integrate into a custom web-application. It is designed for chemists, 
            laboratory scientists and technicians who draw structures and reactions.
            
            ### Features
            - Drawing and editing chemical structures and reactions
            - Loading and saving structures in various formats (SMILES, Molfile, etc.)
            - Clean layout algorithm for structure visualization
            - Stereochemistry support
            
            ### Try Ketcher Online
            You can experiment with Ketcher's full capabilities at [Ketcher Official Demo](https://lifescience.opensource.epam.com/ketcher/).
            
            For more information, visit the [Ketcher GitHub Repository](https://github.com/epam/ketcher).
            """)

    with tab2:
        st.header("Similarity Analysis")
        if st.session_state.similarity_results:
            best_match = st.session_state.similarity_results
            similarity_score = best_match['similarity_score']
            confidence_percentage = round(similarity_score * 100, 2)
            
            st.markdown(f"""
                <div style='background-color:#e0f2f7; padding:15px; border-radius:5px;'>
                    <h3>Overall Similarity Match</h3>
                    <p style='font-size:20px;'>Cosine Similarity Score: <strong>{similarity_score:.4f}</strong></p>
                    <p style='font-size:20px;'>Confidence Percentage: <strong>{confidence_percentage:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

            # --- NEW: Cosine Similarity explanation in Tab 2 ---
            st.markdown("""
            - **Cosine Similarity**: We calculate the cosine similarity between the PolyBERT Generated fingerprint of the input polymer and those 
              in our reference database of Tox21 Molecule Fingerprints. This metric measures how similar two molecular structures are in their vector space 
              representation, with values ranging from 0 (completely different) to 1 (identical).
            """)
            # --- END NEW ---

            if confidence_percentage < 90:
                st.warning("⚠️ **Caution:** The similarity score is below 90%. Toxicity predictions may vary and should be interpreted with care.")
            else:
                st.success("✅ **High Confidence Match:** Similarity score is 90% or above, indicating a strong structural match.")
            
            st.markdown("---")

            st.subheader("Confidence Meter")
            st.plotly_chart(create_confidence_meter(confidence_percentage), use_container_width=True)

            st.subheader("Best Matching Compound from Training Data")
            col_match_struct, col_match_details = st.columns(2)

            with col_match_struct:
                st.markdown("#### Reference Compound Structure")
                ref_mol = Chem.MolFromSmiles(best_match['smiles'])
                if ref_mol:
                    st.image(mol_to_svg(ref_mol), use_column_width=True)
                else:
                    st.write("Could not render reference structure.")

            with col_match_details:
                st.markdown("#### Reference Compound Details")
                st.write(f"**Reference SMILES:** `{best_match['smiles']}`")
                st.write(f"**Predicted Toxicity Level (from Reference):** `{best_match['loc'].upper()}`")
                st.write(f"**Reference Database Index:** `{best_match.get('index', 'N/A')}`") # Assuming 'index' might exist

            st.markdown("---")
            st.subheader("Fingerprint Comparison Visualization")
            if st.session_state.analysis_data.get('monomer_fp') and best_match.get('fingerprints'):
                st.plotly_chart(create_similarity_visualization(st.session_state.analysis_data['monomer_fp'], best_match['fingerprints']), use_container_width=True)
            else:
                st.info("Fingerprints not available for visualization.")

        else:
            st.info("Please process a polymer in the 'Visualization' tab first to perform similarity analysis.")

    with tab3:
        st.header("Toxicity Class Prediction & Hazard Information")
        if st.session_state.similarity_results:
            best_match_loc = st.session_state.similarity_results['loc']
            toxicity_info = get_toxicity_level_info(best_match_loc)
            
            # Improved readability with dynamic text color based on background
            text_color = "black" if toxicity_info['color'] in ["#FFD700", "#5cb85c", "#e0f2f7"] else "white"
            st.markdown(f"""
                <div style='background-color:{toxicity_info['color']}; padding:20px; border-radius:10px; color:{text_color};'>
                    <h2 style='text-align:center; color:{text_color};'>Predicted Toxicity Level: {toxicity_info['label']}</h2>
                    <p style='font-size:18px; text-align:center; color:{text_color};'>{toxicity_info['description']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            st.subheader("Quick Stats")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                st.metric("Predicted Level", toxicity_info['label'].split(' ')[0])
            with col_q2:
                st.metric("Confidence", f"{st.session_state.similarity_results['similarity_score'] * 100:.2f}%")
            with col_q3:
                st.metric("Reference Match", st.session_state.similarity_results['smiles'])

            st.markdown("---")
            st.subheader("REACH 2017 Regulation (Article 57 (ANNEX XIV)) Hazard Properties")
            st.info("This section provides a preliminary indication. A full REACH assessment requires comprehensive data beyond similarity matching.")
            
            hazard_props = get_reach_hazard_properties(best_match_loc)
            col_h1, col_h2, col_h3 = st.columns(3)
            props_list = list(hazard_props.items())
            for i, (prop, status) in enumerate(props_list):
                if i % 3 == 0:
                    with col_h1:
                        st.markdown(f"**{prop}:** {status}")
                elif i % 3 == 1:
                    with col_h2:
                        st.markdown(f"**{prop}:** {status}")
                else:
                    with col_h3:
                        st.markdown(f"**{prop}:** {status}")

            st.markdown("---")
            st.subheader("Safety Recommendations")
            st.info(toxicity_info['recommendations'])

            # --- NEW SECTIONS AT THE END OF TAB 3 ---
            with st.expander("📚 Transfer learning (Zero-shot Learning)", expanded=False): # Updated title
                st.markdown("""
                Our approach leverages transfer learning principles that allow us to make 
                predictions on novel polymer structures that weren't present in the training data using transfer learning of a pre-trained AutoGluon Model of the Tox21 Molecule dataset.
                """)
            with st.expander("☣️ Toxicity Classification Levels", expanded=False):
                st.markdown("""
                Polymers are classified into three concern levels depending on their toxicity properties or Hazard Criteria (0 <= Hazard Criteria <= 8):
                - **Persistent**, **Bioaccumulative(BIOACCUM)**, **carcinogenicity(CARCINOGEN)**, **mutagenic(MUTA)**, **reproductive toxicity(REPROTOX)**, 
                  **specific target organ toxicity(STOT)**, **Endocrine Disrutive Chemicals(EDC)**, and **aquatic toxicity(AQUATOX)**
                
                Polymers are classified into three concern levels:
                
                - **High** ☣️☣️☣️☣️☣️: **(4 < Hazard Criteria <= 8)** These polymers may pose significant health or environmental risks and require strict handling protocols.
                
                - **Medium** ☣️☣️☣️: **(2 < Hazard Criteria <= 4)** Moderately concerning toxicity that requires proper handling and disposal procedures.
                
                - **Low** ☣️: **(0 <= Hazard Criteria < 2)** Minimal toxicity concern under normal usage conditions.
                """)
            with st.expander("📚 References & Further Reading", expanded=False): # Updated title
                st.markdown("""
                - [CompTOX21 Data Base and Challange](https://comptox.epa.gov/dashboard/chemical-lists/tox21sl)
                - [REACH -Registration, Evaluation, Authorisation and restriction of Chemicals) REGULATION Article 57 (ANNEX XIV) TOXICITY CRITERIA](https://reachonline.eu/reach/en/title-vii-chapter-1-article-57.html?kw=classification#anch--classification)
                - [Polymer Fingerprint and PSMILE](https://psmiles.readthedocs.io/en/latest/#what-is-a-psmiles-string)
                - [GitHUb Repository](https://github.com/Ramprasad-Group/psmiles)
                - [List of chemicals with high hazards for categorisation](https://www.industrialchemicals.gov.au/help-and-guides/list-chemicals-high-hazards-categorisation)
                - [Transfer Learning](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00375)
                - [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
                - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
                - [SentenceTransformers Documentation](https://www.sbert.net/)
                - [SMILES Notation for Chemical Structures](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
                """)
            # --- END NEW SECTIONS ---

            st.markdown("---")
            st.subheader("Important Disclaimers")
            st.warning("""
                **Disclaimer:** This tool provides toxicity predictions based on **structural similarity** to known compounds and **polyBERT embeddings**. 
                It is a **screening tool** and not a substitute for rigorous experimental testing or professional toxicological assessment.
                Predictions should be used for **informational purposes only** and not for regulatory decisions.
                The accuracy of predictions depends on the quality and representativeness of the training data.
            """)

            # Generate downloadable report (CSV)
            if st.session_state.analysis_data and st.session_state.similarity_results:
                report_data = {
                    "Input_PSMILES": [st.session_state.analysis_data.get('input_psmiles', 'N/A')],
                    "Canonical_PSMILES": [st.session_state.analysis_data.get('canonical_psmiles', 'N/A')],
                    "Dimer_PSMILES": [st.session_state.analysis_data.get('dimer_psmiles', 'N/A')],
                    "Derived_Monomer_Fingerprint_600_Dims": [str(st.session_state.analysis_data.get('monomer_fp', [])).replace(', ', ';')], # Store entire 600 dims as string
                    "Cosine_Similarity_Score": [st.session_state.similarity_results.get('similarity_score', 'N/A')],
                    "Confidence_Percentage": [f"{confidence_percentage:.2f}%"],
                    "Predicted_Toxicity_Level_Similarity": [toxicity_info['label'].split(' ')[0]],
                    "Best_Match_SMILES": [st.session_state.similarity_results.get('smiles', 'N/A')],
                    "Best_Match_Toxicity_LOC": [st.session_state.similarity_results.get('loc', 'N/A')],
                    "Best_Match_Fingerprint_600_Dims": [str(st.session_state.similarity_results.get('fingerprints', [])).replace(', ', ';')], # Store entire 600 dims as string
                    **{f"REACH_Hazard_{k.replace(' ', '_')}": [v] for k, v in hazard_props.items()},
                    "Safety_Recommendations": [toxicity_info['recommendations']]
                }
                report_df = pd.DataFrame(report_data)
                st.markdown(get_table_download_link(report_df, "PolyToxiQ_Analysis_Report.csv"), unsafe_allow_html=True)
        else:
            st.info("Please process a polymer and perform similarity analysis to see toxicity predictions.")

if __name__ == "__main__":
    main()
