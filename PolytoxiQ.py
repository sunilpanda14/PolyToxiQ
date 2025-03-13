import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor
import os
# Import your config at the top of PolytoxiQ.py
import config

# Page configuration
st.set_page_config(
    page_title="Polymer Toxicity Predictor",
    page_icon="☣️",
    layout="wide"
)

# Then update your load_train_data function
@st.cache_data
def load_train_data():
    data_path = config.get_data_path()
    train_df = pd.read_csv(data_path)
    return train_df

# And update your load_model function
@st.cache_resource
def load_model():
    model_path = config.get_model_path()
    predictor = TabularPredictor.load(model_path)
    return predictor

@st.cache_resource
def load_transformer():
    # Load the polyBERT sentence transformer
    polyBERT = SentenceTransformer('kuelumbus/polyBERT')
    return polyBERT

# Generate fingerprint for input SMILES string
def generate_fingerprint(smile_string, transformer):
    # Generate embeddings (fingerprint) using polyBERT
    embedding = transformer.encode([smile_string])[0]
    return embedding

# Find closest match in training data
def find_closest_match(input_fingerprint, train_df):
    # Extract fingerprints from training data (columns 0-599)
    train_fingerprints = train_df.iloc[:, 0:600].values
    
    # Calculate cosine similarity between input and all training fingerprints
    similarities = cosine_similarity([input_fingerprint], train_fingerprints)[0]
    
    # Find index of highest similarity
    closest_idx = np.argmax(similarities)
    max_similarity = similarities[closest_idx]
    
    return closest_idx, max_similarity

# Visualize the distance between fingerprints
def visualize_distance(similarity_score):
    # The distance is 1 - similarity_score
    distance = 1 - similarity_score
    
    fig = go.Figure()
    
    # Create a distance scale from 0 to 1
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 0],
        mode='lines',
        line=dict(color='lightgrey', width=4),
        showlegend=False
    ))
    
   # Add markers for perfect match, current score, and no match
    fig.add_trace(go.Scatter(
        x=[0, similarity_score, 1], 
        y=[0, 0, 0],
        mode='markers+text',
        marker=dict(size=[15, 25, 15], color=['blue', 'red', 'green']),
        text=['No Match', f'Similarity: {similarity_score:.3f}', 'Perfect Match'],
        textposition='top center',
        showlegend=False
    ))
    
    # Highlight the distance between current similarity and perfect match
    fig.add_shape(
        type="rect",
        x0=0,
        y0=-0.02,
        x1=similarity_score,
        y1=0.02,
        fillcolor="lightgreen",
        opacity=0.5,
        line_width=0,
    )
    # Highlight the distance portion (from current similarity to Perfect Match)
    fig.add_shape(
        type="rect",
        x0=similarity_score,
        y0=-0.02,
        x1=1,
        y1=0.02,
        fillcolor="lightcoral",  # Light red color for distance
        opacity=0.5,
        line_width=0,
    ) 

     # Add similarity label (in the highlighted green area)
    fig.add_annotation(
        x=similarity_score/2,
        y=-0.05,
        text=f"Similarity: {similarity_score:.3f}",
        showarrow=False,
        font=dict(size=14)
    )

    # Add distance label
    fig.add_annotation(
        x=(similarity_score +1)/2,
        y=-0.05,
        text=f"Distance: {distance:.3f}",
        showarrow=False,
        font=dict(size=14)
    )
    
    # Update layout
    fig.update_layout(
        title="Vector Distance Visualization",
        xaxis_title="Similarity Scale (0 = No Match, 1 = Perfect Match)",
        yaxis_showticklabels=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=50),
    )
    
    return fig

# Display hazard emojis based on toxicity level
def display_hazard_emojis(toxicity_level):
    if toxicity_level.lower() == "high":
        return "☣️ ☣️ ☣️ ☣️ ☣️"
    elif toxicity_level.lower() == "medium":
        return "☣️ ☣️ ☣️"
    else:  # low
        return "☣️"

# Main application
def main():
    # Load data and models
    try:
        train_df = load_train_data()
        predictor = load_model()
        transformer = load_transformer()
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        return
    
    # Title and header
    st.title("🧬 PolyToxiQ : A Polymer Toxicity Prediction Tool")
    
    # Project background information
    with st.expander("📚 Project Background & Information", expanded=True):
        st.markdown("""
        ## About This Project
        
        This application predicts the toxicity level of polymers based on their PSMILES string representation using 
        Tranfer learning techniques and Tox21 molecular fingerprinting. 
        
        ### Methodology
        - **AutoGluon & Scikit-learn**: We used AutoGluon's TabularPredictor to build a robust machine learning model 
          that classifies polymers into different toxicity levels ( High, Medium and Low). The model was trained on a carefully curated dataset 
          of Tox 21 datsets (4974) with their known toxicity properties and level of concern(LoC).
        
        - **Cosine Similarity**: We calculate the cosine similarity between the PolyBERT Generated fingerprint of the input polymer and those 
          in our reference database of Tox21 Molecule Fingerprints. This metric measures how similar two molecular structures are in their vector space 
          representation, with values ranging from 0 (completely different) to 1 (identical).
        
        - **Zero-Shot Transfer Learning**: Our approach leverages transfer learning principles that allow us to make 
          predictions on novel polymer structures that weren't present in the training data using Transfer learning of pretrained Autogluon Model of Tox21 Molecule dataset.        
        ### Toxicity Classification Levels
        
        Polymers are classified into three concern levels depeinding on their toxicity properties or Hazard Criteria (0<= Hazard Criteria <= 8):
        - **Persistent**, **Bioaccumulative(BIOACCUM)** ,**carcinogenicity(CARCINOGEN)**, **mutagenic(MUTA)**, **reproductive toxicity(REPROTOX)**, 
           **specific target organ toxicity(STOT)**, **Endocrine Disrutive Chemicals(EDC)**, and **aquatic toxicity(AQUATOX)**
        
        ### Toxicity Classification Levels
        
        Polymers are classified into three concern levels:
        
        - **High** ☣️☣️☣️☣️☣️: **(4 < Hazard Criteria <=8)** These polymers may pose significant health or environmental risks and require strict handling protocols.
        
        - **Medium** ☣️☣️☣️: **(2 < Hazard Criteria <4)** Moderately concerning toxicity that requires proper handling and disposal procedures.
        
        - **Low** ☣️: **(0 <= Hazard Criteria <4)** Minimal toxicity concern under normal usage conditions.
        
        
        ### References & Further Reading
                    
        - [CompTOX21 Data Base and Challange](https://comptox.epa.gov/dashboard/chemical-lists/tox21sl)
        - [REACH REGULATION Article 57 (ANNEX XIV) TOXICITY CRITERIA]()
        - [Polymer Fingerprint and PSMILE](https://psmiles.readthedocs.io/en/latest/#what-is-a-psmiles-string)
        - [GitHUb Repository](https://github.com/Ramprasad-Group/psmiles)
        - [List of chemicals with high hazards for categorisation](https://www.industrialchemicals.gov.au/help-and-guides/list-chemicals-high-hazards-categorisation)
        - [Transfer Learning](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00375)
        - [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
        - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
        - [SentenceTransformers Documentation](https://www.sbert.net/)
        - [SMILES Notation for Chemical Structures](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
        """)
    
    # Input section
    st.header("Enter Polymer Information")
    smile_input = st.text_input("🔍 Polymer SMILES String:", placeholder="Enter SMILES notation for your polymer...")
    
    # Add a sample button
    if st.button("Load Sample SMILES"):
        # You can replace this with a sample SMILES from your dataset
        smile_input = "CC(C)(C)C(=O)OCCC(C)CCC(C)CCC(C)CCC(C)C"
        st.session_state.smile_input = smile_input
        st.rerun()
    
    # Process when input is provided
    if smile_input:
        with st.spinner("🔬 Analyzing polymer structure..."):
            try:
                # Step 1: Generate fingerprint for input SMILES
                input_fingerprint = generate_fingerprint(smile_input, transformer)
                
                # Step 2-4: Find closest match in training data
                closest_idx, similarity_score = find_closest_match(input_fingerprint, train_df)
                
                # Step 5: Get reference SMILES and toxicity level
                reference_smile = train_df.iloc[closest_idx, 601]  # Column 601 is ToxSMILEString
                reference_toxicity = train_df.iloc[closest_idx, 600]  # Column 600 is target
                
                # Get the stored similarity from the training data for comparison
                stored_similarity = train_df.iloc[closest_idx, 602]  # Column 602 is Max_Similarity
                
                # Display results
                st.success("✅ Analysis complete!")
                
                # Create two columns for the output
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔄 Similarity Analysis")
                    
                    # Input and reference SMILES
                    st.markdown("""
                    <style>
                    .smiles-box {
                        padding: 10px;
                        border-radius: 5px;
                        background-color: #f0f2f6;
                        margin-bottom: 10px;
                        font-family: monospace;
                        overflow-x: auto;
                        white-space: nowrap;
                        color: black;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<b>Input SMILES:</b>", unsafe_allow_html=True)
                    st.markdown(f"<div class='smiles-box'>{smile_input}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<b>Reference SMILES:</b>", unsafe_allow_html=True)
                    st.markdown(f"<div class='smiles-box'>{reference_smile}</div>", unsafe_allow_html=True)
                    
                    # Similarity scores
                    st.markdown(f"**Calculated Similarity Score:** {similarity_score:.4f}")
                    st.markdown(f"**Reference Similarity Score:** {stored_similarity:.4f}")
                    
                    # Distance visualization
                    st.plotly_chart(visualize_distance(similarity_score), use_container_width=True)
                
                with col2:
                    st.subheader("☣️ Toxicity Class Prediction")
                    
                    # Set color based on toxicity level
                    if reference_toxicity.lower() == "high":
                        color = "#FF5252"  # Red
                        emoji_count = 5
                    elif reference_toxicity.lower() == "medium":
                        color = "#FFA726"  # Orange
                        emoji_count = 3
                    else:  # low
                        color = "#66BB6A"  # Green
                        emoji_count = 1
                    
                    # Display toxicity level with styling
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center; margin-bottom: 20px;">
                        <h2 style="margin: 0; color: white;">Level of Concern: {reference_toxicity.upper()}</h2>
                        <p style="font-size: 28px; margin: 10px 0 0 0;">{display_hazard_emojis(reference_toxicity)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional information based on toxicity level
                    if reference_toxicity.lower() == "high":
                        st.warning("""
                        **High Toxicity Level**
                        
                        As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as Highly concenerned Chemicals. This polymer may pose significant health or environmental risks. Proper handling protocols and protective measures are strongly recommended.
                        
                        **Caution** : Always follow safety guidelines when working with this material.
                        """)
                    elif reference_toxicity.lower() == "medium":
                        st.info("""
                        **Medium Toxicity Level**
                        
                        As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as Moderately concern  Chemicals. This polymer has moderate toxicity concerns. It requires proper handling and disposal procedures. 
                        
                        **Caution** : Basic protective equipment is recommended when working with this material.
                        """)
                    else:
                        st.success("""
                        **Low Toxicity Level**
                        
                        As per REACH 2017 Regulation(Article 57 (ANNEX XIV)), The Structure of the input polymer matches with the Tox21 Database enlisted as low concern Chemicals. This polymer has minimal toxicity concerns under normal usage conditions.
                        
                        **Caution** : Standard laboratory safety practices are recommended as a precaution.
                        """)
                
                # Fingerprint comparison in tabular format
                st.subheader("🧬 Fingerprint Comparison")
                
                # Only show a subset of fingerprint dimensions for display (first 20)
                display_length = 600
                
                try:
                    # Extract sample fingerprints and ensure proper type conversion
                    input_fingerprint_sample = np.array(input_fingerprint[:display_length], dtype=float)
                    reference_fingerprint_sample = np.array(train_df.iloc[closest_idx, 0:display_length].values, dtype=float)
                    
                    # Calculate absolute difference
                    diff = np.abs(input_fingerprint_sample - reference_fingerprint_sample)
                    
                    # Create lists for dataframe to avoid type issues
                    dimensions = list(range(1, display_length + 1))
                    input_fp_list = [round(float(x), 4) for x in input_fingerprint_sample]
                    ref_fp_list = [round(float(x), 4) for x in reference_fingerprint_sample]
                    diff_list = [round(float(x), 4) for x in diff]
                    
                    # Create dataframe for tabular display
                    vis_df = pd.DataFrame({
                        'Dimension': dimensions,
                        'Input Polymer Fingerprint': input_fp_list,
                        'Reference Tox21 Molecule Fingerprint': ref_fp_list,
                        'Absolute Difference': diff_list
                    })
                    
                    # Add styling to highlight larger differences
                    def highlight_diff(val):
                        if isinstance(val, float) and val > 0.7:
                            return 'background-color: rgba(255, 0, 0, 0.2)'
                        return ''
                    
                    # Display the styled dataframe
                    st.dataframe(
                        vis_df.style.applymap(highlight_diff, subset=['Absolute Difference']),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Summary statistics
                    st.subheader("Fingerprint Similarity Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Difference", f"{float(np.mean(diff)):.4f}")
                    
                    with col2:
                        st.metric("Maximum Difference", f"{float(np.max(diff)):.4f}")
                    
                    with col3:
                        st.metric("Dimensions with High Difference", f"{int(np.sum(diff > 0.1))}/{display_length}")
                    
                except Exception as e:
                    st.error(f"Error displaying fingerprint comparison: {e}")
                    st.info("Displaying simplified fingerprint information instead.")
                    
                    # Simple alternative display
                    st.write("**Input Fingerprint Sample (first 5 dimensions):**")
                    st.write(input_fingerprint[:5])
                    
                    st.write("**Reference Fingerprint Sample (first 5 dimensions):**")
                    st.write(train_df.iloc[closest_idx, 0:5].values)
                
                st.info("Note: Only showing a subset of fingerprint dimensions for clarity. The actual comparison uses all dimensions.")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.error("Please check that the SMILES string is valid and try again.")

    # Footer
    st.markdown("---")
    st.markdown("📊 **Polymer Toxicity Prediction Tool** | Developed with Streamlit, AutoGluon, and SentenceTransformers")

if __name__ == "__main__":
    main()