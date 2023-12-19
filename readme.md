The following modules can be directly executed in order to produce all models, figures and tables presented in the paper and the SI:

* app.py reads the input group data, builds and pickles the models and domains of applicability, and produces most images and tables
* model_use_REACH.py uses the pickled RF model to predict the group membership of all REACH registered substances
* model_use_visualise_fingerprint_bit_heatmap.py uses the pickled RF and produces the visualisation of the 500 globally most important features (fingerprint bits)
* model_use_rf_shap_analysis.py produces all figures related to SHAP values (class or structure specific feature importance)