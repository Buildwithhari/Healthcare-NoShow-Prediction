# Hospital Appointment No-Show Analytics

Live Streamlit dashboard converted from the included Google Colab notebook.

## Deploy
1. Upload `app.py`, `requirements.txt`, and `Hospital_NoShow_Colab.ipynb` to a public GitHub repository.
2. Optional: upload the original `KaggleV2-May-2016.csv` to the same folder **only if redistribution is permitted**. Without it, visitors must upload the dataset themselves.
3. Open https://share.streamlit.io, connect GitHub, select your repository and branch, and set the main file to `app.py`.
4. Deploy and copy the live `*.streamlit.app` link.

## Run locally
`pip install -r requirements.txt`
`streamlit run app.py`

Dataset: https://www.kaggle.com/datasets/joniarroba/noshowappointments

Model follows the notebook: 80/20 stratified split; balanced Decision Tree with max depth 8 and minimum leaf 50. Dashboard filters do not change model training.
