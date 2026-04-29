# ChemML-Pro (SensorNet.ai)
### End-to-End Machine Learning Platform for Chemical Process Prediction

**Built by:** Deepak 
**Education:** IIT Madras, Chemical Engineering (2nd Year)  
**Status:** Actively Building (Phase 0 of 4)  

---

## 🔬 What This Project Does
ChemML-Pro predicts chemical plant outcomes using Machine Learning:
* **Process yield prediction** (regression)
* **Fault classification** (identifying 6 specific chemical fault types)
* **Anomaly detection** (autoencoder for early warning systems)
* **Molecular property prediction** (SMILES → solubility)

## 📊 Dataset
**Tennessee Eastman Process (TEP):** A highly complex chemical process simulation dataset widely used in process control research. It contains 52 continuous sensor variables (temperatures, pressures, flow rates) and 21 distinct fault types. 
*(Note: A 5% downsampled version of the Extended TEP dataset is used for classical ML to optimize training memory).*

## 💻 Tech Stack
* **Language:** Python
* **Data Processing:** NumPy, Pandas
* **Classical ML:** Scikit-learn, XGBoost, Random Forest, SVM
* **Deep Learning:** PyTorch
* **Deployment:** Streamlit

---

## 🚀 Project Phases

| Phase | Status | Description |
| :--- | :--- | :--- |
| **0** | ✅ **Done** | Repository setup, environment config, automated data ingestion & sampling pipeline. |
| **1** | ⏳ **Next** | Exploratory Data Analysis (EDA), correlation heatmaps, class balance checks. |
| **2** | 🗓️ *Planned* | Classical ML Engine: Training XGBoost, RF, and SVM models for fault classification. |
| **3** | 🗓️ *Planned* | Deep Learning: PyTorch Neural Networks and Autoencoders for anomaly detection. |
| **4** | 🗓️ *Planned* | Full Streamlit web application deployment. |

---

## ⚙️ How to Run the Data Pipeline
To automatically build the data folders, ingest the raw dataset, generate a 5% sample, and export a clean version for training:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/SensorNet.ai.git](https://github.com/YOUR_USERNAME/SensorNet.ai.git)

# Navigate into the project directory
cd SensorNet.ai

# Run the data loader script
python src/data_loader.py
