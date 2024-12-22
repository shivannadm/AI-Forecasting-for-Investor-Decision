# AFID: AI-Forecasting-for-Investor-Decision

## Project Overview
This project aims to create a **high-frequency stock price prediction model** using **Artificial Intelligence (AI)** and **Deep Learning (DL)** techniques to forecast stock prices. The model improves upon traditional daily predictions by offering **timely, fine-grained insights** to support more agile trading decisions.

---

## Methodology
1. **Data Collection**: Gather day-based stock price data, such as open, high, low, and closed.
2. **Data Preprocessing**: Clean and normalize data to ensure it is suitable for model input.
3. **Feature Engineering**: Extract key features like trading volume and volatility measures.
4. **Model Design**:
   - Implement LSTM to capture long-term dependencies.
   - Use CNN to identify short-term patterns and trends.
   - Combine CNN-LSTM for an integrated approach to enhance accuracy.
5. **Training and Evaluation**:
   - Split the data into training, validation, and testing sets.

---

## Expected Outcome
The project will deliver an accurate, **real-time stock price prediction model** capable of day-wise predictions. Key benefits include:
- Improved prediction accuracy for both short-term and long-term market movements.
- Granular, daily insights to help investors make timely and informed trading decisions.
- Potential integration into trading platforms or advisory tools for practical applications.

---

## Project Highlights
- **Granularity**: Day-wise predictions for fine-grained insights.
- **Hybrid Approach**: Combines LSTM and CNN for enhanced prediction accuracy.
- **Automation**: AI and ML streamline data processing and feature selection.
- **Real-Time Capability**: Supports agile and responsive trading decisions.

---

## Technologies Used
- **Python**
- **TensorFlow/Keras** (Deep Learning frameworks)
- **Pandas, NumPy** (Data manipulation)
- **Scikit-learn** (Machine Learning utilities)
- **Matplotlib/Seaborn** (Visualization tools)
- **Jupyter Notebook** (Model development environment)

---

## How to Run the Project
1. Clone this repository:
   
bash
   git clone https://github.com/shivannadm/AI-Forecasting-for-Investor-Decision.git
   cd AI-Forecasting-for-Investor-Decision

2. Install required dependencies:
   
bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook to train and evaluate the model.
   
bash
   jupyter notebook

4. Follow the instructions in the notebook to preprocess data, train the models, and generate predictions.

---

## Folder Structure
plaintext
high-frequency-stock-prediction/
│
├── data/                 # Raw and preprocessed data
├── models/               # Saved model files
├── notebooks/            # Jupyter Notebooks for development
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── results/              # Output predictions and performance metrics


---

## Contributors
- **Shivanna** (shivannadm6@gmail.com)

---

## Acknowledgments
Special thanks to the open-source libraries and frameworks that made this project possible, including TensorFlow, Keras, and Scikit-learn.

---

## Contact
For inquiries or collaboration opportunities, feel free to reach out:
- **Email**: shivannadm6@gmail.com
- **LinkedIn**: [Shivanna DM](https://www.linkedin.com/in/shivannadm)

---

**Let's forecast the market with precision and agility!**
