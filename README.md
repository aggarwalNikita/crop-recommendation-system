# 🌾 Crop Recommendation System

A machine learning-based web application that recommends the most suitable crops based on environmental conditions and soil parameters. This system helps farmers make data-driven decisions for optimal crop selection.

## 📊 Project Overview

The Crop Recommendation System uses a Random Forest Classifier trained on Indian agricultural data to predict the best crops for specific environmental conditions. The system considers temperature, humidity, pH levels, rainfall, and district-specific factors to provide accurate crop recommendations along with yield information.

## ✨ Features

- 🤖 **Machine Learning Predictions**: Uses Random Forest Classifier for accurate crop recommendations
- 🌍 **Regional Support**: Covers 311 districts across India
- 📈 **Yield Information**: Provides average yield data for recommended crops
- 🎯 **User-Friendly Interface**: Clean, responsive web interface with input validation
- 📋 **Input Guidance**: Range indicators and placeholder text for all parameters
- 🔍 **Real-time Predictions**: Instant results based on user input

## 🚀 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Data Processing**: Pandas, NumPy
- **Model Storage**: Joblib (pickle files)
- **Frontend**: HTML5, CSS3, Jinja2 templating
- **Data Visualization**: Custom CSS styling

## 📁 Project Structure

```
crop-recommendation-system/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Web interface template
├── static/
│   └── style.css                   # CSS styling
├── models/
│   ├── crop_recommendation_model.pkl    # Trained ML model
│   ├── label_encoder.pkl              # Crop label encoder
│   └── crop_label_encoder.pkl         # Additional encoder
├── indian_crop_weather.csv            # Dataset with crop and weather data
├── Crop_recommendation.csv            # Training dataset
├── crop recommendation.ipynb          # Jupyter notebook for model training
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/aggarwalNikita/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

The application will start running on `http://localhost:5000` or `http://127.0.0.1:5000`

## 📋 Required Dependencies

```
flask
joblib
numpy
pandas
scikit-learn
```

## 🎯 How to Use

1. **Open your web browser** and navigate to `http://localhost:5000`

2. **Enter the required parameters**:
   - **Temperature (°C)**: Range 20-28°C (validation: 15-35°C)
   - **Humidity (%)**: Range 60-80% (validation: 40-95%)
   - **pH**: Range 6.0-6.5 (validation: 5.0-8.0)
   - **Rainfall (mm)**: Range 600-1200mm (validation: 200-2000mm)
   - **District**: Select from 311 available Indian districts

3. **Click "Recommend Crop"** to get predictions

4. **View Results**:
   - Recommended crop type
   - Average yield information in kg/ha

## 🧪 Sample Test Values

### For Rice Cultivation:
- **Temperature**: 25°C
- **Humidity**: 80%
- **pH**: 6.5
- **Rainfall**: 1200mm
- **District**: Durg

### For Wheat Cultivation:
- **Temperature**: 22°C
- **Humidity**: 65%
- **pH**: 6.2
- **Rainfall**: 800mm
- **District**: Jabalpur

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 5 input features (temperature, humidity, pH, rainfall, district)
- **Dataset**: Indian crop and weather data with 50,767+ records
- **Districts Covered**: 311 districts across multiple Indian states
- **Crop Types**: Rice, Wheat, Maize, Chickpea, Cotton, and more

## 🗺️ Supported Regions

The system covers districts from multiple Indian states including:
- **Chhattisgarh**: Durg, Bastar, Raipur, Bilaspur, Raigarh, Surguja
- **Madhya Pradesh**: Jabalpur, Balaghat, Sagar, Damoh, Tikamgarh
- **Andhra Pradesh**: Kurnool, Ananthapur, Chittoor, Kadapa
- **Telangana**: Hyderabad, Nizamabad, Warangal, Nalgonda
- **Karnataka**: Bangalore, Mysore, Mandya, Hassan
- And many more...

## 🚨 Troubleshooting

### Common Issues:

1. **Module Import Errors**: Ensure all dependencies are installed using `pip install -r requirements.txt`

2. **Model Loading Errors**: Verify that all `.pkl` files are present in the `models/` directory

3. **CSV File Errors**: Ensure `indian_crop_weather.csv` is in the root directory

4. **Port Already in Use**: Change the port in `app.py` or stop other Flask applications

## 🔧 Development

### To modify the model:
1. Open `crop recommendation.ipynb`
2. Train your model with new data
3. Save the model using `joblib.dump()`
4. Update the model path in `app.py`

### To add new features:
1. Update the HTML form in `templates/index.html`
2. Modify the prediction logic in `app.py`
3. Update the CSS styling in `static/style.css`

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author & Developer

**NIKITA**

*Full Stack Developer & Data Scientist*

- 🌟 Passionate about agricultural technology and machine learning
- 🎯 Focused on creating solutions that help farmers make better decisions
- 📧 Contact: nikita.aggarwal@email.com
- 🔗 LinkedIn: [linkedin.com/in/nikita-aggarwal](https://linkedin.com/in/nikita-aggarwal)
- 🐙 GitHub: [github.com/aggarwalNikita](https://github.com/aggarwalNikita)

## 🙏 Acknowledgments

- Indian Agricultural Research Institute for the dataset
- scikit-learn community for the machine learning framework
- Flask community for the web framework
- All contributors who helped improve this project

## 📊 Project Stats

- **Total Districts**: 311
- **Data Records**: 50,767+
- **Model Accuracy**: High precision for crop recommendations
- **Supported Crops**: Multiple crop varieties
- **Real-time Predictions**: Instant results

---

### 🌱 *"Empowering farmers with data-driven crop recommendations for sustainable agriculture"*

**Made with ❤️ by NIKITA**
