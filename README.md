# Age & Gender Recognition System (B.Tech Final Year Project)

A complete, state-of-the-art Deep Learning project that performs real-time **Age and Gender Recognition** using Convolutional Neural Networks (CNNs). The application features a robust **TensorFlow/Keras** machine learning pipeline, a lightweight **Flask REST API** backend, and a modern, interactive **React** frontend client.

---

## 🌟 Key Features

* **Multi-Stage CNN Pipeline**: 
  * A primary CNN model classifies input faces by **Gender** (Male / Female).
  * Two separate, gender-specific CNN models perform specialized **Age Range Classification** (`0-12`, `13-19`, `20-30`, `31-45`, `46-60`, `60+`) to maximize accuracy.
* **Flask REST API Backend**: Receives image uploads, decodes them via OpenCV, feeds them to the deep learning models, and returns JSON prediction results in milliseconds.
* **Interactive React Frontend**: A responsive, user-friendly client that allows users to upload/drag-and-drop face images, visualize them, and view high-fidelity gender/age predictions instantly.
* **Comprehensive Engineering Diagrams**: Fully modeled Data Flow Diagrams (DFD Levels 0, 1, 2), Entity Relationship (ER) diagrams, and Use Case diagrams are provided to document the system architecture.

---

## 📁 Repository Structure

```tree
BTECH_PROJECT/
├── app_name/                  # React Frontend Client
│   ├── public/                # Static assets
│   ├── src/                   # React components and application logic
│   ├── package.json           # Frontend dependencies & scripts
│   └── .gitignore             # Frontend git ignore definitions
├── backup/                    # Backup scripts and archived code
├── doc/                       # Project documentation
│   └── Final_Year_Document.docx # Full B.Tech Thesis/Report
├── static/                    # Backend static directories
│   └── uploads/               # Temporary uploads folder
├── .gitignore                 # Root git ignore (filters out huge datasets & models)
├── backend.py                 # Core Flask REST API Server
├── main.py                    # CNN model definition, training pipeline & evaluation
├── DATA.py                    # Dataset utility script
├── finalyrproject.py          # Unified Python training/run script
├── index.html                 # Demo web landing page
├── er_diagram.png             # Database / Entity Relationship Diagram
├── use_case_diagram.png       # Use Case Diagram
├── dfd_level_0.png            # Data Flow Diagram - Level 0
├── dfd_level_1.png            # Data Flow Diagram - Level 1
├── dfd_level_2.png            # Data Flow Diagram - Level 2
└── README.md                  # Project Documentation (This file)
```

> [!NOTE]  
> The raw dataset (`age_gender.csv`) and the trained TensorFlow model weights (`gender_model.h5`, `male_age_model.h5`, `female_age_model.h5`) are excluded from this repository using `.gitignore` to keep the repository lightweight and comply with GitHub's file size limits (< 100 MB). You can easily train them locally using `main.py`.

---

## 🛠️ Tech Stack

### Deep Learning & Machine Learning
* **TensorFlow / Keras**: CNN Architecture design, layer construction, training, and model serialization.
* **OpenCV (cv2)**: Image decoding, color channel conversion, resizing, and pixel-value normalization.
* **Pandas & NumPy**: Structuring dataset inputs, vector manipulations, and matrix transformations.
* **Scikit-learn**: Data division (`train_test_split`), label encoding (`LabelEncoder`), and validation metrics.

### Backend & API
* **Flask**: Micro Web Framework for handling HTTP POST image requests.
* **Flask-CORS**: Facilitating cross-origin resource sharing between the Flask backend (default port `5000`) and the React client (default port `3000`).

### Frontend UI
* **React.js**: Modular component architecture, upload state management, and real-time fetch requests.
* **Vanilla CSS**: Clean, premium styling for a smooth user experience.

---

## 🚀 Installation & Local Setup

To run this project on your local machine, follow these steps:

### 1. Prerequisites
Ensure you have the following installed:
* **Python 3.8 - 3.11**
* **Node.js (v16.x or newer)** and **npm**
* **Git**

### 2. Backend Setup
1. **Clone the Repository**:
   ```bash
   git clone <your-github-repo-url>
   cd BTECH_PROJECT
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate the Virtual Environment**:
   * **Windows**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```
4. **Install Python Dependencies**:
   ```bash
   pip install tensorflow opencv-python flask flask-cors pandas scikit-learn numpy
   ```

### 3. Model Training & Preparation (Optional)
If you want to train the models from scratch using the **UTKFace** dataset:
1. Download the UTKFace cropped dataset or use your custom dataset.
2. Edit the paths in `main.py` (lines 11–13) to point to your dataset directories:
   ```python
   FEMALE_PATH = "path_to_female_images"
   MALE_PATH = "path_to_male_images"
   ALL_PHOTOS_PATH = "path_to_all_utkface_images"
   ```
3. Execute the training script:
   ```bash
   python main.py
   ```
   *This will train the models for 20 epochs and output three files:*
   * `gender_model.h5`
   * `male_age_model.h5`
   * `female_age_model.h5`

4. *Alternatively, place pre-trained `.h5` model files directly in the root directory.*

### 4. Running the Flask Backend
Start the API server by running:
```bash
python backend.py
```
*The backend server will run on `http://127.0.0.1:5000`.*

### 5. Frontend Setup
1. Open a new terminal window/tab.
2. Navigate to the `app_name` folder:
   ```bash
   cd app_name
   ```
3. Install frontend packages:
   ```bash
   npm install
   ```
4. Run the React development server:
   ```bash
   npm start
   ```
   *The client will compile and open in your default browser at `http://localhost:3000`.*

---

## 📊 System Architecture & Diagrams

To help explain the architecture for academic submissions and B.Tech project vivas, the project includes several diagrams:

### Entity Relationship Diagram
Describes the database entities, model inputs, and relationship schemas.
* See [er_diagram.png](er_diagram.png)

### Use Case Diagram
Highlights the interactions between the end-user, the backend Flask middleware, and the underlying Keras classification engines.
* See [use_case_diagram.png](use_case_diagram.png)

### Data Flow Diagrams (DFD)
Documents how image binary data streams from the front-end interface, undergoes transformation via OpenCV, and passes into the trained TensorFlow prediction layers:
* **DFD Level 0**: [dfd_level_0.png](dfd_level_0.png)
* **DFD Level 1**: [dfd_level_1.png](dfd_level_1.png)
* **DFD Level 2**: [dfd_level_2.png](dfd_level_2.png)

---

## 📝 License
This project is for academic and educational purposes as part of a B.Tech degree program. Feel free to use and adapt the code for study or research!
