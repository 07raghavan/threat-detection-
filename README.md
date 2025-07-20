

  <h1>🎧 Multiclass Audio Event Detection with Deep Learning</h1>
<img width="518" height="916" alt="image" src="https://github.com/user-attachments/assets/8c81bcd8-c401-426c-9431-33e94b0aa7e5" />

  <p>A robust pipeline for <strong>multiclass audio event detection</strong> using deep learning. The system classifies audio into four classes:</p>
  <ul>
    <li>🔫 <strong>Gunshot</strong></li>
    <li>🚗 <strong>Vehicle</strong></li>
    <li>🪓 <strong>Wood_Cutting</strong></li>
    <li>✅ <strong>NonThreat</strong></li>
  </ul>

  <p>It includes real-time <strong>alerting logic</strong> (e.g., via LoRa) and deployment-ready <strong>TensorFlow Lite</strong> models for edge devices.</p>

  <h2>📁 Dataset Structure</h2>
  <pre><code>org_dataset/
├── Gunshot/
│   ├── file1.wav
│   └── ...
├── Vehicle/
│   ├── file1.wav
│   └── ...
├── Wood_Cutting/
│   ├── file1.wav
│   └── ...
└── NonThreat/
    ├── file1.wav
    └── ...
</code></pre>
  <ul>
    <li><strong>Format</strong>: .wav files per class</li>
    <li><strong>Holdout Set</strong>: A separate folder (e.g., <code>realtest/</code>) with test samples from each class</li>
  </ul>
<img width="643" height="995" alt="Screenshot 2025-07-21 001031" src="https://github.com/user-attachments/assets/fd68bfd6-aadd-4f19-99de-1ea65c3f3a7c" />

  <h2>🚀 Features</h2>
  <ul>
    <li>📊 <strong>EDA</strong>: Class distribution visualization and spectrogram previews</li>
    <li>🧹 <strong>Preprocessing</strong>: Converts audio to log-mel spectrograms; ready for CNN input</li>
    <li>🧠 <strong>Model</strong>:
      <ul>
        <li>Data augmentation</li>
        <li>Batch normalization</li>
        <li>Dropout</li>
        <li>L2 regularization</li>
      </ul>
    </li>
    <li>🏋️ <strong>Training</strong>:
      <ul>
        <li>Early stopping</li>
        <li>Learning rate reduction</li>
        <li>Checkpointing best model</li>
      </ul>
    </li>
    <li>✅ <strong>Evaluation</strong>: Confusion matrix & classification report</li>
    <li>🛠️ <strong>Deployment</strong>: Converted to TFLite for edge inference</li>
  </ul>

  <h2>🛠️ Installation</h2>

  <h3>1. Clone the Repository</h3>
  <pre><code>git clone https://github.com/yourusername/audio-event-detection.git
cd audio-event-detection</code></pre>

  <h3>2. Create Virtual Environment & Install Dependencies</h3>
  <pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt</code></pre>

  <h3>📦 Main Dependencies</h3>
  <ul>
    <li>tensorflow</li>
    <li>librosa</li>
    <li>numpy</li>
    <li>matplotlib</li>
    <li>scikit-learn</li>
  </ul>

  <h2>📂 Dataset Setup</h2>
  <ul>
    <li>Place your dataset in the structure mentioned above</li>
    <li>Update dataset paths in the scripts if needed</li>
  </ul>
<img width="2163" height="1456" alt="Screenshot 2025-07-20 232645" src="https://github.com/user-attachments/assets/7eb7c1f0-1088-40c7-86c9-bd203962b003" />
  <h2>🔧 Customization</h2>
  <ul>
    <li><strong>Class Names</strong>: Update the <code>class_names</code> list in scripts for custom classes</li>
    <li><strong>Alert Logic</strong>: Modify <code>send_lora_alert()</code> in <code>tflite_inference_lora.py</code></li>
    <li><strong>Paths</strong>: Update dataset/model paths as per your directory setup</li>
  </ul>
<img width="401" height="1392" alt="Screenshot 2025-07-21 001056" src="https://github.com/user-attachments/assets/55fa29bc-3944-4901-b0f0-62abfd30a3ce" />

  <h2>🧠 Model Architecture</h2>
  <pre><code>ConvBlock x3 (Conv2D → BatchNorm → MaxPooling)
↓
Data Augmentation (random translation, zoom)
↓
GlobalAveragePooling2D
↓
Dense Layer → Dropout
↓
Output Layer (Softmax)</code></pre>
  <p>📸 Refer to <code>cnn_model_architecture.png</code> for a diagrammatic view.</p>

  <h2>📊 Results</h2>
  <ul>
    <li><strong>Validation Accuracy</strong>: ~96% on holdout set</li>
    <li><strong>Evaluation Metrics</strong>: Confusion Matrix, Classification Report (from <code>test_holdout.py</code>)</li>
  </ul>
<img width="501" height="280" alt="image" src="https://github.com/user-attachments/assets/70fe860b-2863-4067-8626-69e3d1f939e4" />
<img width="484" height="392" alt="image" src="https://github.com/user-attachments/assets/65b6490e-f1c9-4d12-8f2d-d0299a757894" />



  <h2>📱 Edge Deployment</h2>
  <ul>
    <li><strong>Model File</strong>: <code>multiclass_model.tflite</code> (~122 KB)</li>
    <li><strong>Deploy to</strong>: Raspberry Pi, microcontrollers, IoT devices</li>
    <li><strong>Real-Time Detection</strong>: Included example with alerting logic</li>
  </ul>


  <h2>🙌 Credits</h2>
  <p>Built with 💡 using TensorFlow, Librosa, and open-source tools.</p>

</body>
</html>
