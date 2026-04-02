```bash
$ whoami
```

```
swaroop-sridhar
├── university:     Rensselaer Polytechnic Institute
├── degree:         B.S. Computer Science & Information Technology
├── concentration:  Machine Learning
├── graduation:     May 2026
└── interests:      ML for healthcare, space science, robotics & HCI
```

---

### `research/scorec`

```python
# magnetic_reconnection_classifier.py
# Scientific Computation Research Center (SCOREC) — RPI
# Classifying rare magnetic reconnection events in space weather data

import torch
import torch.nn as nn
from torch.amp import autocast  # bfloat16 mixed-precision — 1.4x speedup

class ReconnectionCNN(nn.Module):
    """
    8.1M parameter classifier for NASA Heliophysics simulation data.
    Training loss: 0.94 → 0.34 via residual connections + batch norm.
    Mixed-precision: 28% faster training, 22% less GPU memory.
    """
    def __init__(self, in_channels=3):  # Psi, B-field, Current
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual = self._make_residual_block(64)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x + self.residual(x)  # skip connection
        return self.head(x)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR  # convergence stability
# gradient clipping · early stopping (patience=20) · physics-preserving augmentation
```

### `research/project_voice`

```python
# biometric_emotion_detection.py
# Undergraduate ML Research Co-Lead
# Assistive tech for nonspeaking individuals with profound disabilities

from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
from models import LSTM, CNN, Transformer, SVM, XGBoost, LightGBM

# 100 clinical samples → 5,000 via GMM synthesis
# Validated: 29/30 features passed Kolmogorov-Smirnov tests
gmm = GaussianMixture(n_components=5).fit(clinical_data)
synthetic = gmm.sample(5000)

# Benchmarked 9 architectures on EMG + EDA biosignals
# Best: LSTM — 70.8% accuracy, 0.68 F1
# Key markers: Corrugator EMG, Zygomaticus EMG, skin conductance
results = benchmark([SVM, XGBoost, LightGBM, CNN, LSTM, Transformer], 
                    signals=["EMG_corrugator", "EMG_zygomaticus", "EDA"])
```

---

### `projects/`

```python
# fytospot.py — Plant Identification System
# ResNet-50 backbone · real-time OpenCV tracking · CustomTkinter GUI

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_species)
# Live camera feed → detection + classification in variable lighting
```

```python
# drug_discovery.py — AI-Driven Molecular Interaction Analysis
# Protein-drug binding prediction · DNA sequence analysis · immune response modeling

pipeline = FeatureEngineering() >> MolecularModel() >> InteractionPredictor()
# Biological data → optimized representations → predicted interactions
```

---

```bash
$ cat achievements.log
```

```
[2025] Best Artificial Intelligence Hack — RPI Hackathon
[2025] Best Use of DigitalOcean Gradient AI — MLH @ RPI Hackathon
[2024] Dean's List — Rensselaer Polytechnic Institute
```

---

```bash
$ cat stack.conf
```

```ini
[languages]
primary    = Python, C++, C, Java, JavaScript, SQL
systems    = Assembly, Rust
functional = Haskell, Erlang, Prolog

[ml_and_research]
frameworks = PyTorch, TensorFlow, scikit-learn
boosting   = XGBoost, LightGBM, CatBoost
compute    = CUDA, mixed-precision (bfloat16)
tools      = NumPy, SciPy, pandas, Matplotlib, OpenCV, Jupyter

[web]
frontend   = React, Next.js, TailwindCSS, HTML/CSS
backend    = Node.js, Flask, REST APIs, PHP
database   = MySQL

[devops]
platforms  = Linux, Docker, Azure
tools      = Git, GitHub, CMake, Vim

[exploring]
next       = Robotics, Human-Computer Interaction
```

---

```bash
$ ./connect.sh
```

```
LinkedIn       → linkedin.com/in/swaroop-sridhar21
Portfolio      → swaroopsridhar.com
Stack Overflow → stackoverflow.com/users/28245048
YouTube        → youtube.com/@SwaroopSridhar-pu4to
```

---

<div align="center">

```


⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠙⣶⡄⠀⠀⠀  ⠀⠀⢀⣶⡞⢳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⣳⡘⠦⠤⠤⠤⠤⠤⣄⢚⣰⣣⡀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣸⠛⠃⠀⠀⠀⡀⠀⠀⠀⠀⢻⣠⣿⢇⣛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢐⠋⠀⠶⣦⡀⠀⣐⠃⢀⣠⡶⠆⠀⠉⢛⣪⢛⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠋⠀⢰⢶⣮⣙⡀⠉⡘⣫⢷⣶⣶⡄⠀⠀⠘⣢⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣭⠀⠀⠈⠉⢉⡍⠀⠈⠁⠉⠉⠉⠉⠤⠀⠈⢠⠙⣭⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣭⠀⠀⠀⣴⣾⣷⣶⠄⢀⠀⠀⣄⠀⠀⠀⠀⢨⠄⣥⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢩⡀⠀⠀⠻⣽⣿⣯⣐⣠⡤⠾⠀⠀⠀⠀⢐⣍⣚⣓⣱⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣽⣤⡀⠀⠀⠙⠯⠉⠉⠀⠀⠀⠀⠀⣠⣼⣬⣿⣭⢩⣡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⡏⠹⣥⢀⡠⠄⠀⠀⠀⠀⢤⡀⠀⢉⡏⠹⠉⠁⢠⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣶⡷⠁⠀⠀⢨⣷⣿⠙⡾⢿⣷⣆⠁⠀⠀⠀⠀⠀⢠⢶⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠶⠂⠀⠀⠀⠀⠾⠇⠚⡞⡟⠳⠄⠘⠶⠄⠀⠀⠀⠀⠶⠶⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢿⠃⠀⢀⡠⠼⠉⠀⠀⠀⢠⢟⡆⠀⠀⠀⠻⠤⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣛⠋⠀⠀⠋⠀⠀⠀⠀⢀⡴⠋⠘⠛⣂⡀⠀⠀⠀⠃⠐⠀⠀⢸⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⣠⣴⡞⠀⠀⠀⠀⠙⢻⣤⠀⠀⠀⠀⠀⠀⢸⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡄⠀⠀⢠⣤⠍⠉⠁⠀⠀⠀⠀⠀⠀⠹⢧⣄⠀⠀⠀⢆⡜⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠈⠉⠉⠉⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

```python
# he's learning unsupervised
```

</div>
