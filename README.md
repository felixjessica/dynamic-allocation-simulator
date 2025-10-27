# 🧠 Dynamic Allocation Simulator  
A Python-based optimisation engine demonstrating AI-driven decision logic for operational resource allocation.

---

## 🎯 Overview  
This project re-imagines my **Udacity AI Programming with Python Nanodegree – Image Classifier** as a **Dynamic Allocation Simulator**.  
The model showcases how lightweight machine-learning logic can drive real-time allocation decisions — assigning the right resource to the right task to maximise utilisation and throughput.

By reframing image classification into resource classification, the simulator replicates the decision logic used in **Dynamic Allocation Services (DAS)** platforms for airports, logistics hubs, and car-park operations.

---

## ⚙️ Problem Statement  
Operational environments such as airports or logistics centres often allocate assets manually — leading to:

- Idle resources during high demand  
- Longer wait and turnaround times  
- Lack of measurable performance visibility  

**Goal:** simulate an intelligent allocation engine that uses data patterns to optimise assignments dynamically, improving overall system efficiency.

---

## 💡 Solution Approach  
The simulator applies classification logic (adapted from an image classifier) to predict optimal resource–task pairings.

### 🧾 Data Preparation & Feature Engineering  
- Synthetic dataset representing operational states (e.g., “available,” “busy,” “idle”).  
- **NumPy** and **pandas** for data transformation and normalisation.  
- Each state reframed as an “image class” to feed into the network.  

---

### ⚙️ Allocation Engine (Model Architecture)

* Adapted the original classifier to output the most efficient allocation for each task.
* Softmax probabilities represent resource options; `argmax` picks the best resource.

```python
prediction = model.predict(task_features)
best_resource = np.argmax(prediction)
```

---

### 📊 KPI Definition & Model Optimisation

* **Accuracy →** Allocation Accuracy
* **Loss →** Operational Inefficiency
* **Custom KPIs:** throughput, idle-time reduction, decision latency

---

### 📈 Evaluation & Visualisation

* **Matplotlib** used to visualise performance improvement.
* Compared manual vs. AI-based allocation performance.

---

## 📊 Key Results (Simulation)

| KPI                   | Description                           | Improvement       |
| --------------------- | ------------------------------------- | ----------------- |
| Allocation Accuracy   | Correct resource–task pairing ratio   | **87%**           |
| Idle Time Reduction   | Decrease in unused resource hours     | **15%**           |
| Throughput Efficiency | Increase in tasks completed per cycle | **+12%**          |
| Decision Latency      | Average allocation time               | **0.2 s / batch** |

> *Replace with your actual results after running Notebook 3.*

---

## 🧱 Repository Structure

```bash
dynamic-allocation-simulator/
├─ notebooks/
│  ├─ 1_data_preparation.ipynb          # Data preprocessing (from original utility.py)
│  ├─ 2_allocation_model.ipynb          # Model setup & training loop
│  └─ 3_kpi_evaluation.ipynb            # KPI calculation & performance plots
├─ src/
│  ├─ preprocess.py                      # Data preprocessing functions
│  ├─ allocation_model.py                # Model architecture & training logic
│  ├─ predict.py                         # Inference script for allocation
│  ├─ metrics.py                         # KPI calculations
│  ├─ batch_infer.py                     # Batch allocation decisions
│  └─ model_utils/
│     └─ classifier.py                   # Model loader and helper functions
├─ visuals/
│  ├─ performance_comparison.png         # Manual vs. AI allocation performance
│  └─ allocation_heatmap.png             # Utilisation heatmap
├─ data/                                 # Synthetic or real datasets
└─ README.md
```

---

## 🧰 Tech Stack

| Category      | Tools & Libraries                                              |
| ------------- | -------------------------------------------------------------- |
| Programming   | Python 3, Jupyter Notebook                                     |
| Data Handling | NumPy, pandas                                                  |
| Visualisation | Matplotlib                                                     |
| Modelling     | Neural-network-based classifier adapted for optimisation logic |

---

## 📈 Visual Outputs

Export the following to the `visuals/` directory:

* `performance_comparison.png` – Bar chart showing manual vs. AI allocation efficiency
* `allocation_heatmap.png` – Heatmap of resource utilisation across time

> *Use the Matplotlib plots generated in Notebook 3.*

---

## 🧩 Insights & Impact

* Demonstrates the **transferability** of AI models from perception tasks (image recognition) to decision-support systems.
* Proves capability in **data modelling**, **KPI definition**, and **algorithmic optimisation** — critical for **Technical Product Managers**.
* Serves as a foundation for extending into real-time DAS simulation with APIs or live dashboards.

---

## 🚀 Future Enhancements

* Integrate **Google OR-Tools** or **PuLP** for linear programming optimisation.
* Build a **Streamlit** dashboard for interactive allocation visualisation.
* Deploy a **FastAPI** microservice to expose the allocation engine as a REST endpoint.

---

## 📎 Author

**Jessica Ofoh**
*Technical PM | AI-Driven Product Manager*

