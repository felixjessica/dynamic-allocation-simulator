# 🧠 Dynamic Allocation Simulator
**A Python-based optimisation engine demonstrating AI-driven decision logic for operational resource allocation**

---

## 🎯 Overview
This project re-imagines my **Udacity AI Programming with Python Nanodegree – Image Classifier** as a **Dynamic Allocation Simulator**.  
The model showcases how lightweight machine-learning logic can drive **real-time allocation decisions** — assigning the right resource to the right task to maximise utilisation and throughput.

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
The simulator applies **classification logic** (adapted from an image classifier) to predict optimal resource–task pairings.

1. **Data Preparation & Feature Engineering**
   - Synthetic dataset representing operational states (e.g., “available,” “busy,” “idle”).
   - **NumPy** and **pandas** for data transformation and normalisation.
   - Reframed each state as an “image class” to feed into the network.
   ```python
   df['normalised_load'] = (df['current_load'] - df['current_load'].min()) / (df['current_load'].max() - df['current_load'].min())
   features = np.array(df[['normalised_load', 'distance', 'priority']])
   ```

2. **Allocation Engine (Model Architecture)**
   - Adapted the original classifier to output the most efficient allocation for each task.
   - **Softmax** probabilities represent resource options; **argmax** picks the best resource.
   ```python
   prediction = model.predict(task_features)
   best_resource = np.argmax(prediction)
   ```

3. **KPI Definition & Model Optimisation**
   - Accuracy → *Allocation Accuracy*
   - Loss → *Operational Inefficiency*
   - Custom KPIs: **throughput**, **idle-time reduction**, **decision latency**.

4. **Evaluation & Visualisation**
   - **Matplotlib** to visualise performance improvement.
   - Compared manual vs. AI-based allocation performance.

---

## 📊 Key Results (Simulation)
| KPI | Description | Improvement |
|------|--------------|-------------|
| **Allocation Accuracy** | Correct resource–task pairing ratio | **87 %** |
| **Idle Time Reduction** | Decrease in unused resource hours | **15 %** |
| **Throughput Efficiency** | Increase in tasks completed per cycle | **+12 %** |
| **Decision Latency** | Average allocation time | **0.2 s / batch** |

> Replace with your actual results after running Notebook 3.

---

## 🧱 Repository Structure
```
dynamic-allocation-simulator/
├─ notebooks/
│  ├─ 1_data_preparation.ipynb          # NumPy/pandas transforms (from original utility.py)
│  ├─ 2_allocation_model.ipynb          # model setup & training loop (from train.py)
│  └─ 3_kpi_evaluation.ipynb            # batch predict + KPI plots (predict.py + metrics)
├─ src/
│  ├─ preprocess.py                      # refactored from utility.py
│  ├─ allocation_model.py                # refactored from train.py
│  ├─ predict.py                         # refactored from predict.py
│  ├─ metrics.py                         # from calculates_results_stats.py (KPIs)
│  ├─ batch_infer.py                     # from classify_images.py (batch allocation decisions)
│  └─ model_utils/
│     └─ classifier.py                   # low-level model loader/helpers
├─ visuals/
│  ├─ performance_comparison.png         # manual vs AI allocation efficiency
│  └─ allocation_heatmap.png             # utilisation heatmap
├─ data/                                 # synthetic or real datasets
└─ README.md
```

---

## 🧰 Tech Stack
| Category | Tools & Libraries |
|-----------|------------------|
| Programming | **Python 3**, Jupyter Notebook |
| Data Handling | **NumPy**, **pandas** |
| Visualisation | **Matplotlib** |
| Modelling | Neural-network-based classifier adapted for optimisation logic |

---

## 📈 Visual Outputs
Export the following to `visuals/`:
- `performance_comparison.png` – Bar chart showing manual vs AI allocation efficiency
- `allocation_heatmap.png` – Heatmap of resource utilisation across time

*(Use your Matplotlib plots from Notebook 3.)*

---

## 🧩 Insights & Impact
- Demonstrates the **transferability of AI models** from perception tasks (image recognition) to **decision-support systems**.
- Proves capability in **data modelling**, **KPI definition**, and **algorithmic optimisation** — critical for Technical Product Managers.
- Serves as a foundation for extending into **real-time DAS simulation** with APIs or live dashboards.

---

## 🚀 Future Enhancements
- Integrate **Google OR-Tools** or **PuLP** for linear programming optimisation.
- Build a **Streamlit** dashboard for interactive allocation visualisation.
- Deploy a **FastAPI** microservice to expose the allocation engine as a REST endpoint.

---

## 🗣️ How to Talk About This Project (Interview)
> “I repurposed my AI Programming Nanodegree Image Classifier into a Dynamic Allocation Simulator that mirrors real-world optimisation engines. The model dynamically assigns resources to tasks, measures allocation KPIs, and demonstrates how AI logic can operationally scale — similar to what Dynamic Allocation Services delivers in live airport environments.”

---

## 📎 Author
**Tony Ofoh**  
Product Designer | Business Analyst | AI-Driven Product Manager  
https://tonyofoh.com · https://linkedin.com/in/tony-ofoh
