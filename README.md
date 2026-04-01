# Public Transport Passenger Usage Analysis (2019 vs 2022)
This project explores public transport usage across **Bus, Tram,** and **Metro** services using datasets from **2019** and **2022.** It was completed as part of a major course module assessment on my MSc Data Science programme. This project covers data handling, exploratory analysis, seasonal trend smoothing, pricing behaviour, and visualisation using Python.

The goal of the project was to understand how travel patterns changed over time and to highlight key differences in behaviour before and after the pandemic.

---

## 📊 What the Project Covers

### **1. Annual and Seasonal Trends**
- Compared daily passenger totals for 2019 and 2022.
- Applied **Fourier smoothing** to reveal clear seasonal peaks.
- 2019 showed higher and more stable usage.
- 2022 displayed more variability, likely reflecting post‑pandemic travel patterns.

### **2. Weekly and Daily Patterns**
- 2019 had a strong weekday–weekend split, with weekdays dominating.
- In 2022, weekend travel increased, narrowing the gap.
- Suggests changes in work routines and leisure travel.

### **3. Ticket Pricing Behaviour (Metro)**
- Analysed Metro journeys from 2022.
- Found a strong linear relationship between **distance travelled** and **ticket price**.
- Regression model:  
  **Price ≈ €3.03 + €0.26 × Distance**
- Actual prices follow step‑based bands, but the model captures the overall trend.

### **4. Mode‑Specific Trends**
- Metro usage increased in 2022.
- Bus journey declined in 2022, while Tram stayed somewhat flat with respect to journeys from 2019.
- May reflect preference for faster or more direct transport options.

### **5. Weekend Totals (2019)**
- **Bus:** 134,616,936  
- **Tram:** 148,263,527  
- **Metro:** 188,894,993  

---

## 🧹 Data Description

### **2019 Dataset (`2019data3.csv`)**
Contains aggregated daily totals:
- Bus pax number (peak & off‑peak)
- Tram pax number (peak & off‑peak)
- Metro pax number (peak & off‑peak)
- Bus price (peak & off‑peak price)
- Tram price (peak & off‑peak price)
- Metro (peak & off‑peak price)
- Date

### **2022 Dataset (`2022data3.csv`)**
Contains journey by mode records:
- Date of journey
- Mode (Bus, Tram, Metro)
- Distance travelled
- Duration
- Ticket price

---

## 🛠️ Tools & Libraries Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scrfft** (for Fourier smoothing)
- **Scikit‑learn** (for regression)

---

## 📈 Skills Demonstrated
- Data handling and preprocessing  
- Exploratory data analysis  
- Fourier smoothing for seasonal patterns  
- Linear regression modelling  
- Visualisation and findings communication  
- Working to tight academic deadlines  

---

## Conclusion
This analysis shows a dynamic transportation environment with changing mode preferences and passenger behaviour. These observations can inform future planning, funding, and policy decisions to improve the effectiveness and accessibility of public transportation.
