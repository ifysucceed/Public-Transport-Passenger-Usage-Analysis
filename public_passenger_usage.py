import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scrfft import scrfft

# General plot display settings
STUDENT_ID = "24159030"
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.linewidth"] = 1   # makes all spines bold
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.labelweight"] = "semibold"
plt.rcParams["axes.titleweight"] = "semibold"

# Loading datasets, using parse_dates to convert text string into date and datetime 
# Done to enable accurate and efficient date readings.
df19 = pd.read_csv("2019data3.csv", parse_dates=["Date"])
df22 = pd.read_csv("2022data3.csv", parse_dates=["Date and time"])

# Prepare 2019 daily totals
df19["Bus total"] = df19["Bus pax number peak"] + df19["Bus pax number offpeak"]
df19["Tram total"] = df19["Tram pax number peak"] + df19["Tram pax number offpeak"]
df19["Metro total"] = df19["Metro pax number peak"] + df19["Metro pax number offpeak"]

df19["Total passengers"] = df19[["Bus total", "Tram total", "Metro total"]].sum(axis=1)
df19["doy"] = df19["Date"].dt.dayofyear
df19["weekday"] = df19["Date"].dt.dayofweek

# Prepare 2022 daily totals with scaling
df22["date"] = pd.to_datetime(df22["Date and time"]).dt.date
df22["weekday"] = pd.to_datetime(df22["Date and time"]).dt.dayofweek
df22["doy"] = pd.to_datetime(df22["Date and time"]).dt.dayofyear

# Raw daily journey counts
daily22_counts = df22.groupby("doy").size().reset_index(name="journeys")

# Scale factor: total passengers (Bus+Tram+Metro) / total rows in dataset
total_target = 638225089 + 852502319 + 1119097445
scale_factor = total_target / len(df22)

# Scale daily counts to passengers
daily22_counts["Total passengers"] = daily22_counts["journeys"] * scale_factor
daily22_total = daily22_counts.copy()
daily22_total["weekday"] = df22.groupby("doy")["weekday"].first().values


def fourier_smooth(doy, y, K=8):
    """
    Smooth daily passenger data using scrfft coefficients.
    Uses all details from scrfft.
    """
    df = pd.DataFrame({"x": np.asarray(doy, dtype=float), "y": np.asarray(y, dtype=float)})
    df = df.dropna().sort_values("x")
    xdata = df["x"].values
    ydata = df["y"].values

    # Get scrfft outputs
    f, a, b = scrfft(xdata, ydata)

    # Uniform evaluation grid (same as scrfft)
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ndata = len(xdata)
    tt = (xmax - xmin) / (ndata - 1) * np.arange(ndata) + xmin

    # Truncated Fourier reconstruction
    y_hat = np.full_like(tt, fill_value=a[0], dtype=float)
    max_k = min(K, len(a) - 1)
    for k in range(1, max_k + 1):
        y_hat += a[k] * np.cos(2 * np.pi * f[k] * tt) + b[k] * np.sin(2 * np.pi * f[k] * tt)

    return tt, y_hat


# Figure 1: Scatter + Fourier smoothing
t19, yhat19 = fourier_smooth(df19["doy"], df19["Total passengers"])
t22, yhat22 = fourier_smooth(daily22_total["doy"], daily22_total["Total passengers"])

fig1, ax = plt.subplots()
ax.scatter(df19["doy"], df19["Total passengers"], s=10, color= "blueviolet", label="2019 daily", alpha=1)
ax.scatter(daily22_total["doy"], daily22_total["Total passengers"], s=10, color= "red", marker="x", label="2022 daily", alpha=0.75)
ax.plot(t19, yhat19, color="blueviolet", linewidth=2.5, label="2019 Fourier (8 terms)")
ax.plot(t22, yhat22, color="red", linewidth=2.5, label="2022 Fourier (8 terms)")
ax.set_xlabel("Day of year (1–365)")
ax.set_ylabel("Total Passengers")
ax.ticklabel_format(style='plain', axis='y')
ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', bottom=True, left=True)
ax.set_title(f"Figure 1 — Daily Totals and Fourier Smoothing (Student ID: {STUDENT_ID})")
ax.legend()
plt.show()


# Figure 2: Weekday averages + X,Y,Z
weekday_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
w19 = df19.groupby("weekday")["Total passengers"].mean().rename(index=weekday_map)
w22 = daily22_total.groupby("weekday")["Total passengers"].mean().rename(index=weekday_map)

fig2, ax = plt.subplots()
width = 0.35
x = np.arange(7)
ax.bar(x - width/2, w19.values, width, label="2019")
ax.bar(x + width/2, w22.values, width, label="2022")
ax.set_xticks(x, labels=list(weekday_map.values()))
ax.ticklabel_format(style='plain', axis='y')
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Daily Passengers")
ax.set_title(f"Figure 2 — Weekday Average (Student ID: {STUDENT_ID})")
ax.legend()
ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', bottom=True, left=True)


# 2019 Weekend totals X,Y,Z
weekend19 = df19[df19["weekday"].isin([5, 6])]
X = int(weekend19["Bus total"].sum())
Y = int(weekend19["Tram total"].sum())
Z = int(weekend19["Metro total"].sum())

txt = f"Weekend totals (2019): X=Bus {X}, Y=Tram {Y}, Z=Metro {Z}"
ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
plt.show()


# Figure 3: Metro price vs distance with fitting line
metro22 = df22[df22["Mode"]=="Metro"].dropna(subset=["Price", "Distance"])
X_lr = metro22[["Distance"]].values
y_lr = metro22["Price"].values
lr = LinearRegression().fit(X_lr, y_lr)
beta0 = lr.intercept_
beta1 = lr.coef_[0]
x_line = np.linspace(X_lr.min(), X_lr.max(), 200)
y_line = beta0 + beta1 * x_line

fig3, ax = plt.subplots()
ax.scatter(metro22["Distance"], metro22["Price"], s=10, alpha=0.6, label="Metro journeys (2022)")
ax.plot(x_line, y_line, color="red", linewidth=2.0, label=f"Fit: Price = {beta0:.2f} + {beta1:.2f}·Distance")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Price (EUR)")
ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', bottom=True, left=True)
ax.set_title(f"Figure 3 — Metro price vs distance (Student ID: {STUDENT_ID})")
ax.legend()
plt.show()


# Figure 4: Total journeys by mode 
mode19_totals = pd.Series({
    "Bus": int(df19["Bus total"].sum()),
    "Tram": int(df19["Tram total"].sum()),
    "Metro": int(df19["Metro total"].sum())
})

mode22_totals = pd.Series({
    "Bus": 638225089,
    "Tram": 852502319,
    "Metro": 1119097445
})

fig4, ax = plt.subplots()
labels = ["Bus", "Tram", "Metro"]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, [mode19_totals.get(m, 0) for m in labels], width, label="2019")
ax.bar(x + width/2, [mode22_totals.get(m, 0) for m in labels], width, label="2022")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.ticklabel_format(style='plain', axis='y')
ax.tick_params(axis='both', which='both', length=5, width=1, direction='out', bottom=True, left=True)
ax.set_xlabel("Mode of Transport")
ax.set_ylabel("Total journeys")
ax.set_title(f"Figure 4 — Total journeys by mode (Student ID: {STUDENT_ID})")
ax.legend()
plt.show()


print(f"Weekend totals 2019 — X (Bus): {X}, Y (Tram): {Y}, Z (Metro): {Z}")
print(f"Linear regression (Metro 2022): Price = {beta0:.2f} + {beta1:.2f} * Distance")

