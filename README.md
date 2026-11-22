
# 🎯 Particle Filter Simulation – Multi-Object Tracking Under Measurement Dropouts

### Probabilistic State Estimation Using Particle Filters, Gaussian Process Regression, Ensemble Prediction, and Forward–Backward Smoothing

This project investigates advanced **particle filtering and state estimation techniques** for tracking two moving objects under noisy and missing observations.

The simulation focuses on tracking the trajectories of two projectiles while handling periods of sensor measurement loss, also known as **measurement dropouts**.

Two approaches are implemented and evaluated:

1. **Particle Filter with Ensemble Pseudo-Measurements**
2. **Forward–Backward Particle Smoothing**

The first approach combines a particle filter with Gaussian Process Regression (GPR), polynomial regression, and linear regression to estimate missing observations.

The second approach uses particle filtering and backward trajectory reconstruction to estimate object trajectories from particle histories.

Both methods are implemented in Python and evaluated using simulated projectile motion.

## 🚀 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fadinrsultan/particle_filter_simulation/blob/main/particle_filter_simulation.ipynb)

**GitHub Repository:**

https://github.com/Fadinrsultan/particle_filter_simulation

## 🎯 Project Objectives

The main objectives of this project are to:

- Simulate the motion of two projectiles under gravitational acceleration.
- Develop particle filtering algorithms for multi-object tracking.
- Estimate hidden states from noisy position measurements.
- Handle temporary and extended observation dropouts.
- Investigate ensemble-based pseudo-measurement generation.
- Apply Gaussian Process Regression to trajectory prediction.
- Implement forward–backward particle smoothing.
- Compare trajectory estimation accuracy.
- Visualize estimated trajectories against ground-truth trajectories.

## ✨ Key Features

- **Two-Object Tracking:** Simultaneous estimation of the trajectories of two projectiles.
- **Particle Filtering:** Sequential Monte Carlo-based state estimation.
- **Measurement Dropout Simulation:** Models intervals with unavailable sensor observations.
- **Gaussian Process Regression:** Predicts missing position information.
- **Ensemble Pseudo-Measurements:** Combines multiple regression-based predictions.
- **Adaptive Uncertainty Scaling:** Increases uncertainty during extended dropouts.
- **Particle Resampling:** Resamples particles according to their estimated importance weights.
- **Forward–Backward Smoothing:** Reconstructs trajectories through particle ancestry.
- **Accuracy Evaluation:** Computes normalized trajectory estimation accuracy.
- **Visualization:** Compares true trajectories, noisy observations, and estimated paths.

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Simulation and algorithm development |
| NumPy | Numerical computations and matrix operations |
| Matplotlib | Trajectory visualization |
| Scikit-learn | Regression models |
| Gaussian Process Regression | Nonlinear trajectory prediction |
| RBF Kernel | Modeling smooth relationships |
| WhiteKernel | Modeling observation noise |
| Linear Regression | Linear trajectory extrapolation |
| Polynomial Regression | Nonlinear trajectory approximation |
| Particle Filtering | Probabilistic state estimation |
| Sequential Monte Carlo | Particle-based inference and resampling |
| Jupyter Notebook | Interactive experimentation |
| Google Colab | Notebook execution |

## 🏗️ System Architecture

The simulation compares two state-estimation approaches.

```text
                  START
                    |
                    v
          Simulate Two Projectiles
                    |
                    v
           Generate True States
                    |
                    v
          Generate Noisy Observations
                    |
                    v
          Introduce Measurement
                 Dropouts
                    |
          +---------+---------+
          |                   |
          v                   v
       METHOD 1            METHOD 2
          |                   |
          v                   v
    Particle Filter     Particle Filter
          |                   |
          v                   v
    Ensemble Models      Forward Filtering
          |                   |
     +----+----+              v
     |    |    |         Store Particle
     v    v    v            Histories
    GPR  Poly Linear          |
     |    |    |              v
     +----+----+        Trace Ancestors
          |                   |
          v                   v
    Generate Pseudo       Backward
      Measurements        Smoothing
          |                   |
          v                   v
    Update Particle      Reconstructed
        Weights           Trajectories
          |                   |
          v                   v
    Resampling          Smoothed Estimate
          |                   |
          +---------+---------+
                    |
                    v
            Accuracy Evaluation
                    |
                    v
          Trajectory Visualization
```

## 📐 Mathematical Model

The environment simulates two projectiles moving in two-dimensional space under gravitational acceleration.

### State Representation

The combined state vector is:

\[
X_t =
[x_1,y_1,v_{x1},v_{y1},
x_2,y_2,v_{x2},v_{y2}]^T
\]

Where:

- `x1, y1`: Position of projectile 1
- `vx1, vy1`: Velocity of projectile 1
- `x2, y2`: Position of projectile 2
- `vx2, vy2`: Velocity of projectile 2

The complete system contains eight state variables.

### Projectile Motion Equations

Horizontal motion:

\[
x_{t+1}=x_t+v_{x,t}\Delta t
\]

Vertical motion:

\[
y_{t+1}=y_t+v_{y,t}\Delta t
-\frac{1}{2}g\Delta t^2
\]

Horizontal velocity:

\[
v_{x,t+1}=v_{x,t}
\]

Vertical velocity:

\[
v_{y,t+1}=v_{y,t}-g\Delta t
\]

Where:

- `g = 9.81 m/s²`
- `Δt` is the simulation time step.

The simulation uses this motion model to generate ground-truth trajectories.

## 🔍 Method 1: Particle Filter with Ensemble Pseudo-Measurements

The first method combines particle filtering with machine learning regression techniques to estimate missing position measurements.

### Concept

During normal operation, the particle filter updates its state estimates using available noisy measurements.

During a measurement dropout, the algorithm attempts to compensate for missing observations by generating pseudo-measurements.

These estimates are obtained from an ensemble of regression models.

### Regression Models

**1. Gaussian Process Regression (GPR)**

Gaussian Process Regression is used to model relationships between time and observed positions.

The implementation incorporates:

- Radial Basis Function (RBF) kernel
- White noise kernel
- Historical position observations

GPR provides predictions that can help approximate object positions during missing observation intervals.

**2. Polynomial Regression**

Polynomial regression approximates the projectile trajectory using a polynomial function.

A second-degree polynomial is used in the experiment.

\[
p(t)=a_0+a_1t+a_2t^2
\]

This model is suitable for approximating the parabolic behavior of projectile motion.

**3. Linear Regression**

Linear regression provides an additional estimate based on recent position measurements.

It is included as part of the ensemble prediction process.

### Ensemble Pseudo-Measurements

The predictions from the three regression models are combined to construct estimated measurements.

These pseudo-measurements are used during dropout intervals to support particle weight updates.

### Adaptive Noise Scaling

The implementation increases measurement uncertainty as dropout duration grows.

This is intended to account for the increasing uncertainty associated with predicting object positions without real observations.

The approach includes configurable parameters for:

- Number of pseudo-measurement samples
- Noise growth interval
- Noise growth factor
- Historical observation window
- Polynomial degree

### Method 1 Configuration

| Parameter | Value |
|-----------|-------|
| Number of projectiles | 2 |
| State dimensions | 8 |
| Simulation steps | 100 |
| Number of particles | 100 |
| Time step | 0.1 s |
| Observation probability | 0.8 |
| Dropout start | Step 30 |
| Dropout end | Step 60 (exclusive) |
| Pseudo-measurement samples | 50 |
| Noise growth interval | 10 steps |
| Noise growth factor | 3.4 |
| Historical window size | 5 |
| Polynomial degree | 2 |
| Gravitational acceleration | 9.81 m/s² |

### Method 1 Results

The notebook reports:

| Projectile | Accuracy |
|------------|----------|
| Ball 1 | 98.66% |
| Ball 2 | 99.21% |

These results indicate low normalized trajectory estimation errors in the recorded simulation.

## 🔄 Method 2: Forward–Backward Particle Smoothing

The second method combines forward particle filtering with backward reconstruction of particle trajectories.

Unlike the ensemble approach, this method reconstructs trajectories using stored particles and their ancestry.

### Concept

The method consists of two main stages.

**Stage 1: Forward Particle Filtering**

The algorithm:

1. Initializes particles representing possible system states.
2. Propagates particles according to the motion model.
3. Incorporates available noisy observations.
4. Calculates particle importance weights.
5. Resamples particles.
6. Stores particle states and ancestry information.

**Stage 2: Backward Particle Smoothing**

After forward filtering, the algorithm reconstructs possible trajectories by tracing particle ancestors backward.

Multiple sampled trajectories are generated.

The final smoothed estimate is obtained by averaging these reconstructed trajectories.

### Particle State Estimation

The particle approximation represents uncertainty in the hidden state.

\[
p(x_t|z_{1:t})
\approx
\sum_{i=1}^{N}
w_t^{(i)}
\delta(x_t-x_t^{(i)})
\]

Where:

- `N` is the number of particles.
- `w` represents particle weights.
- `x` represents candidate states.
- `z` represents observations.

### Backward Smoothing

The algorithm samples particle lineages backward from the final filtering step.

The reconstructed trajectories are averaged to obtain the smoothed state estimate.

### Method 2 Configuration

| Parameter | Value |
|-----------|-------|
| Number of projectiles | 2 |
| State dimensions | 8 |
| Simulation steps | 100 |
| Number of particles | 500 |
| Time step | 0.1 s |
| Observation probability | 0.8 |
| Dropout start | Step 30 |
| Dropout end | Step 80 (exclusive) |
| Smoothing samples | 50 |
| Gravitational acceleration | 9.81 m/s² |

### Method 2 Results

The notebook reports:

| Projectile | Accuracy |
|------------|----------|
| Ball 1 | 98.82% |
| Ball 2 | 99.16% |

These results demonstrate accurate trajectory reconstruction under the simulated observation conditions.

## 📊 Comparison of Both Methods

The following table summarizes the recorded results.

| Metric | Method 1 | Method 2 |
|--------|----------|----------|
| Approach | Ensemble pseudo-measurements | Forward–backward smoothing |
| Gaussian Process Regression | Yes | No |
| Polynomial Regression | Yes | No |
| Linear Regression | Yes | No |
| Backward trajectory reconstruction | No | Yes |
| Number of particles | 100 | 500 |
| Dropout interval | Steps 30–59 | Steps 30–79 |
| Ball 1 accuracy | 98.66% | 98.82% |
| Ball 2 accuracy | 99.21% | 99.16% |

### Interpretation

Both implementations achieve high reported accuracy in the saved simulation outputs.

The ensemble approach uses regression-based predictions to support estimation during observation dropouts.

The forward–backward approach uses particle ancestry to reconstruct trajectories.

**Important:** The experiments use different particle counts and dropout durations. Consequently, the recorded accuracies do not establish that either method is generally superior.

A controlled comparison would require identical simulation settings and repeated experiments.

## 📏 Accuracy Evaluation

The notebook evaluates trajectory estimation using Root Mean Squared Error (RMSE), normalized by a trajectory range.

The reported accuracy follows the general form:

\[
Accuracy(\%)=
100\left(1-\frac{RMSE}{Range}\right)
\]

Where:

- `RMSE` measures the difference between estimated and ground-truth positions.
- `Range` provides the normalization factor.

This is a custom normalized accuracy measure used by the simulation.

It should not be interpreted as the probability that a prediction is correct.

## 📈 Visualization

Both methods generate trajectory plots illustrating the tracking results.

The visualizations compare:

- Ground-truth projectile trajectories
- Available noisy observations
- Estimated projectile trajectories

### Visualization Goals

The plots help assess:

- Tracking accuracy
- Behavior during measurement dropouts
- Differences between the two estimation approaches
- How closely estimated trajectories follow the simulated motion

The horizontal and vertical axes represent positions in meters.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Fadinrsultan/particle_filter_simulation.git

cd particle_filter_simulation
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy matplotlib scikit-learn notebook
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook:

```text
particle_filter_simulation.ipynb
```

Run the cells sequentially to execute both simulation methods.

### 5. Run in Google Colab

You can also launch the notebook directly using:

https://colab.research.google.com/github/Fadinrsultan/particle_filter_simulation/blob/main/particle_filter_simulation.ipynb

## 🧪 Experimental Workflow

The simulation follows these general steps:

1. Initialize the state of two projectiles.
2. Generate ground-truth trajectories using projectile motion equations.
3. Generate noisy position observations.
4. Introduce predefined measurement dropout intervals.
5. Initialize particles with state uncertainty.
6. Propagate particles using the motion model.
7. Update particle weights when measurements are available.
8. Apply the selected dropout-handling approach.
9. Resample particles.
10. Estimate projectile trajectories.
11. Compute normalized estimation accuracy.
12. Visualize results.

## ⚠️ Limitations

The implementation has several limitations:

- The trajectories are simulated rather than obtained from real sensors.
- The model assumes simplified projectile dynamics.
- Gravity is treated as constant.
- Air resistance is not modeled.
- Observations contain simulated Gaussian noise.
- Dropout intervals are predefined.
- Performance depends on particle count and noise settings.
- Random initialization and sampling can affect results.
- Regression-based extrapolation may become less reliable during prolonged dropouts.
- The forward–backward method requires stored particle histories and is intended for smoothing rather than purely online estimation.
- Reported accuracies are based on specific simulation runs and configurations.
- The two experiments do not use identical settings.

## 🔮 Future Improvements

Possible extensions include:

- Adaptive particle count selection
- Systematic and stratified resampling
- Improved particle proposal distributions
- Adaptive process and measurement noise estimation
- Multiple-object data association
- Real-time video-based tracking
- Integration with physical sensor measurements
- Extended Kalman Filter comparison
- Unscented Kalman Filter comparison
- More advanced smoothing algorithms
- Repeated Monte Carlo evaluation
- Confidence interval estimation
- Standardized comparison under identical dropout conditions
- Computational complexity and runtime analysis

## 🌐 Potential Applications

The techniques explored in this project are relevant to:

- Autonomous systems
- Robotics
- Object tracking
- Radar and sensor fusion
- Computer vision
- Navigation systems
- Trajectory prediction
- Probabilistic state estimation
- Tracking under intermittent observations

The present implementation is a simulation and would require further validation before use in real-world tracking systems.

## 📚 References

### Particle Filtering

**Doucet, A., de Freitas, N., & Gordon, N. (2001).**

*Sequential Monte Carlo Methods in Practice.*

Springer.

https://doi.org/10.1007/978-1-4757-3437-9

### Particle Filtering and Smoothing

**Doucet, A., & Johansen, A. M. (2011).**

*A Tutorial on Particle Filtering and Smoothing: Fifteen Years Later.*

https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

### Gaussian Process Regression

**Rasmussen, C. E., & Williams, C. K. I. (2006).**

*Gaussian Processes for Machine Learning.*

MIT Press.

https://gaussianprocess.org/gpml/

### Scientific Computing

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Scikit-learn Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

## 👨‍💻 Author

**Fadi Sultan**

AI & Machine Learning Developer

**GitHub Repository:**

https://github.com/Fadinrsultan/particle_filter_simulation

---

⭐ If you find this project useful, consider giving the repository a star.
