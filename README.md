# TEAM-PHISHERS
Resources used to create an AI/ML based defence decision making system 
mkdir Team_phishers_code
cd Team_phishers_code
git init
# ============================================================
# Imports
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gradio as gr
import matplotlib.pyplot as plt

# ============================================================
# Step 1: Generate 1000 Scenarios
# ============================================================
np.random.seed(42)

weather_opts   = ["Clear", "Rain", "Fog", "Storm"]
comms_opts     = ["Strong", "Weak", "Jammed"]
fatigue_opts   = ["Fresh", "Tired", "Exhausted"]
logistics_opts = ["On-time", "Delayed", "Disrupted"]
civilians_opts = ["None", "Low", "High"]

data = {
    'enemy_strength':   np.random.randint(10, 101, 1000),
    'own_strength':     np.random.randint(10, 101, 1000),
    'distance_km':      np.random.randint(1, 21, 1000),
    'ammo_level':       np.random.randint(0, 101, 1000),
    'terrain_advantage': np.random.choice(["No", "Yes"], 1000),
    'weather':          np.random.choice(weather_opts, 1000),
    'comms':            np.random.choice(comms_opts, 1000),
    'fatigue':          np.random.choice(fatigue_opts, 1000),
    'logistics':        np.random.choice(logistics_opts, 1000),
    'civilians':        np.random.choice(civilians_opts, 1000),
}

df = pd.DataFrame(data)

# Define rules for "action" (simplified heuristic)
def decide_action(row):
    if row['enemy_strength'] > row['own_strength'] and row['ammo_level'] < 30:
        return 0  # Retreat
    elif row['civilians'] != "None" and row['enemy_strength'] > row['own_strength']:
        return 1  # Observe
    elif row['fatigue'] == "Exhausted" or row['logistics'] == "Disrupted":
        return 1  # Observe
    else:
        return 2  # Engage

df['action'] = df.apply(decide_action, axis=1)

# ============================================================
# Step 2: Encode categorical values for ML
# ============================================================
encode_maps = {
    'terrain_advantage': {"No": 0, "Yes": 1},
    'weather': {w: i for i, w in enumerate(weather_opts)},
    'comms': {c: i for i, c in enumerate(comms_opts)},
    'fatigue': {f: i for i, f in enumerate(fatigue_opts)},
    'logistics': {l: i for i, l in enumerate(logistics_opts)},
    'civilians': {c: i for i, c in enumerate(civilians_opts)}
}

df_encoded = df.replace(encode_maps)

# Train model
X = df_encoded.drop('action', axis=1)
y = df_encoded['action']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# Step 3: Prediction Function
# ============================================================
def predict_action(enemy_strength, own_strength, distance_km, ammo_level, terrain_advantage,
                   weather, comms, fatigue, logistics, civilians):

  # Encode inputs
  new_data = pd.DataFrame([{
        'enemy_strength': enemy_strength,
        'own_strength': own_strength,
        'distance_km': distance_km,
        'ammo_level': ammo_level,
        'terrain_advantage': encode_maps['terrain_advantage'][terrain_advantage],
        'weather': encode_maps['weather'][weather],
        'comms': encode_maps['comms'][comms],
        'fatigue': encode_maps['fatigue'][fatigue],
        'logistics': encode_maps['logistics'][logistics],
        'civilians': encode_maps['civilians'][civilians]
    }])

  # Prediction
  prediction = model.predict(new_data)[0]
    actions = {
        0: "Retreat 🏳️ via safe route",
        1: "Hold & Observe 👀 until reinforcements",
        2: "Engage ⚔️ (consider air/artillery support)"
    }

  # Explanation
  explanation = []
    if enemy_strength > own_strength:
        explanation.append("Enemy stronger than own troops")
    if ammo_level < 30:
        explanation.append("Low ammunition")
    if fatigue == "Exhausted":
        explanation.append("Troops exhausted")
    if civilians != "None" and prediction == 2:
        explanation.append("Civilian risk if engaging")
    if not explanation:
        explanation.append("Conditions favorable")

  # Visualization
  fig, ax = plt.subplots(1, 2, figsize=(10, 4))

  # Bar chart (Force Comparison)
  ax[0].bar(["Enemy Strength", "Own Strength"], [enemy_strength, own_strength], color=['red','green'])
    ax[0].set_title("Force Comparison")

  # Radar chart (Readiness Factors)
   labels = ["Ammo", "Comms", "Fatigue", "Logistics"]
    values = [
        ammo_level/100,
        1 - encode_maps['comms'][comms]/2,
        1 - encode_maps['fatigue'][fatigue]/2,
        1 - encode_maps['logistics'][logistics]/2
    ]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

  ax[1] = plt.subplot(122, polar=True)
    ax[1].plot(angles, values, 'o-', linewidth=2)
    ax[1].fill(angles, values, alpha=0.25)
    ax[1].set_thetagrids(np.degrees(angles[:-1]), labels)
    ax[1].set_title("Battlefield Readiness")

  plt.tight_layout()

  return actions[prediction], "Reason: " + ", ".join(explanation), fig

# ============================================================
# Step 4: Gradio UI
# ============================================================
iface = gr.Interface(
    fn=predict_action,
    inputs=[
        gr.Slider(0, 100, value=65, label="Enemy Strength"),
        gr.Slider(0, 100, value=70, label="Own Strength"),
        gr.Slider(1, 20, value=4, label="Distance (km)"),
        gr.Slider(0, 100, value=60, label="Ammo Level"),
        gr.Radio(["No", "Yes"], value="Yes", label="Terrain Advantage"),
        gr.Radio(weather_opts, value="Clear", label="Weather"),
        gr.Radio(comms_opts, value="Strong", label="Communication Status"),
        gr.Radio(fatigue_opts, value="Fresh", label="Troop Fatigue"),
        gr.Radio(logistics_opts, value="On-time", label="Logistics"),
        gr.Radio(civilians_opts, value="None", label="Civilian Presence")
    ],
    outputs=["text", "text", "plot"],
    title="🪖 Battlefield Decision Support System",
    description="Advanced decision-making with explainable AI and visualization."
)

iface.launch()

git commit -m "Initial commit"
